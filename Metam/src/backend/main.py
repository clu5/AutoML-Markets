import argparse
import cProfile
import logging
import multiprocessing
import pstats
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import src.backend.join_path as join_path
import src.backend.profile_weights as profile_weights
import src.backend.querying as querying
import yaml
from src.backend.join_column import JoinColumn
from src.backend.oracle_factory import OracleFactory

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_data(df):
    # Fill NaN values
    df = df.fillna(0)

    # Encode categorical variables
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category").cat.codes

    return df


def process_join_path(
    args: Tuple[str, Path, pd.DataFrame, str, int, int]
) -> List[JoinColumn]:
    jp, data_path, base_df, class_attr, array_loc, uninfo = args
    new_col_lst = []

    try:
        df_l = pd.read_csv(data_path / jp.join_path[0].tbl, low_memory=False)
        df_r = pd.read_csv(data_path / jp.join_path[1].tbl, low_memory=False)

        if (
            jp.join_path[1].col not in df_r.columns
            or jp.join_path[0].col not in df_l.columns
        ):
            return new_col_lst

        for col in df_r.columns:
            if (
                col == jp.join_path[1].col
                or jp.join_path[0].col == class_attr
                or col == class_attr
            ):
                continue
            jc = JoinColumn(
                jp, df_r, col, base_df, class_attr, array_loc + len(new_col_lst), uninfo
            )
            new_col_lst.append(jc)

    except Exception as e:
        print(f"Error processing join path: {e}")

    return new_col_lst


def parallel_process_join_paths(
    joinable_options, data_path, base_df, class_attr, uninfo
):
    num_processes = multiprocessing.cpu_count()  # Use all available CPU cores

    with multiprocessing.Pool(processes=num_processes) as pool:
        args = [
            (jp, data_path, base_df, class_attr, i * 1000, uninfo)
            for i, jp in enumerate(joinable_options)
        ]
        results = pool.map(process_join_path, args)

    return [
        item for sublist in results for item in sublist
    ]  # Flatten the list of lists


def main():
    parser = argparse.ArgumentParser(description="Run Metam data augmentation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        use_multiprocessing = config["metam"].get("use_multiprocessing", False)
        num_processes = config["metam"].get(
            "num_processes", multiprocessing.cpu_count()
        )

        path = Path(config["paths"]["data"]).resolve()
        query_data = config["query"]["data"]
        class_attr = config["query"]["class_attr"]
        query_path = path / query_data
        epsilon = config["metam"]["epsilon"]
        theta = config["metam"]["theta"]
        uninfo = config["metam"]["uninfo"]
        filepath = Path(config["paths"]["model"]).expanduser()

        logger.info(f"Loading base dataset from {query_path}")
        base_df = pd.read_csv(query_path)
        # Preprocess the initial dataframe
        base_df = preprocess_data(base_df)
        logger.info(f"Base dataset shape: {base_df.shape}")

        logger.info(f"Unique values in class column: {base_df[class_attr].unique()}")
        logger.info(f"Data types of all columns:\n{base_df.dtypes}")
        logger.info(
            f"Missing values in class column: {base_df[class_attr].isnull().sum()}"
        )

        max_iterations = config["metam"].get("max_iterations", float("inf"))
        max_join_paths = config["metam"].get("max_join_paths", float("inf"))

        joinable_options = join_path.get_join_paths_from_file(
            query_data, str(filepath)
        )[:max_join_paths]

        # After getting joinable_options
        logger.info(f"Number of joinable options found: {len(joinable_options)}")

        tpot_config = config["metam"].get("tpot", {})
        oracle = OracleFactory.create(config["oracle"]["type"], config=tpot_config)
        orig_metric = oracle.train(base_df, config["query"]["class_attr"])

        if orig_metric is None:
            logger.error(
                f"Unable to train oracle. Please check if {class_attr} column exists and contains appropriate values."
            )
            sys.exit(1)

        logger.info(f"Original metric: {orig_metric}")

        if use_multiprocessing:
            with multiprocessing.Pool(num_processes) as pool:
                new_col_lst = pool.starmap(
                    process_join_path,
                    [
                        (
                            jp,
                            Path(config["paths"]["data"]),
                            base_df,
                            config["query"]["class_attr"],
                            i * 1000,
                            config["metam"]["uninfo"],
                        )
                        for i, jp in enumerate(joinable_options)
                    ],
                )
            new_col_lst = [item for sublist in new_col_lst for item in sublist]
        else:
            new_col_lst = []
            for i, jp in enumerate(joinable_options):
                new_col_lst.extend(
                    process_join_path(
                        jp,
                        Path(config["paths"]["data"]),
                        base_df,
                        config["query"]["class_attr"],
                        i * 1000,
                        config["metam"]["uninfo"],
                    )
                )

        # new_col_lst = parallel_process_join_paths(
        #    joinable_options,
        #    Path(config["paths"]["data"]),
        #    base_df,
        #    config["query"]["class_attr"],
        #    config["metam"]["uninfo"],
        # )

        # After parallel processing
        logger.info(f"Number of new columns found: {len(new_col_lst)}")

        # Before clustering
        if len(new_col_lst) == 0:
            logger.warning(
                "No new columns found. Skipping clustering and further processing."
            )
            return base_df

        num_clusters = min(len(new_col_lst), config["metam"]["num_clusters"])
        centers, assignment, clusters = join_path.cluster_join_paths(
            new_col_lst, num_clusters, epsilon
        )
        logger.info(f"Number of clusters: {len(clusters)}")

        for k, cluster in enumerate(clusters):
            logger.info(f"Cluster {k}, size: {len(cluster)}")

        weights = profile_weights.initialize_weights(new_col_lst[0], {})

        # augmented_df = querying.run_metam(
        #    len(centers),
        #    oracle,
        #    centers,
        #    theta,
        #    orig_metric,
        #    base_df,
        #    new_col_lst,
        #    weights,
        #    class_attr,
        #    clusters,
        #    assignment,
        #    uninfo,
        #    epsilon,
        # )
        augmented_df = querying.run_metam(
            config["metam"]["tau"],
            oracle,
            candidates,
            theta,
            orig_metric,
            base_df,
            new_col_lst,
            weights,
            config["query"]["class_attr"],
            clusters,
            assignment,
            config["metam"]["uninfo"],
            epsilon,
        )

        output_path = config["output"]["path"]
        augmented_df.to_csv(output_path, index=False)
        logger.info(f"Augmented data saved to '{output_path}'")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(15)
