import argparse
import logging
import sys
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="Run Metam data augmentation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)

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
        logger.info(f"Base dataset shape: {base_df.shape}")

        logger.info(f"Unique values in class column: {base_df[class_attr].unique()}")
        logger.info(f"Data types of all columns:\n{base_df.dtypes}")
        logger.info(
            f"Missing values in class column: {base_df[class_attr].isnull().sum()}"
        )

        joinable_options = join_path.get_join_paths_from_file(query_data, str(filepath))

        oracle = OracleFactory.create(config["oracle"]["type"])
        orig_metric = oracle.train(base_df, class_attr)

        if orig_metric is None:
            logger.error(
                f"Unable to train oracle. Please check if {class_attr} column exists and contains appropriate values."
            )
            sys.exit(1)

        logger.info(f"Original metric: {orig_metric}")

        new_col_lst = []
        for i, jp in enumerate(joinable_options):
            logger.info(f"Processing join path {i+1}/{len(joinable_options)}")
            try:
                df_l = pd.read_csv(path / jp.join_path[0].tbl, low_memory=False)
                df_r = pd.read_csv(path / jp.join_path[1].tbl, low_memory=False)

                if (
                    jp.join_path[1].col not in df_r.columns
                    or jp.join_path[0].col not in df_l.columns
                ):
                    logger.warning(
                        f"Skipping join due to missing columns: {jp.join_path[1].col} or {jp.join_path[0].col}"
                    )
                    continue

                for col in df_r.columns:
                    if (
                        col == jp.join_path[1].col
                        or jp.join_path[0].col == class_attr
                        or col == class_attr
                    ):
                        continue
                    jc = JoinColumn(
                        jp, df_r, col, base_df, class_attr, len(new_col_lst), uninfo
                    )
                    new_col_lst.append(jc)

            except Exception as e:
                logger.error(f"Error processing join path: {e}", exc_info=True)

        logger.info(f"Total join columns found: {len(new_col_lst)}")

        num_clusters = min(len(new_col_lst), config["metam"]["num_clusters"])
        centers, assignment, clusters = join_path.cluster_join_paths(
            new_col_lst, num_clusters, epsilon
        )
        logger.info(f"Number of clusters: {len(clusters)}")

        for k, cluster in enumerate(clusters):
            logger.info(f"Cluster {k}, size: {len(cluster)}")

        weights = profile_weights.initialize_weights(new_col_lst[0], {})

        augmented_df = querying.run_metam(
            len(centers),
            oracle,
            centers,
            theta,
            orig_metric,
            base_df,
            new_col_lst,
            weights,
            class_attr,
            clusters,
            assignment,
            uninfo,
            epsilon,
        )

        output_path = config["output"]["path"]
        augmented_df.to_csv(output_path, index=False)
        logger.info(f"Augmented data saved to '{output_path}'")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
