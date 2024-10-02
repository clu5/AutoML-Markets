import copy
import math
import operator
import os
import pickle
import random
import sys
import warnings
from distutils.ccompiler import new_compiler
from pathlib import Path

import pandas as pd
import src.backend.group_helper as group_helper
import src.backend.join_path as join_path
import src.backend.profile_weights as profile_weights
import src.backend.querying as querying
from sklearn import datasets, linear_model
from sklearn.feature_selection import mutual_info_classif
# Oracle implementation, any file containing Oracle class can be used as a task
from src.backend.classifier_oracle import Oracle
from src.backend.dataset import Dataset
from src.backend.join_column import JoinColumn
from src.backend.join_path import JoinKey, JoinPath


random.seed(0)

path = Path(
    "../aurum-datadiscovery/mimic_csvs/"
).resolve()  # Add the path to all datasets
query_data = "patients.csv"  # Add name of initial dataset
class_attr = "anchor_age"  # column name of prediction attribute
query_path = path / query_data


epsilon = 0.05  # Metam parameter
theta = 0.90  # Required utility

uninfo = (
    0  # Number of uninformative profiles to be added on top of default set of profiles
)

filepath = Path(
    "~/.aurum/models/mimic_model"
).expanduser()  # File containing all join paths


def main():
    try:
        base_df = pd.read_csv(query_path)
        print(f"Loaded base dataset from {query_path}")
        print(f"Base dataset shape: {base_df.shape}")

        print(f"Unique values in class column: {base_df[class_attr].unique()}")
        print(f"Data types of all columns:\n{base_df.dtypes}")
        print(
            f"Check no missing values in class column: {base_df[class_attr].isnull().sum()}"
        )

        joinable_options = join_path.get_join_paths_from_file(query_data, str(filepath))

        files = [f for f in path.iterdir() if f.is_file()]

        dataset_lst = []
        data_dic = {}

        oracle = Oracle("random forest")
        orig_metric = oracle.train_classifier(base_df, class_attr)
        # orig_metric=oracle.train_classifier(base_df,'class')

        if orig_metric is None:
            print(
                f"Error: Unable to train classifier. Please check if {class_attr} column exists and contains at least two classes."
            )
            sys.exit(1)

        print("original metric is ", orig_metric)

        i = 0
        new_col_lst = []
        skip_count = 0

        for i, jp in enumerate(joinable_options):
            print(f"Processing join path {i+1}/{len(joinable_options)}")
            try:
                df_l = pd.read_csv(path / jp.join_path[0].tbl, low_memory=False)
                df_r = pd.read_csv(path / jp.join_path[1].tbl, low_memory=False)

                if (
                    jp.join_path[1].col not in df_r.columns
                    or jp.join_path[0].col not in df_l.columns
                ):
                    print(
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
                print(f"Error processing join path: {e}")
                continue

        print(f"Total join columns found: {len(new_col_lst)}")

        # Adjust the number of clusters based on the number of join columns
        num_clusters = min(len(new_col_lst), 100)  # Limit to 20 clusters or less
        centers, assignment, clusters = join_path.cluster_join_paths(
            new_col_lst, num_clusters, epsilon
        )
        print(f"Number of clusters: {len(clusters)}")

        for k, cluster in enumerate(clusters):
            print(f"Cluster {k}, size: {len(cluster)}")

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

        augmented_df.to_csv("augmented_data.csv", index=False)
        print("Augmented data saved to 'augmented_data.csv'")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
