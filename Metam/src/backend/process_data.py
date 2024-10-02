"""
This file can be used to run for schools data with classification task.

For other tasks, please change line 19 to import respective Oracle, e.g. regression_oracle for Figure 3b
causal_oracle for Figure 3c.

"""

import copy
import math
import operator
import os
import pickle
import profile
import random
import sys
from distutils.ccompiler import new_compiler
from pathlib import Path

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.feature_selection import mutual_info_classif

import group_helper
import join_path
import profile_weights
import querying
from classifier_oracle import Oracle as ClassifierOracle
from dataset import Dataset
from join_column import JoinColumn
from join_path import JoinKey, JoinPath
from regression_oracle import Oracle as RegressionOracle

random.seed(0)

# path='/home/cc/opendata_cleaned/'#open_data_usa/' # path to csv files
# query_data='base_school.csv'
# class_attr='class'
# query_path=path+"/"+query_data
# model_path = '../../opendata_graph/'#'/Users/sainyam/Documents/MetamDemo/models/'#  # path to the graph model
# filepath='/home/cc/network_opendata_06.csv'


# Set up paths
path = Path(
    "../aurum-datadiscovery/mimic_csvs/"
).resolve()  # Add the path to all datasets
query_data = "patients.csv"  # Add name of initial dataset
class_attr = "anchor_age"  # column name of prediction attribute
query_path = path / query_data
model_path = "~/.aurum/models/mimic_model"
filepath = Path(
    "~/.aurum/models/mimic_model"
).expanduser()  # File containing all join paths

# Parameters
uninfo = 0
epsilon = 0.05
theta = 0.90


def detect_task_type(df, target_col):
    if df[target_col].dtype in ["int64", "float64"]:
        unique_values = df[target_col].nunique()
        if unique_values > 10:  # Arbitrary threshold, adjust as needed
            return "regression"
    return "classification"


def main():
    base_df = pd.read_csv(query_path)

    task_type = detect_task_type(base_df, class_attr)
    print(f"Detected task type: {task_type}")

    if task_type == "classification":
        oracle = ClassifierOracle("random forest")
    else:
        oracle = RegressionOracle("random forest")

    orig_metric = oracle.train_classifier(base_df, class_attr)
    print(f"Original {task_type} metric: {orig_metric}")

    options = join_path.get_join_paths_from_file(query_data, filepath)
    new_col_lst, skip_count = join_path.get_column_lst(
        options, path, base_df, class_attr, uninfo
    )

    print(f"Skip count: {skip_count}")
    print(f"Number of new columns: {len(new_col_lst)}")

    centers, assignment, clusters = join_path.cluster_join_paths(
        new_col_lst, 100, epsilon
    )
    print(f"Number of clusters: {len(clusters)}")

    tau = len(centers)
    weights = profile_weights.initialize_weights(new_col_lst[0], {})

    metric = orig_metric
    initial_df = copy.deepcopy(base_df)
    candidates = centers if tau > 1 else list(range(len(new_col_lst)))

    augmented_df = querying.run_metam(
        tau,
        oracle,
        candidates,
        theta,
        metric,
        initial_df,
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


if __name__ == "__main__":
    main()
