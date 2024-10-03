import copy
import math
import operator
import pickle
import profile
import random
import sys
from os import listdir
from os.path import isfile, join
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.feature_selection import mutual_info_classif

from . import dataset, group_helper, join_column, join_path, profile_weights


def run_metam(
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
):
    likelihood_num = [1 for _ in clusters]
    likelihood_den = [1 for _ in clusters]

    cluster_size = {
        k: sum(1 for j in assignment if assignment[j] == k)
        for k in range(len(clusters))
    }

    for k, size in cluster_size.items():
        print(f"Cluster {k}, size: {size}")

    stopping_criterion = 1000
    base_df = copy.deepcopy(initial_df)
    orig_metric = oracle.train(base_df, class_attr)

    with open("output.txt", "w") as fout:
        total_queries = 0
        it = 0
        grp_size = 1
        grp_queried_cand = {}
        overall_queried = {}

        max_iterations_without_improvement = 10
        iterations_without_improvement = 0


        while metric < theta and total_queries <= stopping_criterion and iterations_without_improvement < max_iterations_without_improvement:
            queried_cand = {}
            curr_max = metric
            curr_max_grp = metric
            max_candidate = initial_df
            max_candidate_grp = base_df

            while len(queried_cand) < tau and curr_max <= metric:
                if it == 0 or not queried_cand:
                    sorted_cand = profile_weights.sort_candidates(
                        new_col_lst, candidates, weights, overall_queried
                    )

                candidate_id = next(
                    (c for c, _ in sorted_cand if c not in queried_cand), None
                )
                if candidate_id is None:
                    break

                print(f"Chosen candidate in iteration {len(queried_cand)}")
                print(f"Candidate id and score: {candidate_id}, {sorted_cand[0][1]}")

                if sorted_cand[0][1] == 0:
                    break

                merged_df = copy.deepcopy(initial_df)
                merged_df[new_col_lst[candidate_id].column] = new_col_lst[
                    candidate_id
                ].merged_df[new_col_lst[candidate_id].column]
                tmp_metric = max(oracle.train(merged_df, class_attr), metric)

                print(
                    f"Iteration metric: {tmp_metric}, Table: {new_col_lst[candidate_id].join_path.join_path[1].tbl}, Column: {new_col_lst[candidate_id].join_path.join_path[1].col}"
                )

                queried_cand[candidate_id] = tmp_metric - metric
                total_queries += 1
                fout.write(f"{max(curr_max, curr_max_grp)} {total_queries}\n")

                if tmp_metric > curr_max:
                    curr_max = tmp_metric
                    max_candidate = merged_df
                    print(
                        f"New best metric: {tmp_metric}, Table: {new_col_lst[candidate_id].join_path.join_path[1].tbl}"
                    )
                    print(new_col_lst[candidate_id].profile_values)

                if tmp_metric > curr_max:
                    curr_max = tmp_metric
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1


            if len(grp_queried_cand) == len(new_col_lst):
                grp_size *= 2

            jc_lst, jc_representation = group_helper.identify_group_query(
                new_col_lst,
                clusters,
                grp_size,
                likelihood_num,
                likelihood_den,
                grp_queried_cand,
            )

            if not jc_lst:
                print("No more join columns to process. Exiting.")
                break

            grp_merged_df = copy.deepcopy(base_df)
            for jc in jc_lst:
                grp_merged_df[jc.column] = jc.merged_df[jc.column]

            tmp_metric = max(
                oracle.train(grp_merged_df, class_attr), orig_metric
            )
            if tmp_metric > orig_metric and len(jc_lst) == 1:
                candidates.append(jc_lst[0].loc)
            if len(jc_lst) == 1:
                queried_cand[jc_lst[0].loc] = tmp_metric - orig_metric

            grp_queried_cand[jc_representation] = tmp_metric
            total_queries += 1
            fout.write(f"{max(curr_max, curr_max_grp)} {total_queries}\n")

            if tmp_metric > curr_max_grp:
                curr_max_grp = tmp_metric
                max_candidate_grp = grp_merged_df
                print(
                    f"New best group metric: {tmp_metric}, Table: {new_col_lst[candidate_id].join_path.join_path[1].tbl}"
                )
                print(new_col_lst[candidate_id].profile_values)

            for jc in jc_lst:
                clust_id = assignment[jc]
                if tmp_metric > orig_metric:
                    likelihood_num[clust_id] += 1
                likelihood_den[clust_id] += 1

            if it == 0:
                weights = profile_weights.get_weights(
                    new_col_lst, base_df, queried_cand, weights, uninfo
                )

            weights = profile_weights.get_weights(
                new_col_lst, base_df, queried_cand, weights, uninfo
            )

            print(f"Length of candidates: {len(candidates)}")
            print(f"Best augmentation metric: {max(curr_max, curr_max_grp)}")

            for c in queried_cand:
                overall_queried[c] = queried_cand[c]

            if curr_max_grp > curr_max:
                metric = curr_max_grp
                initial_df = max_candidate_grp
                with open("log.txt", "a") as fout1:
                    fout1.write(
                        f"{new_col_lst[candidate_id].join_path.join_path[1].tbl};{new_col_lst[candidate_id].join_path.join_path[1].col} {metric}\n"
                    )
            elif curr_max > metric:
                metric = curr_max
                initial_df = max_candidate
                with open("log.txt", "a") as fout1:
                    fout1.write(
                        f"{new_col_lst[candidate_id].join_path.join_path[1].tbl};{new_col_lst[candidate_id].join_path.join_path[1].col} {metric}\n"
                    )

            it += 1

    return initial_df
