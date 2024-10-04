import copy
from functools import lru_cache
import math
import operator
import pickle
import profile
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from os import listdir
from os.path import isfile, join

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier  # We'll use TPOT for AutoML

from . import dataset, group_helper, join_column, join_path, profile_weights


class Exp3:
    def __init__(self, n_arms, gamma):
        self.n_arms = n_arms
        self.gamma = gamma
        self.weights = np.ones(n_arms)

    def select_arm(self):
        probs = self.distr()
        return np.random.choice(self.n_arms, p=probs)

    def distr(self):
        d = (1 - self.gamma) * self.weights / np.sum(self.weights)
        return d + self.gamma / self.n_arms

    def update(self, arm, reward):
        self.weights[arm] *= np.exp(self.gamma * reward / self.n_arms)


@lru_cache(maxsize=None)
def cached_sort_candidates(new_col_lst, candidates, weights, overall_queried):
    return profile_weights.sort_candidates(new_col_lst, candidates, weights, overall_queried)


def run_metam_online(
    oracle,
    candidates,
    theta,
    metric,
    initial_df,
    new_col_lst,
    class_attr,
    epsilon,
    max_iterations=1000,
):
    base_df = initial_df.copy()
    current_metric = metric

    # Initialize Exp3 for join path selection
    exp3 = Exp3(len(candidates), gamma=0.1)

    # Initialize AutoML (TPOT)
    tpot = TPOTClassifier(
        generations=5, population_size=20, cv=5, random_state=42, verbosity=2
    )

    for iteration in range(max_iterations):
        # Select a join path using Exp3
        selected_arm = exp3.select_arm()
        candidate_id = candidates[selected_arm]

        # Apply the selected join
        merged_df = base_df.copy()
        merged_df[new_col_lst[candidate_id].column] = new_col_lst[
            candidate_id
        ].merged_df[new_col_lst[candidate_id].column]

        # Split the data
        X = merged_df.drop(columns=[class_attr])
        y = merged_df[class_attr]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Run AutoML
        tpot.fit(X_train, y_train)
        tmp_metric = tpot.score(X_test, y_test)

        # Calculate reward
        reward = max(0, tmp_metric - current_metric)

        # Update Exp3
        exp3.update(selected_arm, reward)

        # Update current metric and dataset if improved
        if tmp_metric > current_metric:
            current_metric = tmp_metric
            base_df = merged_df
            print(f"New best metric: {current_metric}")
            print(
                f"Selected join: {new_col_lst[candidate_id].join_path.join_path[1].tbl}"
            )

        # Check stopping criterion
        if current_metric >= theta or iteration == max_iterations - 1:
            break

    return base_df, current_metric


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
    if not new_col_lst:
        logger.warning("No new columns found. Returning original dataset.")
        return initial_df

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

        while (
            metric < theta
            and total_queries <= stopping_criterion
            and iterations_without_improvement < max_iterations_without_improvement
        ):
            queried_cand = {}
            curr_max = metric
            curr_max_grp = metric
            max_candidate = initial_df
            max_candidate_grp = base_df

            while len(queried_cand) < tau and curr_max <= metric:
                if it == 0 or not queried_cand:
                    #sorted_cand = profile_weights.sort_candidates(
                    #    new_col_lst, candidates, weights, overall_queried
                    #)
                    sorted_cand = cached_sort_candidates(
                        tuple(new_col_lst), tuple(candidates), tuple(weights.items()), tuple(overall_queried.items())
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
                #tmp_metric = max(oracle.train(merged_df, class_attr), metric)
                tmp_metric = oracle.train(merged_df, class_attr)

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

            tmp_metric = max(oracle.train(grp_merged_df, class_attr), orig_metric)
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
