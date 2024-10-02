import copy
import math
import random
from audioop import cross
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from numpy import nan as NaN
from scipy.stats import chi2_contingency, chisquare
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import adjusted_mutual_info_score


class JoinColumn:
    def __init__(
        self, join_path, df, column, base_df, class_attr, array_loc, uninformative=0
    ):
        self.join_path = join_path
        self.loc = array_loc
        self.column = column
        self.orig_name = column
        self.base_copy = base_df.copy()
        self.df = df.drop_duplicates(
            subset=self.join_path.join_path[1].col, keep="first"
        ).copy()
        self.profiles = {
            "sim": self.syntactic,
            "corr": self.corr,
            "mutual": self.syntactic,
        }
        self.class_attr = class_attr
        self.key_r = self.join_path.join_path[1].col

        if column in base_df.columns:
            new_name = f"{column}_new"
            self.df = self.df.rename(columns={column: new_name})
            self.column = new_name

        if self.key_r in base_df.columns:
            new_name = f"{self.key_r}_new"
            self.df = self.df.rename(columns={self.key_r: new_name})
            self.key_r = new_name

        try:
            self.merged_df = pd.merge(
                self.base_copy,
                self.df[[self.key_r, self.column]],
                left_on=self.join_path.join_path[0].col,
                right_on=self.key_r,
                how="left",
            )
        except:
            self.merged_df = self.base_copy.copy()
            self.merged_df[self.column] = 0

        self.copied_df = self.merged_df.copy()

        for c in self.copied_df.select_dtypes(include=["object"]).columns:
            self.copied_df[c] = self.copied_df[c].astype("category").cat.codes

        self.profile_values = self.calculate_profiles(uninformative)

    def calculate_profiles(self, uninformative):
        profile_values = {
            "mutual": {},
            "corr": {},
            "nan": {
                "all": 1
                - self.merged_df[self.column].isna().sum() / len(self.merged_df)
            },
            "sim": {},
            "uninfo": {str(i): random.random() for i in range(uninformative)},
        }

        self.copied_df[self.column] = self.copied_df[self.column].fillna(-1)
        corr_values = self.copied_df.corr()[self.column]

        for c in self.copied_df.columns:
            if c == self.column:
                continue
            self.copied_df[c] = self.copied_df[c].fillna(-1)
            profile_values["corr"][c] = (
                corr_values[c] if not np.isnan(corr_values[c]) else 0
            )
            profile_values["mutual"][c] = adjusted_mutual_info_score(
                self.copied_df[c][:100], self.copied_df[self.column][:100]
            )
            profile_values["sim"][c] = self.syntactic(c)

        return profile_values

    def syntactic(self, col):

        return SequenceMatcher(None, col, self.orig_name).ratio()

    def corr(self, col):

        # print (self.merged_df[[col,self.column]][:10])
        cross_tab = pd.crosstab(
            self.merged_df[col][:4], self.merged_df[self.column][:4]
        )
        # print (cross_tab)

        try:
            chi2, p, dof, ex = chi2_contingency(cross_tab)
        except:
            return 0
        # print (chi2,p)
        # print (self.merged_df.corr(method='pearson', min_periods=1))

        if p < 0.1:
            print(cross_tab)
            print(chi2, p)
            print(self.merged_df[[col, self.column]][:10])
            return chi2
        else:
            return 0

    def get_distance(self, jc2):
        dist = 0
        # print (self.profile_values)
        # print (jc2.profile_values)
        # print (self.base_copy)
        for prof in self.profile_values.keys():
            prof_dic = self.profile_values[prof]
            for col in prof_dic.keys():  # self.base_copy.columns:
                curr = abs(
                    abs(self.profile_values[prof][col])
                    - abs(jc2.profile_values[prof][col])
                )
                if curr > dist:
                    dist = curr
        return dist

        # Option 1: correlation between columns
        try:
            corr = self.dataset.df[self.column].corr(jc2.dataset.df[jc2.column])
        except:
            corr = 0
        return 1 - abs(corr)
