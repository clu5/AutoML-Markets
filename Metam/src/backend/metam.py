# src/backend/metam.py
import logging
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

from .join_path import get_join_paths_from_file, cluster_join_paths
from .join_column import JoinColumn
from .profile_weights import initialize_weights
from .querying import run_metam

logger = logging.getLogger(__name__)

class Metam:
    def __init__(self, config: Dict, oracle):
        self.config = config
        self.oracle = oracle

    def run(self, base_df: pd.DataFrame) -> pd.DataFrame:
        filepath = Path(self.config['paths']['model']).expanduser()
        joinable_options = get_join_paths_from_file(self.config['query']['data'], str(filepath))

        new_col_lst = self._process_join_paths(joinable_options, base_df)
        logger.info(f"Total join columns found: {len(new_col_lst)}")

        centers, assignment, clusters = self._cluster_join_paths(new_col_lst)

        weights = initialize_weights(new_col_lst[0], {})

        return run_metam(
            len(centers),
            self.oracle,
            centers,
            self.config['metam']['theta'],
            #self.oracle.train_classifier(base_df, self.config['query']['class_attr']),
            self.oracle.train(base_df, self.config['query']['class_attr']),
            base_df,
            new_col_lst,
            weights,
            self.config['query']['class_attr'],
            clusters,
            assignment,
            self.config['metam']['uninfo'],
            self.config['metam']['epsilon']
        )

    def _process_join_paths(self, joinable_options: List, base_df: pd.DataFrame) -> List[JoinColumn]:
        new_col_lst = []
        for jp in tqdm(joinable_options, desc="Processing join paths"):
            try:
                df_l = pd.read_csv(Path(self.config['paths']['data']) / jp.join_path[0].tbl, low_memory=False)
                df_r = pd.read_csv(Path(self.config['paths']['data']) / jp.join_path[1].tbl, low_memory=False)

                if jp.join_path[1].col not in df_r.columns or jp.join_path[0].col not in df_l.columns:
                    logger.warning(f"Skipping join due to missing columns: {jp.join_path[1].col} or {jp.join_path[0].col}")
                    continue

                for col in df_r.columns:
                    if col == jp.join_path[1].col or jp.join_path[0].col == self.config['query']['class_attr'] or col == self.config['query']['class_attr']:
                        continue
                    jc = JoinColumn(jp, df_r, col, base_df, self.config['query']['class_attr'], len(new_col_lst), self.config['metam']['uninfo'])
                    new_col_lst.append(jc)

            except Exception as e:
                logger.error(f"Error processing join path: {e}", exc_info=True)

        return new_col_lst

    def _cluster_join_paths(self, new_col_lst: List[JoinColumn]):
        num_clusters = min(len(new_col_lst), self.config['metam']['num_clusters'])
        centers, assignment, clusters = cluster_join_paths(new_col_lst, num_clusters, self.config['metam']['epsilon'])
        logger.info(f"Number of clusters: {len(clusters)}")
        for k, cluster in enumerate(clusters):
            logger.info(f"Cluster {k}, size: {len(cluster)}")
        return centers, assignment, clusters
