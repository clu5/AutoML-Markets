import logging
import pickle
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
# field_path = Path('../aurum-datadiscovery/knowledgerepr')
# print(field_path.resolve())
# sys.path.append(field_path)
from knowledgerepr import fieldnetwork

from . import join_column

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def cached_get_fields_of_source(network, source):
    return network.get_fields_of_source(source)


@lru_cache(maxsize=None)
def cached_neighbors_id(network, field, relation):
    return network.neighbors_id(field, relation)


def get_join_paths_from_file(query_data, model_path):
    network, schema_sim_index, content_sim_index = load_aurum_network(model_path)

    logger.info(f"Loaded network with {network.graph_order()} nodes")
    logger.info(f"Number of tables in the network: {network.get_number_tables()}")
    logger.info(f"Sample of node IDs: {list(network.iterate_ids())[:5]}")
    logger.info(
        f"Sample of field info: {[network.get_info_for([nid]) for nid in list(network.iterate_ids())[:5]]}"
    )

    # Use the correct attribute name
    graph = network._FieldNetwork__G
    logger.info(f"Network loaded. Nodes: {len(graph)}, Edges: {len(graph.edges())}")

    join_paths = []
    query_fields = network.get_fields_of_source(query_data)
    logger.info(f"Found {len(query_fields)} fields for {query_data}")

    for field in query_fields:
        pkfk_neighbors = network.neighbors_id(field, fieldnetwork.Relation.PKFK)
        logger.info(f"Found {len(pkfk_neighbors)} PKFK neighbors for field {field}")

        for neighbor in pkfk_neighbors:
            jk1 = JoinKey("", "", 0, 0)
            jk2 = JoinKey("", "", 0, 0)

            jk1.tbl = query_data
            jk1.col = network.get_info_for([field])[0][3]  # field name

            jk2.tbl = neighbor.source_name
            jk2.col = neighbor.field_name

            join_paths.append(JoinPath([jk1, jk2]))

    logger.info(f"Retrieved {len(join_paths)} join paths from Aurum")
    return join_paths



def get_join_paths_from_aurum(network, query_data):
    logger.info(f"Searching for join paths for query data: {query_data}")
    options = []

    # Get all source names in the network
    all_sources = list(network._get_underlying_repr_table_to_ids().keys())
    logger.info(f"All sources in the network: {all_sources}")

    # Try to find the query_data in the network
    matching_sources = [s for s in all_sources if query_data.lower() in s.lower()]
    logger.info(f"Matching sources: {matching_sources}")

    if not matching_sources:
        logger.warning(f"No exact match found for {query_data}. Using all sources.")
        matching_sources = all_sources

    for source in matching_sources:
        # Get all fields for the query table
        query_fields = network.get_fields_of_source(source)
        logger.info(f"Found {len(query_fields)} fields for {source}")
        logger.info(f"Fields: {query_fields}")

        for field in query_fields:
            # Get PKFK relationships
            pkfk_neighbors = network.neighbors_id(field, fieldnetwork.Relation.PKFK)

            # Handle DRS object
            if isinstance(pkfk_neighbors, api.apiutils.DRS):
                pkfk_neighbors = pkfk_neighbors.data

            logger.info(f"Found {len(pkfk_neighbors)} PKFK neighbors for field {field}")

            for neighbor in pkfk_neighbors:
                field_info = network.get_info_for([field])[0]
                jk1 = JoinKey(source, field_info[3])  # field name
                jk2 = JoinKey(neighbor.source_name, neighbor.field_name)
                options.append(JoinPath([jk1, jk2]))
                logger.info(f"Added join path: {options[-1].to_str()}")

    logger.info(f"Found {len(options)} join paths")
    return options

def load_aurum_network(path):
    path = Path(path).expanduser()
    logger.info(f"Loading Aurum network from: {path}")

    try:
        with open(path / "graph.pickle", "rb") as f:
            G = pickle.load(f)
        logger.info(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    except Exception as e:
        logger.error(f"Failed to load graph.pickle: {str(e)}")
        return None, None, None

    try:
        with open(path / "id_info.pickle", "rb") as f:
            id_to_info = pickle.load(f)
        logger.info(f"Loaded id_info with {len(id_to_info)} entries")
    except Exception as e:
        logger.error(f"Failed to load id_info.pickle: {str(e)}")
        return None, None, None

    try:
        with open(path / "table_ids.pickle", "rb") as f:
            table_to_ids = pickle.load(f)
        logger.info(f"Loaded table_ids with {len(table_to_ids)} entries")
        logger.info(f"Table names in the model: {list(table_to_ids.keys())}")
    except Exception as e:
        logger.error(f"Failed to load table_ids.pickle: {str(e)}")
        return None, None, None

    network = fieldnetwork.FieldNetwork(G, id_to_info, table_to_ids)

    # Print more information about the loaded model
    logger.info(f"Number of nodes in the network: {network.graph_order()}")
    logger.info(f"Number of tables in the network: {network.get_number_tables()}")

    # Print sample of table names and their fields
    sample_tables = list(table_to_ids.keys())[:5]  # Print first 5 tables
    for table in sample_tables:
        fields = network.get_fields_of_source(table)
        logger.info(f"Table: {table}, Fields: {fields}")

    # Print sample of PKFK relationships
    logger.info("Sample of PKFK relationships:")
    for node in list(G.nodes())[:5]:  # Print PKFK for first 5 nodes
        pkfk_neighbors = network.neighbors_id(node, fieldnetwork.Relation.PKFK)
        if isinstance(pkfk_neighbors, api.apiutils.DRS):
            pkfk_neighbors = pkfk_neighbors.data
        logger.info(f"Node: {node}, PKFK neighbors: {[n.nid for n in pkfk_neighbors]}")

    return network, None, None



def get_column_lst(joinable_lst):
    i = 0
    skip_count = 0
    new_col_lst = []
    while i < len(joinable_lst):
        print(i, len(new_col_lst))
        jp = joinable_lst[i]
        print(
            jp.join_path[0].tbl,
            jp.join_path[0].col,
            jp.join_path[1].tbl,
            jp.join_path[1].col,
        )
        if jp.join_path[1].tbl in ignore_lst or jp.join_path[0].tbl in ignore_lst:
            i += 1
            continue
        if (
            jp.join_path[1].tbl == "s27g-2w3u.csv"
            or jp.join_path[0].tbl == "s27g-2w3u.csv"
        ):
            skip_count += 1
            i += 1
            continue
        if (
            jp.join_path[0].tbl in size_dic.keys()
            and jp.join_path[1].tbl in size_dic.keys()
        ):
            if (
                size_dic[jp.join_path[0].tbl] > 1000000
                or size_dic[jp.join_path[1].tbl] > 1000000
            ):
                skip_count += 1
                i += 1
                continue

        if jp.join_path[0].tbl not in data_dic.keys():
            df_l = pd.read_csv(path + "/" + jp.join_path[0].tbl, low_memory=False)
            data_dic[jp.join_path[0].tbl] = df_l
            print("dataset size is ", df_l.shape)
        else:
            df_l = data_dic[jp.join_path[0].tbl]
        if jp.join_path[1].tbl not in data_dic.keys():
            df_r = pd.read_csv(path + "/" + jp.join_path[1].tbl, low_memory=False)
            data_dic[jp.join_path[1].tbl] = df_r
            print("dataset size is ", df_r.shape)
        else:
            df_r = data_dic[jp.join_path[1].tbl]
        collst = list(df_r.columns)
        if (
            jp.join_path[1].col not in df_r.columns
            or jp.join_path[0].col not in df_l.columns
        ):
            i += 1
            continue
        if (
            df_r.dtypes[jp.join_path[1].col] == "float64"
            or df_r.dtypes[jp.join_path[1].col] == "int64"
        ):
            skip_count += 1
            i += 1
            continue
        for col in collst:
            if (
                jp.join_path[1].tbl == "2013_NYC_School_Survey.csv"
                or jp.join_path[1].tbl == "5a8g-vpdd.csv"
            ):
                continue
            if (
                col == jp.join_path[1].col
                or jp.join_path[0].col == "class"
                or col == "class"
            ):
                continue
            jc = join_column.JoinColumn(
                jp, df_r, col, base_df, class_attr, len(new_col_lst), uninfo
            )
            new_col_lst.append(jc)
            if (
                jc.column == "School Type" and jp.join_path[1].tbl == "bnea-fu3k.csv"
            ):  # 2012-2013 ENVIRONMENT GRADE':# and jp.join_path[1].tbl=='test1.csv':
                f1 = open("log.txt", "a")
                f1.write(
                    str(len(new_col_lst) - 1)
                    + " "
                    + jc.column
                    + " "
                    + jc.join_path.join_path[1].tbl
                    + " "
                    + jc.join_path.join_path[1].col
                    + "\n"
                )
                print(col, jc.merged_df, len(new_col_lst) - 1, "test1")
                f1.close()
        i += 1
    return (new_col_lst, skip_count)


class JoinPath:
    def __init__(self, join_key_list):
        self.join_path = join_key_list

    def to_str(self):
        format_str = ""
        for i, join_key in enumerate(self.join_path):
            format_str += join_key.tbl[:-4] + "." + join_key.col
            if i < len(self.join_path) - 1:
                format_str += " JOIN "
        return format_str

    def set_df(self, data_dic):
        for i, join_key in enumerate(self.join_path):
            join_key.dataset = data_dic[join_key.tbl]

    def print_metadata_str(self):
        print(self.to_str())
        for join_key in self.join_path:
            print(join_key.tbl[:-4] + "." + join_key.col)
            print(
                "datasource: {}, unique_values: {}, non_empty_values: {}, total_values: {}, join_card: {}, jaccard_similarity: {}, jaccard_containment: {}".format(
                    join_key.tbl,
                    join_key.unique_values,
                    join_key.total_values,
                    join_key.non_empty,
                    get_join_type(join_key.join_card),
                    join_key.js,
                    join_key.jc,
                )
            )

    def get_distance(self, join_path2):
        # return distance between the join paths

        return 0


class JoinKey:
    def __init__(self, col_drs, unique_values, total_values, non_empty):
        self.dataset = ""
        try:
            self.tbl = col_drs.source_name
            self.col = col_drs.field_name
        except:
            self.tbl = ""
            self.col = ""

        self.unique_values = unique_values
        self.total_values = total_values
        self.non_empty = non_empty
        try:
            if col_drs.metadata == 0:
                self.join_card = 0
                self.js = 0
                self.jc = 0
            else:
                self.join_card = col_drs.metadata["join_card"]
                self.js = col_drs.metadata["js"]
                self.jc = col_drs.metadata["jc"]
        except:
            self.js = 0


def get_join_type(join_card):
    if join_card == 0:
        return "One-to-One"
    elif join_card == 1:
        return "One-to-Many"
    elif join_card == 2:
        return "Many-to-One"
    else:
        return "Many-to-Many"


def find_farthest(distance_dic):
    max_dist = -1
    max_dis_index = -1
    for index in distance_dic.keys():
        if distance_dic[index] > max_dist:
            max_dist = distance_dic[index]
            max_dist_index = index

    print(max_dist, max_dist_index)
    return max_dist_index


def get_clusters(assignment, k):
    clusters = []
    i = 0
    while i < k:
        clusters.append([])
        i += 1

    for c in assignment.keys():
        lst = clusters[assignment[c]]
        lst.append(c)
        clusters[assignment[c]] = lst
    return clusters


def cluster_join_paths(joinable_lst, k, epsilon):
    i = 0
    random.seed(0)
    centers = []
    assignment = {}
    distance = {}
    max_dist = 0
    while i < k:
        if i == 0:
            centers.append(random.randint(0, len(joinable_lst)))
        else:
            centers.append(find_farthest(distance))
        # Assignment
        iter = 0
        for j in joinable_lst:
            if i == 0:
                assignment[j] = 0
                distance[iter] = j.get_distance(joinable_lst[centers[-1]])
                if distance[iter] > max_dist:
                    max_dist = distance[iter]
            else:
                new_dist = j.get_distance(joinable_lst[centers[-1]])
                if (
                    new_dist < distance[iter]
                ):  # j.get_distance(joinable_lst[centers[assignment[j]]]):
                    assignment[j] = len(centers) - 1
                    distance[iter] = (
                        new_dist  # j.get_distance(joinable_lst[centers[-1]])
                    )
                    if distance[iter] > max_dist:
                        max_dist = distance[iter]
            iter += 1
            # update assignment
        if max_dist < epsilon:
            break
        i += 1
    return (centers, assignment, get_clusters(assignment, k))


# def get_join_paths_from_file(query_data, join_paths):
#    network, schema_sim_index, content_sim_index = load_aurum_network(join_paths)
#    # logger.info(f"Network loaded. Nodes: {len(network.nodes())}, Edges: {len(network.edges())}")
#    paths = get_join_paths_from_aurum(network, query_data)
#    logger.info(f"Retrieved {len(paths)} join paths from Aurum")
#    return paths


# def get_join_paths_from_file(querydata,filepath):
#    df=pd.read_csv(filepath)
#
#    subdf=df[df['tbl1']==querydata]
#    subdf2=df[df['tbl2']==querydata]
#
#    options=[]
#
#    for index,row in subdf.iterrows():
#        jk1=JoinKey('','',0,0)
#        jk2=JoinKey('','',0,0)
#        jk1.tbl=row['tbl1']
#        jk1.col=row['col1']
#
#        jk2.tbl=row['tbl2']
#        jk2.col=row['col2']
#        ret_jp = JoinPath([jk1,jk2])
#        options.append(ret_jp)
#
#
#    for index,row in subdf2.iterrows():
#        jk1=JoinKey('','',0,0)
#        jk2=JoinKey('','',0,0)
#        jk1.tbl=row['tbl1']
#        jk1.col=row['col1']
#
#        jk2.tbl=row['tbl2']
#        jk2.col=row['col2']
#        ret_jp = JoinPath([jk2,jk1])
#        options.append(ret_jp)
#
#
#    return options
