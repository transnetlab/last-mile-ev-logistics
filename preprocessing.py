"""
This file preprocesses the Amazon data for the Local Search algorithm
"""
import urllib
import pickle
import requests
import warnings
import networkx
import json, math
import osmnx as ox
import pandas as pd
import networkx as nx
import seaborn as sns
import multiprocessing
from tqdm import tqdm
from time import time, sleep
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from helpers.loggers import log_line
from helpers.graphBuilder import get_route_information, shift_bbox, read_dataset, get_primal_edges_closest_to_delivery_location, check_negative_energy_self_cycles
from helpers.functions import process_dual_graph, custom_shuffle

ox.settings.use_cache = True
sns.set_style("whitegrid")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")


def read_energy_file(energies_file: str, route_num: int, plot_energies_histogram: bool) -> pd.DataFrame:
    """
    This function reads the energy file and returns the corresponding pandas dataframe.

    Args:
        energies_file (str): File name
        route_num (int): Route number
        plot_energies_histogram (bool): Whether to plot the histogram of energies

    Returns:
        austin_df (pd.DataFrame): Pandas dataframe of the energy dump file
    """
    file = energies_file + "_" + "dual" + "_" + str(route_num) + ".csv"
    austin_df = pd.read_csv(file).drop_duplicates().reset_index(drop=True)
    austin_df.v1 = austin_df.v1.apply(lambda x: eval(x))
    austin_df.v2 = austin_df.v2.apply(lambda x: eval(x))

    if plot_energies_histogram:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        lb = math.floor(min(austin_df['Energy']))
        ub = math.ceil(max(austin_df['Energy']))
        bins = ub - lb
        n, bins, patches = ax.hist(austin_df['Energy'], bins=bins, range=(lb, ub), color='lightblue')

        # add text annotations just above each bin
        for i in range(len(patches)):
            # bin_x = patches[i].get_x()
            bin_y = patches[i].get_height()
            bin_center = patches[i].get_x() + patches[i].get_width() / 2
            ax.text(bin_center, bin_y + max(n) * 0.01, f"{int(bin_y)} ({bin_y / len(austin_df) * 100:.1f}%)",
                    ha='center', va='bottom', color='black', rotation=90)

        ax.set_xlim(min(bins), max(bins))
        ax.set_ylim(0, max(n) * 1.6)

        ax.set_title('Austin Energy Distribution')

        # count the number of positive and negative values
        positive_count = (austin_df['Energy'] > 0).sum()
        negative_count = (austin_df['Energy'] < 0).sum()

        # print the counts with labels
        print(f"Positive values: {positive_count * 100 / austin_df.shape[0]}")
        print(f"Negative values: {negative_count * 100 / austin_df.shape[0]}")

        ax.set_xlabel('Energy Value')
        ax.set_ylabel('Frequency')
        plt.show()
        plt.clf()
    return austin_df

def get_coordinates(G: networkx.classes.multidigraph.MultiDiGraph, node_id: int) -> tuple:
    """
    Given a node in the graph, this function gives the latitude and longitude corresponding to that node.

    Args:
        G (networkx.classes.multidigraph.MultiDiGraph): Primal Graph
        node_id (int): Node ID

    Returns:
        (lon, lat): Tuple of latitude and longitude coordinates of the node
    """
    lat = G.nodes()[node_id]['y']
    lon = G.nodes()[node_id]['x']
    return lon, lat


def get_cartesian_coord_for_lat_long(lon: float, lat: float) -> tuple:
    """
    Given the longitude and latitude associated with a location, this function gives the cartesian (x,y) coordinates associated with that location while assuming a spherical Earth with a specific radius.

    Args:
        lon (float): Longitude
        lat (float): Latitude

    Returns:
        (x, y): Tuple of Cartesian coordinates
    """
    radius = 6371
    lon_rad = math.radians(lon)
    lat_rad = math.radians(lat)
    x = radius * math.cos(lat_rad) * math.cos(lon_rad)
    y = radius * math.cos(lat_rad) * math.sin(lon_rad)
    return x, y


def get_turn_type(G: networkx.classes.multidigraph.MultiDiGraph, u: int, v: int, w: int, tolerance: float, pred_count: int, success_count: int) -> int:
    """
    Given the latitudes and longitudes associated with three nodes, this function returns a string indicating whether the direction at the intersection is left, right or straight.

    Args:
        G (networkx.classes.multidigraph.MultiDiGraph): Primal Graph
        u (int): Node ID
        v (int): Node ID
        w (int): Node ID
        tolerance (float): Tolerance for the angle between the vectors ab and bc
        pred_count (int): Number of predecessors
        success_count (int): Number of successors

    Returns:
        cost (int): 1 or 0
    """
    # v, tolerance = node, straight_angle_tolerance
    cost = 0
    if pred_count == 1 and success_count == 1:
        return cost
    # Get the coordinates of the three nodes
    coords1 = get_coordinates(G, u)
    coords2 = get_coordinates(G, v)
    coords3 = get_coordinates(G, w)

    # Convert the latitude and longitude coordinates to x,y Cartesian coordinates
    x1, y1 = get_cartesian_coord_for_lat_long(coords1[0], coords1[1])
    x2, y2 = get_cartesian_coord_for_lat_long(coords2[0], coords2[1])
    x3, y3 = get_cartesian_coord_for_lat_long(coords3[0], coords3[1])

    # Compute the vectors ab and bc
    ab = (x2 - x1, y2 - y1)
    bc = (x3 - x2, y3 - y2)

    # Compute the angle between the vectors ab and bc
    angle = math.atan2(ab[0] * bc[1] - ab[1] * bc[0], ab[0] * bc[0] + ab[1] * bc[1])
    # sanity_check_23(angle)
    if angle >= tolerance:
        cost = 1
    return cost


def log_graph_details(route_num: int) -> None:
    """
    Logs the details of the route graph

    Args:
        route_num (int): Route number

    Returns:
        None
    """
    (G, L, terminal_set, _, _, _, _, _, _, tw_term_count, _, _, _) = (get_route_information(route_num))
    with open(f'./lsInputs/route_{route_num}_turns.pkl', 'rb') as file:
        turn_file = pickle.load(file)
    turns_count = sum(val for from_node, adj_dict in turn_file.items() for to_node, val in adj_dict.items())
    line_local = f"{route_num},{len(G.nodes())},{len(G.edges())},{len(L.nodes())},{len(L.edges())},{len(terminal_set)},{tw_term_count},{len(terminal_set) - tw_term_count},{turns_count}\n"
    log_line(line_local, f"./logs/graph_stats.txt", False)
    return None


def check_negative_cycle_primal(route_num: int) -> None:
    """
    Checks for negative cycles in the primal graph

    Args:
        route_num (int): Route number

    Returns:
        None
    """
    with open(f'./lsInputs/route_{route_num}_G.pkl', 'rb') as file:
        route_graph = pickle.load(file)
    energies = pd.read_csv(f"lsInputs/energy_primal_{route_num}.csv")
    energies_dict = {(row.v1, row.v2): row.energy for _, row in energies.iterrows()}
    cost_dict = defaultdict(dict)
    for i in route_graph.nodes():
        for j in route_graph.neighbors(i):
            cost_dict[i][j] = energies_dict[(i, j)]
    for j in route_graph.nodes():
        cost_dict[-1][j] = 0
    cost_dict = dict(cost_dict)

    def weight_fn(u, v, d):
        return cost_dict[u][v]

    flag = nx.negative_edge_cycle(route_graph, weight=weight_fn)

    # cycles = nx.simple_cycles(route_graph)
    # for cycle in tqdm(cycles):
    #     cycle.append(cycle[0])
    #     energy_sum = [energies_dict[cycle[node], cycle[node+1]] for node in range(len(cycle)-1)]
    #     if sum(energy_sum) < 0:
    #         print("Cycle found")
    #         break
    #
    #     print(cycle)
    # for s in tqdm(route_graph.nodes()):
    #     try:
    #         temp = nx.find_negative_cycle(route_graph, s, weight=weight_fn)
    #         break
    #     except:
    #         continue
    line_local = f"{route_num},{flag}\n"
    log_line(line_local, f"./logs/negative_cycle_primal.txt", False)
    return None


def check_negative_cycle_dual(route_num: int) -> None:
    """
    Checks for negative cycles in the dual graph

    Args:
        route_num (int): Route number

    Returns:
        None
    """
    (_, L, _, _, _, energy_dual, direction_dual, _, _, _, _, _, _) = (get_route_information(route_num))
    alpha_beta = [(1, 0.001), (1, 0), (0, 1), (0.001, 1)]
    for alpha, beta in alpha_beta:
        cost_dict = defaultdict(dict)
        for i in L.nodes():
            for j in L.neighbors(i):
                cost_dict[i][j] = alpha * energy_dual[i][j] + beta * direction_dual[i][j]
        for j in L.nodes():
            cost_dict[-1][j] = 0
        cost_dict = dict(cost_dict)

        def weight_fn(u, v, d):
            return cost_dict[u][v]

        flag = nx.negative_edge_cycle(L, weight=weight_fn)
        line_local = f"{route_num},{alpha},{beta},{flag}\n"
        log_line(line_local, f"./logs/negative_cycle_dual.txt", False)
    return None


def remove_duplicate_elevations() -> None:
    """
    Removes duplicate elevations from the elevation file

    Returns:
        None
    """
    elevation_file = pd.read_csv(node_elevationFile_primal, header=None).drop_duplicates()
    elevation_file.to_csv(node_elevationFile_primal, header=False, index=False)
    return None


def elevation_fn(lat: float, lon: float) -> tuple:
    """
    Get the elevation of a point using the USGS Elevation Point Query Service

    Args:
        lat (float): Latitude
        lon (float): Longitude

    Returns:
        elevation_val (float): Elevation of the point
        fail_flag (bool): Flag to indicate if the elevation value was obtained successfully
    """
    # USGS Elevation Point Query Service
    url = r'https://epqs.nationalmap.gov/v1/json?'
    # define rest query params
    params = {'x': lon, 'y': lat, 'units': 'Meters'}
    try:
        result_local = requests.get((url + urllib.parse.urlencode(params)), timeout=20)
        elevation_val = result_local.json()['value']
        fail_flag = False
    except:
        elevation_val, fail_flag = None, True
    return elevation_val, fail_flag


def get_height(coord: tuple) -> float:
    """
    Get the height of a point using the USGS Elevation Point Query Service

    Args:
        coord (tuple): Latitude and Longitude of the point

    Returns:
        height (float): Height of the point
    """
    count = 0
    lat = coord[0]
    lon = coord[1]
    fail_flag = True
    while fail_flag:
        elevation, fail_flag = elevation_fn(lat, lon)
        sleep(1 * count)
        count += 1
        if count > max_apicall_limit:
            break
    if not fail_flag:
        height = float(elevation)
    else:
        height = None
    return height


def process_primal_graph(G: networkx.classes.multidigraph.MultiDiGraph, straight_angle_tolerance: float) -> dict:
    """
    This function creates an adjacency list dictionary the primal Graph

    Args:
        G (networkx.classes.multidigraph.MultiDiGraph): Primal graph
        straight_angle_tolerance (float): Tolerance for straight angle

    Returns:
        adj_list_dict (dict): Adjacency list dictionary for the primal graph
    """
    adj_list_dict = {}

    class adjacency_list:
        def __init__(self):
            self.node = -1
            self.adj_nodes_turns = []
            self.successors_nodes = []

    for node in G.nodes():
        predecessor_nodes = list(G.predecessors(node))
        successor_nodes = list(G.successors(node))
        predecessor_len = len(predecessor_nodes)
        successor_len = len(successor_nodes)
        tmp = {}
        for u in predecessor_nodes:
            for w in successor_nodes:
                tmp[(u, node, w)] = get_turn_type(G, u, node, w, straight_angle_tolerance, predecessor_len, successor_len)

        adj_list_dict[node] = adjacency_list()
        adj_list_dict[node].node = node
        adj_list_dict[node].successors_nodes = list(G.successors(node))
        adj_list_dict[node].adj_nodes_turns = tmp

    return adj_list_dict


def check_terminal_connectivity(route_num: int) -> None:
    """
    Checks the connectivity of the terminals in the route graph

    Args:
        route_num (int): Route number

    Returns:
        None
    """
    amazon_del_coordinates, _, _ = read_dataset(route_num)
    with open(f'./lsInputs/route_{route_num}_G.pkl', 'rb') as ff:
        route_graph = pickle.load(ff)
    fixed_pass_list_temp = get_primal_edges_closest_to_delivery_location(amazon_del_coordinates, route_graph)
    all_reached = 1
    for u, _, _, _ in fixed_pass_list_temp:
        nodes_reached = nx.single_source_dijkstra(route_graph, u)
        for _, v, _, _ in fixed_pass_list_temp:
            if v not in nodes_reached[1].keys():
                all_reached = 0
        if not all_reached:
            break
    line_local = f"{route_num},{all_reached}\n"
    log_line(line_local, f"./logs/terminalConnectivity.txt", False)
    return None


def get_direction_between_nodes(v_i: tuple[int, int, int], v_j: tuple[int, int, int], primal_graph_processed: dict) -> int:
    """
    This function gives the direction between two nodes

    Args:
        v_i (tuple[int, int, int]): first node in dual graph
        v_j (tuple[int, int, int]): second node in dual graph
        primal_graph_processed (dict[tuple[int, int, int], ClassInstance]): Key: node in primal graph, Value: Node details in a class instance

    Returns:
        cost (int): 1 if LEFT ; 0 if right or STRAIGHT
    """
    prev_node = v_i[0]
    init_node = v_i[1]
    end_node = v_j[1]
    cost = primal_graph_processed[init_node].adj_nodes_turns[(prev_node, init_node, end_node)]
    return cost


def get_distance_between_nodes(v_i: tuple[int, int, int], v_j: tuple[int, int, int], G: networkx.classes.multidigraph.MultiDiGraph) -> float:
    """
    This function returns the distance between two nodes in the graph.

    Args:
        v_i (tuple[int, int, int]): first node in dual graph
        v_j (tuple[int, int, int]): second node in dual graph
        G (networkx.classes.multidigraph.MultiDiGraph): Primal graph

    Returns:
        Average distance between the 2 corresponding edges in primal graph.
    """
    prev_node = v_i[0]
    init_node = v_i[1]
    end_node = v_j[1]
    dist1 = G[int(init_node)][int(end_node)][0]['length']
    dist2 = G[int(prev_node)][int(init_node)][0]['length']
    return 0.5 * (dist1 + dist2)

def generate_cost_dicts(L: networkx.classes.multidigraph.MultiDiGraph, G: networkx.classes.multidigraph.MultiDiGraph, dual_graph_processed: dict, primal_graph_processed: dict,
                        energies_dict: dict[tuple[int, int], float]) -> tuple:
    """
    Generates distance, direction and energy cost dictionaries for the dual graph

    Args:
        L (networkx.classes.multidigraph.MultiDiGraph): Dual Graph
        G (networkx.classes.multidigraph.MultiDiGraph): Primal Graph
        dual_graph_processed (dict[tuple[int, int, int], ClassInstance]): Key: node in dual graph, Value: Node details in a class instance
        primal_graph_processed (dict[tuple[int, int, int], ClassInstance]): Key: node in primal graph, Value: Node details in a class instance
        energies_dict (dict[tuple[int, int], float]): Key: Tuple of node IDs, Value: Energy cost between the nodes

    Returns:
        distance_dual (dict[tuple[int, int, int], dict[tuple[int, int, int], float]]): Nested dictionary with distance between nodes in the Dual Graph
        direction_dual (dict[tuple[int, int, int], dict[tuple[int, int, int], int]]): Nested dictionary with direction between nodes in the Dual Graph
        energy_dual (dict[tuple[int, int, int], dict[tuple[int, int, int], float]]): Nested dictionary with energy cost between nodes in the Dual Graph

    """
    distance_dual = {v_i: {} for v_i in L.nodes}
    direction_dual = {v_i: {} for v_i in L.nodes}
    energy_dual = {v_i: {} for v_i in L.nodes}

    for v_i in L.nodes:
        v_i_temp = v_i[:3]
        for v_j in dual_graph_processed[v_i].successors_nodes:
            v_j_temp = v_j[:3]
            # These are the copies created for terminals. We don't need to calculate the distance, direction and energy for these nodes
            if v_i_temp == v_j_temp:
                distance_dual[v_i][v_j] = 0
                direction_dual[v_i][v_j] = 0
                energy_dual[v_i][v_j] = 0
            else:
                distance_dual[v_i][v_j] = get_distance_between_nodes(v_i_temp, v_j_temp, G)
                direction_dual[v_i][v_j] = get_direction_between_nodes(v_i_temp, v_j_temp, primal_graph_processed)
                energy_dual[v_i][v_j] = energies_dict[(v_i_temp, v_j_temp)]
    return distance_dual, direction_dual, energy_dual


def generate_pickles(route_num: int) -> None:
    """
    Generates pickle files for the given route number

    Args:
        route_num (int): Route number

    Returns:
        None
    """
    # route_num = 1629
    dual_energy_df = read_energy_file(energies_file, route_num, plot_energies_histogram)
    energies_dict = {(row['v1'], row['v2']): row['energy'] for _, row in dual_energy_df.iterrows()}

    amazon_del_coordinates, _, _, _ = read_dataset(route_num)
    with open(f'./lsInputs/route_{route_num}_G.pkl', 'rb') as f:
        G = pickle.load(f)
    fixed_pass_list_temp = get_primal_edges_closest_to_delivery_location(amazon_del_coordinates, G)

    L = nx.line_graph(G)
    nx.relabel_nodes(L, {node: (node[0], node[1], node[2], 0) for node in L.nodes()}, copy=False)

    fixed_pass_list_count = Counter(fixed_pass_list_temp)
    for node, count in fixed_pass_list_count.items():
        for copy_idx in range(1, count):
            new_node = (node[0], node[1], node[2], copy_idx)
            for (from_node, to_node) in L.in_edges(node):
                L.add_edge(from_node, new_node)
            for (from_node, to_node) in L.out_edges(node):
                L.add_edge(new_node, to_node)
            prev_copy = (node[0], node[1], node[2], copy_idx - 1)
            L.add_edge(prev_copy, new_node)
            L.add_edge(new_node, prev_copy)
    primal_graph_processed = process_primal_graph(G, straight_angle_tolerance)
    dual_graph_processed = process_dual_graph(L)
    distance_dual, direction_dual, energy_dual = generate_cost_dicts(L, G, dual_graph_processed, primal_graph_processed, energies_dict)

    with open(f'./lsInputs/route_{route_num}_sd.pkl', 'wb') as file:
        pickle.dump(distance_dual, file)
    with open(f'./lsInputs/route_{route_num}_turns.pkl', 'wb') as file:
        pickle.dump(direction_dual, file)
    with open(f'./lsInputs/route_{route_num}_distance.pkl', 'wb') as file:
        pickle.dump(energy_dual, file)
    check_negative_energy_self_cycles(energy_dual, dual_graph_processed, L, route_num)
    return None


def read_dataset_coords(route_num: int) -> tuple:
    """
    Reads the amazon dataset and extracts the bounding box coordinates for the given route number

    Args:
        route_num (int): Route number

    Returns:
        max_lat (float): Maximum latitude
        min_lat (float): Minimum latitude
        max_long (float): Maximum longitude
        min_long (float): Minimum longitude
    """

    with open(fr'./lsInputs/route_data.json', 'r') as f:
        data = json.load(f)
    route = list(data.keys())[route_num]
    stop_list = data[route]["stops"]
    terminals = [(stop["lat"], stop["lng"]) for stop in stop_list.values()]
    lat, long = zip(*terminals)
    max_lat, min_lat = max(lat), min(lat)
    max_long, min_long = max(long), min(long)
    return max_lat, min_lat, max_long, min_long


def get_osm_austin_graph(all_routes: list) -> tuple:
    """
    Returns the bounding box coordinates for the Austin OSM graph

    Args:
        all_routes (list): List of all routes

    Returns:
        max_lat (float): Maximum latitude
        min_lat (float): Minimum latitude
        max_long (float): Maximum longitude
        min_long (float): Minimum longitude
    """
    with open(f'./lsInputs/states.txt', 'r') as file:
        data = file.readlines()
    data = [line.strip() for line in data]
    data = [line[1:-1] for line in data]
    data = [line.split(',') for line in data]
    df_state = pd.DataFrame(data)
    df_state.columns = ['Route_ID', 'Terminals', 'G_Nodes', 'G_Edges', 'L_Nodes', 'L_Edges', 'Terminals', 'Ratio', 'State']
    df_state.to_csv(f'./lsInputs/state_df.csv', index=False)
    print("Number of routes = ", len(all_routes))

    max_lat, max_long = -1000, -1000
    min_lat, min_long = 1000, 1000

    for route_num in tqdm(all_routes):
        max_lat, min_lat, max_long, min_long = read_dataset_coords(route_num)

        if max_lat > max_lat:
            max_lat = max_lat
        if max_long > max_long:
            max_long = max_long
        if min_lat < min_lat:
            min_lat = min_lat
        if min_long < min_long:
            min_long = min_long
    print(max_lat, min_lat, max_long, min_long)
    return max_lat, min_lat, max_long, min_long


def get_elevationdict() -> dict:
    """
    Reads the elevation file and returns a dictionary of node and its elevation

    Returns:
        elevation_dict (dict): Dictionary of node and its elevation
    """
    data = pd.read_csv(node_elevationFile_primal, header=None)
    elevation_dict = {}
    for _, row in data.iterrows():
        elevation_dict[int(row[0])] = float(row[1])
    return elevation_dict


def lat_long(dual_node: tuple, G: networkx.classes.multidigraph.MultiDiGraph) -> tuple[float, float]:
    """
    Returns the latitude and longitude of the midpoint of the dual node

    Args:
        dual_node (tuple): Dual node
        G (networkx.classes.multidigraph.MultiDiGraph): Primal graph

    Returns:
        lat_mid (float): Latitude of the midpoint of the dual node
        long_mid (float): Longitude of the midpoint of the dual node
    """
    node_1 = dual_node[0]
    node_2 = dual_node[1]
    lat1 = G.nodes[node_1]['y']
    lat2 = G.nodes[node_2]['y']
    long1 = G.nodes[node_1]['x']
    long2 = G.nodes[node_2]['x']
    lat_mid = (lat1 + lat2) / 2
    long_mid = (long1 + long2) / 2
    return lat_mid, long_mid


def energy_fn(length: float, slope: float, speed: float) -> float:
    """
    Calculates the energy required to traverse an edge

    Args:
        length (float): Length of the edge in meter
        slope (float): Slope of the edge
        speed (float): Speed of the vehicle in km/hr

    Returns:
        energy_val (float): Energy required to traverse an edge
    """
    psi = math.atan(slope)
    energy_val = factor * (m * g * (f * math.cos(psi) + math.sin(psi)) + 0.5 * rho * cx * A * speed * speed / (3.6 * 3.6)) * length
    if energy_val <= 0:
        energy_val = regenerative_factor * energy_val
    return energy_val


def extract_osm_graph(route_num: int) -> None:
    """
    Extracts the OSM graph for a route and saves it to a pickle file

    Args:
        route_num (int): Route number

    Returns:
        None
    """
    max_lat, min_lat, max_long, min_long = read_dataset_coords(route_num)
    original_bbox = (max_lat, max_long), (max_lat, min_long), (min_lat, max_long), (min_lat, min_long)
    new_bounding_box = shift_bbox(original_bbox, bbox_padding)
    g_local = ox.graph_from_bbox(new_bounding_box[0][0], new_bounding_box[2][0], new_bounding_box[1][1],
                                 new_bounding_box[0][1], simplify=simplify)
    for v1 in g_local.nodes():
        for v2 in g_local[v1].keys():
            for keys in g_local[v1][v2].keys():
                try:
                    if type(g_local[v1][v2][keys]["maxspeed"]) == list:
                        val = sum([int(x.split(" ")[0]) for x in g_local[v1][v2][0]["maxspeed"]]) / len(g_local[v1][v2][0]["maxspeed"])
                    else:
                        val = int(g_local[v1][v2][keys]["maxspeed"].split(" ")[0])
                    g_local[v1][v2][keys]["speed_kph"] = val
                except:
                    # ValueError: comes because there are values like 40n
                    g_local[v1][v2][keys]["speed_kph"] = default_speed
    with open(f'./lsInputs/route_{route_num}_G.pkl', 'wb') as file:
        pickle.dump(g_local, file)
    return None


def save_energies(route_num: int) -> list:
    """
    Saves the energies to a file

    Args:
        route_num (int): Route number

    Returns:
        dataframe_values (list): List of energies and slopes for each edge
    """
    with open(f'./lsInputs/route_{route_num}_G.pkl', 'rb') as file:
        route_graph = pickle.load(file)
    dataframe_values = []
    for v1, v2, edge_data in route_graph.edges(data=True):
        try:
            speed = edge_data['speed_kph']
            length = edge_data['length']
            height = elevation_dict[v2] - elevation_dict[v1]
            slope = round(height / length, 6)
            energy = round(energy_fn(length, slope, speed), 6)
            if str(slope) == "nan" or str(energy) == "nan":
                # This happens when the length of the edge is 0. This happens because prev_node and init_node have same coordinates. simplify=False should remove this.
                slope, energy = 0, 0
            dataframe_values.append((v1, v2, energy, slope))
        except KeyError:
            print(route_num, v1, v2, edge_data)
    filename = energies_file + "_" + "primal" + "_" + str(route_num) + ".csv"
    pd.DataFrame(dataframe_values, columns=['v1', 'v2', 'energy', 'slope']).to_csv(f'{filename}', index=False)

    l_local = nx.line_graph(route_graph)
    dataframe_values = []
    for v1, v2 in l_local.edges():
        speed = (float(route_graph[v1[0]][v1[1]][0]['speed_kph']) + float(route_graph[v2[0]][v2[1]][0]['speed_kph'])) / 2
        prev_node = v1[0]
        init_node = v1[1]
        end_node = v2[1]
        length = 0.5 * (route_graph[prev_node][init_node][0]['length'] + route_graph[init_node][end_node][0]['length'])
        height = (elevation_dict[v2[0]] + elevation_dict[v2[1]]) / 2 - (elevation_dict[v1[0]] + elevation_dict[v1[1]]) / 2
        slope = round(height / length, 6)
        energy = round(energy_fn(length, slope, speed), 6)
        if str(slope) == "nan" or str(energy) == "nan":
            # This happens when the length of the edge is 0. This happens because prev_node and init_node have same coordinates. simplify=False should remove this.
            slope, energy = 0, 0
        dataframe_values.append((v1, v2, energy, slope))
    filename = energies_file + "_" + "dual" + "_" + str(route_num) + ".csv"
    pd.DataFrame(dataframe_values, columns=['v1', 'v2', 'energy', 'slope']).to_csv(f'{filename}', index=False)
    return dataframe_values


def fetch_height(inputs: tuple) -> None:
    """
    Fetches the height of a node using the USGS Elevation Point Query Service

    Args:
        inputs (tuple): Node and its coordinates

    Returns:
        None
    """
    node, coords = inputs
    height = get_height(coords)
    if height is None:
        print("Elevation API Failed")
    else:
        line_local = f"{node},{height}\n"
        log_line(line_local, node_elevationFile_primal, False)
    return None


def get_nodes(route_num: int) -> list:
    """
    Get the nodes that are not in the elevation file

    Args:
        route_num (int): Route number

    Returns
        new_nodes (list): List of nodes that are not in the elevation file
    """
    with open(f'./lsInputs/route_{route_num}_G.pkl', 'rb') as f:
        route_graph = pickle.load(f)
    new_nodes = set(route_graph.nodes()) - set(elevation_dict.keys())
    new_nodes = [(n, (route_graph.nodes[n]["y"], route_graph.nodes[n]["x"])) for n in new_nodes]
    return new_nodes


def save_dual_energies(route_num: int) -> None:
    """
    Saves the dual energies to a file

    Args:
        route_num (int): Route number

    Returns:
        None
    """
    with open(f'./lsInputs/route_{route_num}_G.pkl', 'rb') as file:
        route_graph = pickle.load(file)
    l_local = nx.line_graph(route_graph)
    dual_elevations = [(n, (elevation_dict[n[0]] + elevation_dict[n[1]]) / 2) for n in l_local.nodes()]

    for node, ele in dual_elevations:
        line_local = f"{node},{ele}\n"
        log_line(line_local, node_elevationFile_dual, False)
    return None


def plot_time_window_histogram() -> None:
    """
    Plots the histogram of the time windows

    Returns:
        None
    """
    good_routes_local = pd.read_csv(fr"./lsInputs/working_amazon_routes.csv")
    graph_stats_local = pd.read_csv(fr"./logs/graph_stats.txt")
    data = graph_stats_local[graph_stats_local.Route_num.isin(good_routes_local.Route_num.tolist())].TWSubset.tolist()

    plt.hist(data, bins=10, color='skyblue', edgecolor='black')
    plt.xticks(range(1, 32, 3))
    plt.yticks(range(0, 22, 2))
    plt.xlabel('Time Windows')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel('Number of Routes')
    plt.title('TW Histogram')
    plt.savefig(f"./logs/TW_histogram.png")
    return None


if __name__ == "__main__":
    energies_file = f'./lsInputs/energy'
    node_elevationFile_primal = f'./lsInputs/node_elevationFile_primal.txt'
    node_elevationFile_dual = f'./lsInputs/node_elevationFile_dual.txt'
    state = "Austin"
    cores = 214
    THREAD_COUNT = 500
    bbox_padding = 3
    all_routes = [
        18, 64, 65, 83, 170, 185, 239, 264, 269, 273, 284, 293, 296, 317, 351, 399, 431, 432,
        488, 556, 610, 640, 696, 712, 722, 731, 741, 763, 792, 794, 795, 824, 896, 909, 1043,
        1055, 1099, 1103, 1148, 1187, 1230, 1236, 1333, 1349, 1362, 1383, 1398, 1501, 1509,
        1554, 1569, 1609, 1611, 1629, 1643, 1665, 1695, 1719, 1730, 1833, 1838, 1840, 1842,
        1860, 1895, 1897, 1921, 1925, 1970, 1974, 1986, 1993, 1998, 2092, 2151, 2157, 2163,
        2180, 2189, 2233, 2251, 2270, 2340, 2356, 2392, 2407, 2455, 2528, 2529, 2541, 2567,
        2571, 2581, 2616, 2685, 2698, 2710, 2725, 2779, 2826, 2884, 2928, 2964, 2992, 3044,
        3056, 3107, 3122, 3154, 3237, 3261, 3287, 3295, 3311, 3390, 3412, 3436, 3464, 3484,
        3571, 3583, 3618, 3702, 3704, 3742, 3752, 3793, 3822, 3833, 3837, 3890, 3909, 3923,
        3999, 4039, 4068, 4163, 4213, 4214, 4236, 4239, 4248, 4260, 4265, 4269, 4288, 4311,
        4332, 4336, 4463, 4494, 4505, 4530, 4542, 4556, 4597, 4636, 4656, 4674, 4699, 4726,
        4732, 4756, 4804, 4834, 4837, 4862, 4889, 4946, 5007, 5090, 5094, 5095, 5096, 5098,
        5102, 5119, 5167, 5224, 5255, 5278, 5280, 5288, 5311, 5419, 5429, 5437, 5485, 5528,
        5539, 5544, 5555, 5556, 5605, 5623, 5633, 5646, 5654, 5668, 5686, 5696, 5740, 5766,
        5824, 5828, 5837, 5916, 5935, 5954, 5963, 5993, 6003, 6106, 6107
    ]
    m = 27216  # kg
    g = 9.8  # m/s^2
    f = 0.0058  # coefficient of rolling resistance
    cx = 0.6  # coefficient of air drag
    rho = 1.1  # kg/m^3
    A = 5.4  # m^2
    factor = 1 / 3600 * 1 / 1000  # converting joules to kWh
    regenerative_factor = 0.7  # regenerative braking factor
    default_speed = 25  # default speed in km/hr
    max_apicall_limit = 6  # maximum number of API calls
    debug = False  # Debug mode
    simplify = True  # Simplify the graph
    plot_energies_histogram = False  # Plot the histogram of energies
    straight_angle_tolerance = math.pi / 4  # 45 degrees

    # ############################################################################
    # print("Extracting OSM graph for each route. Expected runtime: 1000 seconds")
    # start = time()
    # with multiprocessing.Pool(cores) as pool:
    #     pool.map(extract_osm_graph, all_routes)
    # print(f"Time taken to extract OSM graphs = {round(time() - start, 2)} seconds")
    # # ##########################################################################
    # print("Checking connectivity of terminals. Expected runtime: 1000 seconds")
    # start = time()
    # line = "Route_num,Success\n"
    # log_line(line, f"./logs/terminalConnectivity.txt", True)
    # with multiprocessing.Pool(cores) as pool:
    #     pool.map(check_terminal_connectivity, all_routes)
    # print(f"Time taken to check connectivity = {round(time() - start, 2)} seconds")
    # ############################################################################
    # print("Extracting elevations for primal nodes")
    # start = time()
    # Uncomment with care! This will delete the contents of the elevation files
    # # with open(node_elevationFile_primal, 'a') as file:
    # #     file.truncate(0)
    # # with open(node_elevationFile_dual, 'a') as file:
    # #     file.truncate(0)
    # elevation_dict = get_elevationdict()
    # with multiprocessing.Pool(cores) as pool:
    #     result = pool.map(get_nodes, all_routes)
    # remaining_nodes = list(set([y for x in result for y in x]))
    # remaining_nodes = custom_shuffle(remaining_nodes)
    # print(f"Number of remaining nodes = {len(remaining_nodes)}")
    # with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
    #     executor.map(fetch_height, remaining_nodes)
    # print(f"Time taken to extract remaining elevations = {round(time() - start, 2)} seconds")
    # remove_duplicate_elevations()
    # print("Generating elevations for dual nodes")
    # elevation_dict = get_elevationdict()
    # with multiprocessing.Pool(cores) as pool:
    #     pool.map(save_dual_energies, all_routes)
    # ############################################################################
    # print("Generating edge energy and slope values for primal and dual graph. Expected runtime: 600 seconds")
    # start = time()
    # elevation_dict = get_elevationdict()
    # with multiprocessing.Pool(cores) as pool:
    #     pool.map(save_energies, all_routes)
    # print(f"Time taken to generate energy files = {round(time() - start, 2)} seconds")
    ###########################################################################
    print("Generating final pickle files. Expected time: 600 Seconds")
    start = time()
    with multiprocessing.Pool(cores) as pool:
        pool.map(generate_pickles, all_routes)
    print(f"Time taken to generate final pickle files = {round(time() - start, 2)} seconds")
    # # ##########################################################################
    # print("Checking negative edge cycles in primal. Expected runtime: 300 seconds")
    # start = time()
    # line = "Route_num,Flag\n"
    # log_line(line, f"./logs/negative_cycle_primal.txt", True)
    # with multiprocessing.Pool(cores) as pool:
    #     pool.map(check_negative_cycle_primal, all_routes)
    # print(f"Time taken to check primal negative cycles = {round(time() - start, 2)} seconds")

    # start = time()
    # print("Checking negative edge cycles in dual. Expected runtime: 600 seconds")
    # line = "Route_num,Alpha,Beta,Flag\n"
    # log_line(line, f"./logs/negative_cycle_dual.txt", True)
    # with multiprocessing.Pool(cores) as pool:
    #     pool.map(check_negative_cycle_dual, all_routes)
    # print(f"Time taken to check dual negative cycles = {round(time() - start, 2)} seconds")
    ##########################################################################
    print("Loging graph stats. Expected runtime: 450 seconds")
    line = "Route_num,PrimalNodes,PrimalEdges,DuaNodes,DualEdges,TotalTerminals,TWSubset,NonTWSubset,TurnsCount\n"
    log_line(line, f"./logs/graph_stats.txt", True)
    start = time()
    with multiprocessing.Pool(cores) as pool:
        pool.map(log_graph_details, all_routes)
    print(f"Time taken to log stats = {round(time() - start, 2)} seconds")
    graph_stats = pd.read_csv(fr"./logs/graph_stats.txt")
    graph_stats.sort_values(by=["TotalTerminals"]).reset_index(drop=True).to_csv(fr"./logs/graph_stats.txt", index=False)
    ###########################################################################
    negative_cycle_dual = pd.read_csv(f"./logs/negative_cycle_dual.txt")
    negative_cycle_primal = pd.read_csv(f"./logs/negative_cycle_primal.txt")
    terminalConnectivity = pd.read_csv(f"./logs/terminalConnectivity.txt")
    graph_stats = pd.read_csv(f"./logs/graph_stats.txt")
    route_with_negative_primal_cycle = negative_cycle_primal[negative_cycle_primal['Flag'] == True]['Route_num'].tolist()
    route_with_negative_dual_cycle = negative_cycle_dual[negative_cycle_dual['Flag'] == True]['Route_num'].tolist()
    no_tw_routes = graph_stats[graph_stats['TWSubset'] == 0]['Route_num'].tolist()
    not_reachable_routes = terminalConnectivity[terminalConnectivity['Success'] == 0]['Route_num'].tolist()
    good_routes = set(all_routes) - set(route_with_negative_primal_cycle) - set(route_with_negative_dual_cycle) - set(not_reachable_routes) - set(no_tw_routes)
    print(
        f"Routes with no TW termials = {len(graph_stats[graph_stats['TWSubset'] == 0])}, {round(len(graph_stats[graph_stats['TWSubset'] == 0]) / len(graph_stats) * 100, 2)}%")
    print(
        f"Routes where terminals are not reachable in primal = {len(terminalConnectivity[terminalConnectivity['Success'] == 0])}, {round(len(terminalConnectivity[terminalConnectivity['Success'] == 0]) / len(terminalConnectivity) * 100, 2)}%")
    print(
        f"Routes with negative cycles in primal = {negative_cycle_primal['Flag'].value_counts()[True]}, {round((negative_cycle_primal['Flag'].value_counts()[True]) / len(all_routes) * 100, 2)}%")
    print(
        f"Routes with negative cycles in dual = {len(set(negative_cycle_dual[negative_cycle_dual['Flag'] == True].Route_num.tolist()))}, {round(len(set(negative_cycle_dual[negative_cycle_dual['Flag'] == True].Route_num.tolist())) / len(all_routes) * 100, 2)}%")
    print(
        f"Routes with negative cycles in either primal or dual = {len(set(route_with_negative_primal_cycle + route_with_negative_dual_cycle))}, {round(len(set(route_with_negative_primal_cycle + route_with_negative_dual_cycle)) / len(all_routes) * 100, 2)}%")
    print(
        f"Routes with negative cycles in both primal and dual = {len(set(route_with_negative_primal_cycle).intersection(set(route_with_negative_dual_cycle)))}, {round(len(set(route_with_negative_primal_cycle).intersection(set(route_with_negative_dual_cycle))) / len(all_routes) * 100, 2)}%")
    print(f"Candidates for final set of routes = {len(good_routes)}, {round(len(good_routes) / len(all_routes) * 100, 2)}%")
    pd.DataFrame(good_routes, columns=["Route_num"]).to_csv(f"./logs/good_routes.txt", index=False)
    plot_time_window_histogram()
