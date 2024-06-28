"""
This file contains the functions to read the dataset, create the primal and dual graphs, and related functions
"""
import osmnx as ox
import numpy as np
import networkx as nx
from datetime import datetime
from datetime import timedelta
from collections import Counter
from collections import defaultdict
from haversine import haversine_vector
import math, json, pickle, networkx, statistics

from helpers.functions import process_dual_graph


def shift_bbox(bbox: tuple, distance: int) -> list:
    """
    Shift the bounding box by the input distance

    Args:
        bbox (tuple): list of 4 coordinates, i.e., the corner points of the bounding box
        distance (float): distance to shift the bounding box

    Returns:
        new_points (list): list of 4 coordinates, i.e., the corner points of the new bounding box
    """
    R = 6371  # radius of Earth in kilometers
    lat_shift = (distance / R) * (180 / math.pi)  # latitude shift in degrees
    lon_shift = (distance / R) * (180 / math.pi) / math.cos(bbox[0][0] * math.pi / 180)  # longitude shift in degrees

    corner_points = bbox
    new_points = [(corner_points[0][0] + lat_shift, corner_points[0][1] + lon_shift), (corner_points[1][0] + lat_shift, corner_points[1][1] - lon_shift),
                  (corner_points[2][0] - lat_shift, corner_points[2][1] + lon_shift), (corner_points[3][0] - lat_shift, corner_points[3][1] - lon_shift)]

    return new_points


def read_dataset(route_num: int) -> tuple:
    """
    This function extracts data from an Amazon dataset.

    Args:
        route_num (int): Route number

    Returns:
        terminals (list): list of coordinates of delivery locations
        corner_points (list): list of corner points of the bounding box
        time_windowlist (list): list of time windows for each terminal
        depot_idx (int): index of the depot in the terminals list
    """
    with open(f'./lsInputs/route_data.json', 'r') as f:
        data = json.load(f)
    route = list(data.keys())[route_num]

    departure = data[route]['departure_time_utc']
    departure = datetime.strptime(departure, '%H:%M:%S')
    departure = str(departure)
    departure = datetime.strptime(departure, '%Y-%m-%d %H:%M:%S')

    temlist = data[route]['stops'].keys()
    with open(f'./lsInputs/package_data.json', 'r') as f:
        package_data = json.load(f)
    time_windowlist = []
    service_time = []  # Service time is not used in the code
    for term in temlist:
        try:
            service_time.append(list(package_data[route][term].values())[0]["planned_service_time_seconds"])
        except IndexError:
            service_time.append(300)
        try:
            # Get the time window and convert it to string
            temp = list(package_data[route][term].values())[0]["time_window"]
            temp["start_time_utc"] = str(temp["start_time_utc"])
            temp["end_time_utc"] = str(temp["end_time_utc"])

            # Convert the time window to datetime object as long as it is not nan
            if temp["start_time_utc"] != "nan" or temp["end_time_utc"] != "nan":
                temp["start_time_utc"] = datetime.strptime(str(temp["start_time_utc"]), '%Y-%m-%d %H:%M:%S')
                temp["end_time_utc"] = datetime.strptime(str(temp["end_time_utc"]), '%Y-%m-%d %H:%M:%S')

                # Calculate the time difference between the departure time and the time window
                time_start = temp["start_time_utc"] - datetime.combine(temp["start_time_utc"].date(), departure.time())

                time_end = temp["end_time_utc"] - datetime.combine(temp["end_time_utc"].date(), departure.time())

                # If the time difference for time window end is negative, add 24 hours to the time window (this means that the time window ends the next day)
                if time_end.total_seconds() < 0:
                    new_date = temp["end_time_utc"].date() + timedelta(days=-1)

                    time_end = temp["end_time_utc"] - datetime.combine(new_date, departure.time())

                # If the time difference for time window start is negative, set it to 0 (this means that the time window starts before the departure time)
                if time_start.total_seconds() < 0:
                    time_start = 0

                # Append the time window to the list in seconds
                if time_start != 0:
                    time_windowlist.append((time_start.total_seconds(), time_end.total_seconds()))
                else:
                    time_windowlist.append((time_start, time_end.total_seconds()))
            else:
                time_windowlist.append((-float("inf"), float("inf")))

        except IndexError:
            time_windowlist.append((-float("inf"), float("inf")))
    stop_list = data[route]["stops"]
    terminals = [(stop["lat"], stop["lng"]) for stop in stop_list.values()]
    try:
        depot = [(metadata["lat"], metadata["lng"]) for metadata in stop_list.values() if metadata["type"] == "Station"][0]
        depot_idx = [x for x in range(len(terminals)) if terminals[x] == depot][0]
    except IndexError:
        raise ValueError("Depot not found. Check here")
    lat, long = zip(*terminals)
    max_lat, min_lat = max(lat), min(lat)
    max_long, min_long = max(long), min(long)
    corner_points = (max_lat, max_long), (max_lat, min_long), (min_lat, max_long), (min_lat, min_long)
    if len(terminals) != len(time_windowlist) or len(terminals) != len(service_time):
        raise FileExistsError("Error in read_dataset function. Error code: 2134")
    return terminals, corner_points, time_windowlist, depot_idx


def get_primal_edges_closest_to_delivery_location(amazon_del_coordinates: list[tuple], G: networkx.classes.multidigraph.MultiDiGraph) -> list:
    """
    This function gets the list of primal graph edges that are the closest to the delivery locations.

    Args:
        amazon_del_coordinates (list): list of coordinates of delivery locations
        G (networkx.classes.multidigraph.MultiDiGraph): Primal Graph

    Returns:
        delivery_edges (list): list of edges from the primal graph that are nearest to the delivery locations

    """
    node_list = [(node[0], (node[1]["y"], node[1]["x"])) for node in G.nodes(data=True)]
    node_id, node_coordinates = zip(*node_list)
    nearest_primal_nodes = []
    for delivery in amazon_del_coordinates:
        nearest_primal_nodes.append(node_id[np.argmin(haversine_vector(node_coordinates, len(node_coordinates) * [delivery]))])
    lat_list, long_list = zip(*amazon_del_coordinates)
    delivery_edges = ox.nearest_edges(G, long_list, lat_list, return_dist=True)
    delivery_edges, dist_list = delivery_edges
    delivery_edges = [(*edge, 0) for edge in delivery_edges]
    return delivery_edges


def create_line_graph(G: networkx.classes.multidigraph.MultiDiGraph, fixed_pass_list_temp: list) -> tuple:
    """
    This function creates a line graph from the primal graph.

    Args:
        G (networkx.classes.multidigraph.MultiDiGraph): Primal Graph
        fixed_pass_list_temp (list): delivery locations

    Returns:
        L (networkx.classes.multidigraph.MultiDiGraph): Line Graph
        fixed_pass_list_new (list): delivery locations
    """
    L = nx.line_graph(G)
    nx.relabel_nodes(L, {node: (node[0], node[1], node[2], 0) for node in L.nodes()}, copy=False)

    fixed_pass_list_new = fixed_pass_list_temp[:]
    unique_terminals = set(fixed_pass_list_temp)
    for term in unique_terminals:
        index_to_be_fixed = [idx for idx, n in enumerate(fixed_pass_list_new) if n == term]
        for idx, pos in enumerate(index_to_be_fixed):
            fixed_pass_list_new[pos] = (term[0], term[1], term[2], idx)

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
    return L, fixed_pass_list_new


def get_route_information(route_num: int) -> tuple:
    """
    This function takes the route number as an input and returns the graph from the pickle files associated with that route.

    Args:
        route_num (int): Route number

    Returns:
        G (networkx.classes.multidigraph.MultiDiGraph): Primal Graph
        L (networkx.classes.multidigraph.MultiDiGraph): Line Graph
        terminal_set (set): Set of terminal nodes
        time_window_dict (dict): The dictionary containing the time windows for each terminal
        time_cost_dict (dict): Dictionary of travel times
        energy_dual (dict): Energy cost
        direction_dual (dict): Direction cost
        mean_energy_dual (float): Mean energy cost for the dual graph
        mean_turn_dual (float): Mean turn cost for the dual graph
        tw_term_count (int): Count of terminals with time windows
        stdev_energy_dual (float): Standard deviation of energy costs for the dual graph
        stdev_turn_dual (float): Standard deviation of turn costs for the dual graph
        depot (tuple): Depot node
    """
    print("*" * 100, "\nStarting Route No:", route_num)
    amazon_del_coordinates, original_bbox, time_windowlist, depot_idx = read_dataset(route_num)
    with open(f'./lsInputs/route_{route_num}_G.pkl', 'rb') as f:
        G = pickle.load(f)

    try:
        with open(f'./lsInputs/route_{route_num}_L.pkl', 'rb') as f:
            L = pickle.load(f)
        with open(f'./lsInputs/route_{route_num}_fixed_pass_list.pkl', 'rb') as f:
            fixed_pass_list = pickle.load(f)
        # print(f"Loaded L and fixed_pass_list from pickle files for route {route_num}")
    except FileNotFoundError:
        # temp because copies of the nodes are created later in the dual graph
        fixed_pass_list_temp = get_primal_edges_closest_to_delivery_location(amazon_del_coordinates, G)
        L, fixed_pass_list = create_line_graph(G, fixed_pass_list_temp)
        with open(f'./lsInputs/route_{route_num}_L.pkl', 'wb') as f:
            pickle.dump(L, f)
        with open(f'./lsInputs/route_{route_num}_fixed_pass_list.pkl', 'wb') as f:
            pickle.dump(fixed_pass_list, f)
        # print(f"Created L and fixed_pass_list for route {route_num}")
    depot = fixed_pass_list[depot_idx]

    tw_term_count = 0
    time_window_dict = {}
    for (idx, (lowerTime, upperTime)) in enumerate(time_windowlist):
        time_window_dict[fixed_pass_list[idx]] = (lowerTime, upperTime)
        if lowerTime == -float("inf") or upperTime == float("inf"):
            continue
        else:
            tw_term_count += 1

    # sanity_check_22(time_window_dict, depot, route_num)
    dual_graph_processed = process_dual_graph(L)

    with open(f'./lsInputs/route_{route_num}_sd.pkl', 'rb') as f:
        distance_dual = pickle.load(f)

    with open(f'./lsInputs/route_{route_num}_turns.pkl', 'rb') as f:
        direction_dual = pickle.load(f)

    with open(f'./lsInputs/route_{route_num}_distance.pkl', 'rb') as f:
        energy_dual = pickle.load(f)

    for i in list(L.nodes()):
        for j in dual_graph_processed[i].successors_nodes:
            energy_dual[i][j] = round(energy_dual[i][j], 3)

    time_cost_dict = defaultdict(dict)
    for v1 in list(L.nodes):
        for v2 in dual_graph_processed[v1].successors_nodes:
            time_cost_dict[v1][v2] = distance_dual[v1][v2] / (0.277778 * ((float(G[v1[0]][v1[1]][0]['speed_kph']) + float(G[v2[0]][v2[1]][0]['speed_kph'])) / 2))

    edge_energies = [energy_dual[i][j] for i in L.nodes for j in dual_graph_processed[i].successors_nodes]
    all_turns = [direction_dual[i][j] for i in L.nodes for j in dual_graph_processed[i].successors_nodes]
    mean_energy_dual = statistics.mean(edge_energies)
    mean_turn_dual = statistics.mean(all_turns)
    stdev_energy_dual = statistics.stdev(edge_energies)
    stdev_turn_dual = statistics.stdev(all_turns)

    terminal_set = set(tuple(fixed_pass_list))
    # progress_file = f"./logs/{route_num}/progressTrack.txt"
    # log_line("Graph read successfully\n", progress_file, False)
    return G, L, terminal_set, time_window_dict, time_cost_dict, energy_dual, direction_dual, mean_energy_dual, mean_turn_dual, tw_term_count, stdev_energy_dual, stdev_turn_dual, depot


def check_negative_energy_self_cycles(energy_dual: dict, adj_list_dict: dict, L: networkx.classes.multidigraph.MultiDiGraph, route_num: int) -> None:
    """
    This function checks for negative energy cycles in the line graph.

    Args:
        energy_dual (dict): Energy cost
        adj_list_dict (dict): dictionary of adjacent nodes
        L (networkx.classes.multidigraph.MultiDiGraph): Line Graph
        route_num (int): Route number

    Returns:
        None

    """
    for i in L.nodes():
        for j in adj_list_dict[i].successors_nodes:
            if energy_dual[i][j] <= 0:
                try:
                    if energy_dual[j][i] + energy_dual[i][j] < 0:
                        print(f"value 1: {energy_dual[j][i]}")
                        print(f"value 2: {energy_dual[i][j]}")
                        print(f"sum: {energy_dual[j][i] + energy_dual[i][j]}")
                        print(f"i: {i}, j: {j} in route {route_num}")
                        raise Warning(f"-ve energy self-cycle found. Exiting...")
                except KeyError:
                    continue
    return None


"""

def find_area(bbox) -> float:
    '''
    Calculate the distance between opposite corners of the bounding box

    Args:
        bbox (list): list of 4 coordinates, i.e., the corner points of the bounding box

    Returns:
        area (float): area of the bounding box in km


    '''
    d1 = haversine_dist(bbox[0], bbox[2])
    d2 = haversine_dist(bbox[1], bbox[3])
    area = d1 * d2
    return round(area, 2)


def load_graph_ip(L, energy_cost, turn_cost, time_cost_dict, time_windows, terminals):
    nodes_list = list(L.nodes())
    energy_dual = defaultdict(dict)
    direction_dual = defaultdict(dict)
    time_cost_dict = defaultdict(dict)

    for v_i in nodes_list:
        for v_j in L.neighbors(v_i):
            energy_dual[v_i][v_j] = energy_cost[(v_i, v_j)]
            direction_dual[v_i][v_j] = turn_cost[(v_i, v_j)]
            time_cost_dict[v_i][v_j] = time_cost_dict[(v_i, v_j)]

    fixed_pass_list = terminals
    time_window_dict = time_windows

    max_energy = max(energy_cost.values())
    min_energy = min(energy_cost.values())
    mean_energy_dual = sum(energy_cost.values()) / len(energy_cost)
    mean_turn_dual = sum(turn_cost.values()) / len(turn_cost)
    return L, fixed_pass_list, energy_dual, direction_dual, time_cost_dict, time_window_dict, max_energy, min_energy, mean_energy_dual, mean_turn_dual
def haversine_dist(coord1, coord2) -> float:
    '''    
    Find the haversine distance between the coordinates.
    
        Args:
            coord1 (tuple): First Coordinate
            coord2 (tuple): Second Coordinate
    
        Returns:
            d: (float) haversine distance
    
    '''
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # radius of the earth in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d

"""
