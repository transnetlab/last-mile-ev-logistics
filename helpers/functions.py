"""
This file contains helper functions for the Local Search
"""
import numpy as np
import math, shutil
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from paretoset import paretoset
from scipy.stats import gaussian_kde
import networkx, random, time, pickle
from matplotlib.colors import LinearSegmentedColormap

from helpers.sanityChecks import sanity_check_16, sanity_check_17, sanity_check_23

sns.set_style("whitegrid")


def get_uniform_random_number(lower: int, upper: int, seed: int) -> float:
    """
    This function returns a random number between two numbers.

    Args:
        lower (int): Lower bound
        upper (int): Upper bound
        seed (int): Seed for random number generation

    Returns:
        val (float): Random number between lower and upper
    """
    # np.random.seed(seed)
    val = np.random.uniform(lower, upper)
    return val


def custom_shuffle(data: list, seed: int = 0) -> list:
    """
    This function shuffles a list.

    Args:
        data (list): List to be shuffled
        seed (int): Seed for random number generation

    Returns:
        data_copy (list): Shuffled list

    """
    # np.random.seed(seed)
    data_copy = data[:]
    np.random.shuffle(data_copy)
    return data_copy


def get_random(data: list, probs: list, seed: int = 0) -> int:
    """
    This function gives a random element from a list.

    Args:
        data (list): List of data
        probs (list): List of probabilities
        seed (int): Seed for random number generation

    Returns:
        data[rnd] (int): Random element from the list
    """
    # np.random.seed(seed)
    if probs:
        rnd = np.random.choice(len(data), p=probs)
    else:
        rnd = np.random.choice(len(data))
    return data[rnd]


def get_random_sample(data: list, n: int, seed: int = 0) -> list:
    """
    This function gives a random sample of given size from a list.

    Args:
        data (list): List of data
        n (int): Size of the sample
        seed (int): Seed for random number generation

    Returns:
        data_subset (list): Random sample of size n from the list
    """
    # random.seed(seed)
    if n > len(data):
        data_subset = custom_shuffle(data)
    else:
        data_subset = random.sample(data, n)
    return data_subset


def cyclize(path: list[tuple[int, int, int]], start_index: int) -> list[tuple[int, int, int]]:
    """
    Given a tour and an index, this function returns a tour that starts as well as ends at that index.

    Args:
        path (list[tuple[int, int, int]]): A tour in Dual Graph
        start_index (int): Index in the tour

    Returns:
        newtour (list[tuple[int, int, int]]): A tour in Dual Graph starting and ending from the index
    """
    # From the start, till the depot + from (depot + 1) to start (start is included). Depot + 1 avoids adding depot twice in the path
    return list(path[start_index:] + path[1:start_index + 1])


def remove_consecutive_duplicates(path: tuple) -> tuple:
    """
    This function removes consecutive duplicates nodes from a tour.

    Args:
        path (tuple[tuple[int, int, int]]): A tour/path in Dual Graph

    Returns:
        result (tuple[tuple[int, int, int]]): A tour in Dual Graph without consecutive duplicates
    """
    return tuple([path[i] for i in range(len(path)) if i == 0 or path[i] != path[i - 1]])


def get_subpath(input_path: list, start_index: int, end_index: int, exclude_extreme_pts: bool) -> list:
    """
    Given tour, a start index and an end index, this function returns the subpath within the tour that lies between the start index and the end index.

    Args:
        input_path (list): The tour
        start_index (int): Start index
        end_index (int): End index
        exclude_extreme_pts (bool): Whether to exclude the extreme points

    Returns:
        path (list): Subpath of the tour between start_index and end_index
    """
    path = cyclize(input_path, start_index)
    if end_index < start_index:
        path = path[0:len(path) - start_index + end_index]
    else:
        # +1 to include the end_index
        path = path[0:end_index - start_index + 1]
    if exclude_extreme_pts:
        path.pop(0)
        path.pop(-1)
    return path


def get_terminal_position(path: list, terminals: set) -> list[int]:
    """
    This function gives the position of terminals in a path.

    Args:
        path (list[tuple[int, int, int]]): A path/tour in Dual Graph
        terminals (set[tuple[int,int,int]]): List of terminals in the dual graph

    Returns:
        index_val (list[int]): List of positions of terminals in the path
    """
    return [index for index, node in enumerate(path) if node in terminals]


def get_path_length(path: list, objective_dict: dict) -> float:
    """
    Returns the length of a path based on the objective dictionary

    Args:
        path (list): A path/tour
        objective_dict (dict): Objective dictionary

    Returns:
        val (float): Length of the path
    """
    if len(path) <= 1:
        raise ValueError("Path length is less than 2. Check here")
    return sum([objective_dict[path[node]][path[node + 1]] for node in range(len(path) - 1) if path[node] != path[node + 1]])


def get_path_length_using_energy_turns(path: list, combined_dict: dict) -> tuple[float, float]:
    """
    This function gives the length of a path based on the combined dictionary.

    Args:
        path (list): A path/tour
        combined_dict (dict): The dictionary containing the combined energy and direction values

    Returns:
        e (float): Energy cost of the path
        t (float): Turns cost of the path
    """
    if len(path) <= 1:
        raise ValueError("Path length is less than 2. Check here")
    e, t = zip(*[combined_dict[tail][head] for tail, head in zip(path, path[1:])])
    return sum(e), sum(t)


def get_successors_nodes_in_dual(node: tuple, L: networkx.classes.multidigraph.MultiDiGraph) -> list:
    """
    Given a dual graph and node, this function gives a list of nodes that are adjacent to the given node.

    Args:
        node: The node for which to find adjacent nodes.
        L (networkx.classes.multidigraph.MultiDiGraph): Dual graph

    Returns:
        A list of adjacent nodes.
    """
    # get the neighbors of the node in the graph
    successors_nodes = list(L.successors(node))
    if not successors_nodes:
        return []
    if len(successors_nodes[0]) == 4:
        # This format is used by graph in LS when we create copies of terminals
        adj_nodes = [(n[0], n[1], n[2], n[3]) for n in successors_nodes]
    elif len(successors_nodes[0]) == 3:
        # This format is useed while generating energy dictionaries for dual graph
        adj_nodes = [(n[0], n[1], n[2]) for n in successors_nodes]
    elif len(successors_nodes[0]) == 2:
        # This format is used by graph in LS by primal graph
        adj_nodes = [(n[0], n[1]) for n in successors_nodes]
    else:
        raise ValueError("Check the format of neighbors")
    return adj_nodes


def process_dual_graph(L: networkx.classes.multidigraph.MultiDiGraph) -> dict:
    """
    Given a dual graph, this function creates an adjacency list dictionary.

    Args:
        L (networkx.classes.multidigraph.MultiDiGraph): Dual graph

    Returns:
        adj_list_dict_dual(dict[tuple[int, int, int], ClassInstance]): Key: node in dual graph, Value: Node details in a class instance
    """

    class AdjacencyList:
        def __init__(self):
            self.node = -1
            self.adj_nodes_turns = []
            self.successors_nodes = []

    # create a list of nodes and a dictionary of adjacency lists
    adj_list_dict_dual = {node: AdjacencyList() for node in L.nodes}

    # create the adjacency list for each node
    for node in L.nodes:
        adj_list_dict_dual[node].node = node
        adj_list_dict_dual[node].successors_nodes = get_successors_nodes_in_dual(node, L)
    return adj_list_dict_dual


def get_pareto_set(list_of_tuples: list[tuple[float, float]]) -> np.ndarray:
    """
    This function returns the Pareto set from a list of tuples

    Args:
        list_of_tuples (list[tuple[float, float]]): List of tuples

    Returns:
        mask (np.ndarray): Boolean mask with `True` for observations in the Pareto set.
    """
    objective_values_array = np.vstack([np.array([obj1, obj2]) for obj1, obj2 in list_of_tuples])
    mask = paretoset(objective_values_array, sense=["min", "min"], use_numba=True)
    return mask


def print_experiment_details(ls_constants: dict) -> None:
    """
    This function prints the experiment details

    Args:
        ls_constants (dict): Constants for the local search

    Returns:
        None
    """
    print("*" * 100)
    print("Experiment Details")
    print(f" - allowed_time:              {ls_constants['allowed_time']}")
    print(f" - allowed_init_time:         {ls_constants['allowed_init_time']}")
    print(f" - allowed_ls_time:           {ls_constants['allowed_ls_time']}")
    print(f" - parallelize:               {ls_constants['parallelize']}")
    print(f" - cores:                     {ls_constants['cores']}")
    print(f" - generate_log_file:         {ls_constants['generate_log_file']}")
    print("*" * 100)
    return None


def copy_folder(source_folder: str, destination_folder: str) -> None:
    """
    This function copies a folder from source to destination

    Args:
        source_folder (str): Source folder
        destination_folder (str): Destination folder

    Returns:
        None
    """
    try:
        shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
    return None


def convert_to_csv(ls_constants: dict) -> None:
    """
    This function converts the log file to a CSV file

    Args:
        ls_constants (dict): Constants for the local search

    Returns:
        None
    """

    pd.read_csv(f'./logs/final_LSlog_file.txt').to_csv(f'./logs/final_LSlog_file.csv', index=False)
    with open(f'./logs/configuration_params.pkl', 'wb') as f:
        pickle.dump(ls_constants, f)
    return None


def plot_kd_distribution() -> None:
    """
    This function generates a Kernel Density plot

    Returns:
        None
    """
    consumption_values = [(1, 150), (2, 140), (3, 135), (4, 130), (5, 120), (10, 60), (15, 15), (20, 10), (22, 8), (23, 7), (24, 5), (25, 0)]
    data_x, data_y = zip(*consumption_values)
    # Step 1: Plot consumption values
    plt.figure(figsize=(10, 7))  # Increased figsize for better visibility
    plt.scatter(data_x, data_y, color='b', s=200)
    xticks = np.arange(0, max(data_x) + 1, 5)  # adjust step as required
    yticks = np.arange(25, max(data_y) + 1, 25)  # adjust step as required
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel("Turn count", fontsize=22)
    plt.ylabel("Energy (kWh)", fontsize=22)
    plt.grid(False)
    plt.savefig(f'./kd1.pdf')
    plt.clf()

    # Step 2: Fit density function
    plt.figure(figsize=(10, 7))  # Increased figsize for better visibility
    data = np.vstack([data_x, data_y])
    kde = gaussian_kde(data)
    xmin = np.array(data_x).min()
    xmax = np.array(data_x).max()
    ymin = np.array(data_y).min()
    ymax = np.array(data_y).max()
    x_val, y_val = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([x_val.ravel(), y_val.ravel()])
    z_val = np.reshape(kde(positions).T, x_val.shape)
    colors = [(1, 1, 0), (1, 0, 0)]  # Yellow to blue
    cmap_name = 'yellow_to_blue'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors)

    cf = plt.imshow(z_val, cmap=cm, origin='lower', extent=(0, xmax, 0, ymax), aspect='auto')
    cax = plt.colorbar(cf, location="right")
    cax.set_label('Density', fontsize=20)
    cax.ax.tick_params(labelsize=20)
    # plt.plot(data_x, data_y, 'k.', markersize=2)
    plt.xlabel("Turn count", fontsize=22)
    plt.ylabel("Energy (kWh)", fontsize=22)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(False)
    plt.savefig(f'./kd2.pdf')
    plt.clf()

    # Step 3: Scatter plot with color-coded probabilities
    density = gaussian_kde(data).evaluate(data)
    density = density / np.sum(density)
    pi_d = [(1 - val) for val in density]
    pi_sum = np.sum(pi_d)
    probabilities = [val / pi_sum for val in pi_d]

    plt.figure(figsize=(10, 7))  # Increased figsize for better visibility
    plt.scatter(data_x, data_y, c=probabilities, cmap=cm, edgecolors='k', s=200)
    cax = plt.colorbar(location="right")
    cax.set_label('Probability', fontsize=20)
    cax.ax.tick_params(labelsize=20)
    plt.xlabel("Turn count", fontsize=22)
    plt.ylabel("Energy (kWh)", fontsize=22)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(False)
    plt.savefig(f'./kd3.pdf')
    plt.clf()
    return None


def get_penalty(path_test: list, time_cost_dict: dict, time_window_dict: dict, depot: tuple) -> tuple:
    """
    This function calculates the penalty for a given path

    Args:
        path_test (list[int]): Path or tour
        time_cost_dict (dict): Dictionary of travel times
        time_window_dict (dict): The dictionary containing the time windows for each terminal
        depot (tuple): Depot node

    Returns:
        tour_penalty + waiting (float): Penalty + waiting for the tour
        is_feasible (bool): True if the tour is feasible; False otherwise
        unsatisfied_terminals (dict): Dictionary of unsatisfied terminals
    """
    # first visit to a terminal is considered to define the waiting time and penalty includes waiting time and unsatisfied terminals also include terminals with waiting
    terminal_remaining = set(time_window_dict.keys()) - {depot}
    arrival_time_sofar, tour_waiting = 0, 0
    unsatisfied_terminals_waiting = {}
    unsatisfied_terminals_penalty = {x: [] for x in terminal_remaining}
    terminal_remaining_dict = {x: True for x in terminal_remaining}
    for node_idx, tail_node in enumerate(path_test[:-1]):
        head_node = path_test[node_idx + 1]
        arrival_time_sofar += time_cost_dict[tail_node][head_node]
        try:
            if terminal_remaining_dict[head_node]:
                terminal_satisfied, waiting_penalty = tw_satisfy(arrival_time_sofar, time_window_dict[head_node])
                if terminal_satisfied:
                    terminal_remaining_dict[head_node] = False
                    arrival_time_sofar += waiting_penalty
                    tour_waiting += waiting_penalty
                    if waiting_penalty > 0:
                        unsatisfied_terminals_waiting[head_node] = waiting_penalty
                else:
                    unsatisfied_terminals_penalty[head_node].append(waiting_penalty)
        except KeyError:
            pass
    unsatisfied_terminals = {terminal: min(penalties) for terminal, penalties in unsatisfied_terminals_penalty.items() if penalties}
    tour_penalty = sum(unsatisfied_terminals.values())
    unsatisfied_terminals.update(unsatisfied_terminals_waiting)
    is_feasible = True if tour_penalty == 0 else False
    return tour_penalty + tour_waiting, is_feasible, unsatisfied_terminals


def warm_start_quickest_paths(terminal_set: set, L: networkx.classes.multidigraph.MultiDiGraph, time_cost_dict: dict) -> tuple:
    """
    This function calculates the quickest paths between all pairs of terminals

    Args:
        terminal_set (set): Set of terminal nodes
        L (networkx.classes.multidigraph.MultiDiGraph): Dual Graph
        time_cost_dict (dict): Dictionary of travel times

    Returns:
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
        end (int): Time taken to calculate the quickest paths
    """
    quickest_paths_dict = {}

    def weight_weighted(u, v, d):
        return time_cost_dict[u][v]

    start = time.time()
    for source_terminal in terminal_set:
        _, path_dijk = nx.single_source_dijkstra(L, source_terminal, weight=weight_weighted)
        for target_terminal in terminal_set:
            if source_terminal == target_terminal: continue
            try:
                quickest_paths_dict[(source_terminal, target_terminal)] = path_dijk[target_terminal]
            except KeyError:
                pass
    end = round(time.time() - start)
    return quickest_paths_dict, end


def tw_satisfy(time_taken: float, time_window: tuple) -> tuple:
    """
    This function checks if the time taken by a path is within the time window of the corresponding terminals.

    Args:
        time_taken (float): Time taken by the path to reach the terminal
        time_window (tuple): Time window of the terminal

    Returns:
        True, waiting if the time taken is within the time window; False, waiting_time if the time taken is outside the time window
    """
    if time_window[0] <= time_taken <= time_window[1]:
        # If the time taken is within the time window, then no waiting time
        return True, 0
    elif time_taken <= time_window[0]:
        # If the time taken is less than the upper bound of the time window, then waiting time is upper bound - time taken
        return True, time_window[0] - time_taken
    else:
        return False, time_taken - time_window[1]
        # If the time taken is greater than the upper bound of the time window, then satisfy is False


def get_closest_point(point_set: list[list], target: tuple) -> list:
    """
    Returns the closest point in a set of points to a target point.

    Args:
        point_set (list[list]): List of points
        target (tuple): Target point

    Returns:
        closest_point (list): Closest point in the set of points to the target point
    """
    a, b = target
    min_distance = float('inf')
    closest_point = None
    for point in point_set:
        x, y = point
        distance = math.sqrt((x - a) ** 2 + (y - b) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    return closest_point


def test_solutions(set_z: list[tuple[list[tuple[int, int, int]], float, float]], terminal_set: set, time_cost_dict: dict, time_window_dict: dict, depot: tuple, route_num: int) -> None:
    """
    This function tests the solutions

    Args:
        set_z (list[tuple[list[tuple[int, int, int]], float, float]): List of solutions
        terminal_set (set): Set of terminal nodes
        time_cost_dict (dict): Dictionary of travel times
        time_window_dict (dict): The dictionary containing the time windows for each terminal
        depot (tuple): Depot node
        route_num (int): Route number

    Returns:
        None
    """
    sanity_check_16(terminal_set, set_z)
    for idx, (path, e, t) in enumerate(set_z):
        penalty, feasible, _ = get_penalty(path, time_cost_dict, time_window_dict, depot)
        if not feasible:
            if not feasible: raise ValueError(f"Path {idx} in set_z failed: Time-window not satisfied!")

    # sanity_check_17(set_z, route_num)
    return None


def write_ip_ls_stats(terminal_percent: int, route_id: int, L: networkx.classes.multidigraph.MultiDiGraph, ip_time: float, time_for_last_imp: float, len_bstsp_tours: int, len_bstsptw_tours: int,
                      terminal_list: set[int], ip_tours: list[tuple[list[tuple[int, int, int]], float, float]], set_z: list[tuple[list[tuple[int, int, int]], float, float]]) -> None:
    """
    This function writes the stats of IP and LS in a file

    Args:
        terminal_percent (int): Percentage of terminals
        route_id (int): Route ID
        L (networkx.classes.multidigraph.MultiDiGraph): Dual Graph
        ip_time (float): Time taken by IP
        time_for_last_imp (float): Time for last improvement in LS
        len_bstsp_tours (int): Number of BSTSP tours in LS
        len_bstsptw_tours (int): Number of BSTSPTW tours in LS
        terminal_list (set[int]): Set of terminals
        ip_tours (list[tuple[list[tuple[int, int, int]], float, float]]): List of IP tours
        set_z (list[tuple[list[tuple[int, int, int]], float, float]]): List of LS tours

    Returns:
        None

    """
    if set_z:
        ls_paths, ls_energy, ls_turns = zip(*set_z)
        ls_energy, ls_turns, ls_paths = zip(*sorted(zip(ls_energy, ls_turns, ls_paths)))
    else:
        ls_energy, ls_turns, ls_paths = [], [], []

    if ip_tours:
        ip_energy, ip_turns, ip_paths = zip(*ip_tours)
        ip_energy, ip_turns, ip_paths = zip(*sorted(zip(ip_energy, ip_turns, ip_paths)))
    else:
        ip_energy, ip_turns, ip_paths = [], [], []

    with open(f"./logs/IPRes/{terminal_percent}_Instance_{route_id}.txt", 'w') as file:
        file.write(f"Graph Nodes: {len(L.nodes)} and Graph Edges {len(L.edges())}\n")
        file.write(f"Number of terminals: {len(terminal_list)}\n")
        file.write(f"Terminals: {terminal_list}\n")
        file.write("Time:\n")
        file.write(f"   IP: {round(ip_time)}\n")
        file.write(f"   LS: {round(time_for_last_imp, 1)}\n")
        file.write("LS Scalerization:\n")
        file.write(f"   BSTSP  : {len_bstsp_tours}\n")
        file.write(f"   BSTSPTW: {len_bstsptw_tours}\n")
        file.write("IP Paths:\n")
        for path in ip_paths:
            file.write(f"   {path}\n")
        file.write("LS Paths:\n")
        for path in ls_paths:
            file.write(f"   {path}\n")
        file.write(f"IP Solution: {ip_energy}, {ip_turns}\n")
        file.write(f"LS Solution: {ls_energy}, {ls_turns}\n")
    return None


def plot_ip_figures(ip_tours: list[tuple[list[tuple[int, int, int]], float, float]], set_z: list[tuple[list[tuple[int, int, int]], float, float]], instance_id: str, route_id: int,
                    terminal_percent: int) -> None:
    """
    This function plots the energy vs turns for IP and LS solutions

    Args:
        ip_tours (list[tuple[list[tuple[int, int, int]], float, float]]): List of IP tours
        set_z (list[tuple[list[tuple[int, int, int]], float, float]]): List of LS tours
        instance_id (str): Instance ID
        route_id (int): Route ID
        terminal_percent (int): Percentage of terminals

    Returns:
        None
    """
    if set_z:
        ls_paths, ls_energy, ls_turns = zip(*set_z)
        ls_energy, ls_turns, ls_paths = zip(*sorted(zip(ls_energy, ls_turns, ls_paths)))
    else:
        ls_energy, ls_turns, ls_paths = [], [], []

    if ip_tours:
        ip_energy, ip_turns, ip_paths = zip(*ip_tours)
        ip_energy, ip_turns, ip_paths = zip(*sorted(zip(ip_energy, ip_turns, ip_paths)))
    else:
        ip_energy, ip_turns, ip_paths = [], [], []

    plt.figure(figsize=(9, 7))  # Adjust figure size for better visualization
    # plt.scatter(ls_turns, ls_energy, facecolors='none', edgecolors='magenta', marker='o', s=300)
    # plt.plot(ls_turns, ls_energy, 'ko-', label='LS', color="magenta", linewidth=4, markersize=0, alpha=0.5)
    plt.plot(ls_turns, ls_energy, 'ko-', label='LS', color="magenta", linewidth=4, markerfacecolor='none', markersize=20, marker='o', alpha=0.5, markeredgewidth=4)
    plt.plot(ip_turns, ip_energy, 'gs--', label='MIP', linewidth=2, markersize=14, marker='X', color="black")
    plt.xlabel("Turn count", fontsize=22)
    plt.ylabel("Energy (kWh)", fontsize=22)
    plt.title(f"Instance Id {instance_id} ({terminal_percent}% case)", fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(False)
    plt.legend(loc='upper right', fontsize=24, frameon=True, edgecolor='black', shadow=True)
    plt.savefig(f"./logs/IPRes/{terminal_percent}_Instance_{route_id}.pdf")
    # plt.show()
    plt.clf()
    return None
