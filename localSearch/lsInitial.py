"""
This file contains the functions to run the scalerization algorithm for the local search
"""
import time
import pickle
import warnings
import networkx
from collections import defaultdict

from helpers.orTools import solve_tsp_using_or_tools, get_tsp_routes
from helpers.shortestPath import johnson_shortest_paths
from localSearch.commonFunctions import get_quickest_path, get_time_left
from helpers.loggers import log_johnson, log_ortools_failure, log_endpoints, log_line, log_scalerization
from helpers.functions import remove_consecutive_duplicates, get_path_length, get_penalty, get_pareto_set, cyclize

warnings.filterwarnings("ignore")


def mixintprog(alpha: float, beta: float, energy_dual: dict, direction_dual: dict, terminal_set: set, L: networkx.classes.multidigraph.MultiDiGraph, route_num: int, ls_constants: dict) -> tuple:
    """
    This function is used to get the initital solution that can be used as an input to the local search algorith. This involves the following steps.
 1. Create a single-criteria Steiner TSP.
 2. Convert the Steiner TSP into a TSP using Johnson's algorithm.
 3. Solve the TSP using Google OR Tools.

    Args:
        alpha (float): Direction weight
        beta (float): Energy weight
        energy_dual (dict): Energy cost
        direction_dual (dict): Direction cost
        terminal_set (set): Set of terminal nodes
        L (networkx.classes.multidigraph.MultiDiGraph): Line Graph
        route_num (int): Route number
        ls_constants (dict): Constants for the local search

    Returns:
        tour (list): New tour
        tour_energy (int): Energy of the tour
        tour_turns (int): Direction of the tour
        cost_paths (dict): Updated dictionary of paths
    """
    terminal_list = list(terminal_set)
    # beta, alpha  = (1, 0.00001)
    or_factor = ls_constants["or_factor"]

    part1_start = time.time()
    cost_matrix = [[1000000000000000 for _ in range(len(terminal_set))] for _ in range(len(terminal_set))]
    cost_paths = {}

    cost_dict = defaultdict(dict)
    for i in L.nodes():
        for j in L.neighbors(i):
            cost_dict[i][j] = beta * energy_dual[i][j] + alpha * direction_dual[i][j]

    cost_johnson, bellman_time, terminal_time = johnson_shortest_paths(L, cost_dict, terminal_set)

    if cost_johnson is None:
        log_johnson(route_num)
        return None, None, None, None
    for s_node_id, s_node in enumerate(terminal_set):
        for d_node_id, d_node in enumerate(terminal_set):
            if s_node != d_node:
                shortest_path_dist = cost_johnson[s_node][d_node][0]
                shortest_path = cost_johnson[s_node][d_node][1]
                cost_matrix[s_node_id][d_node_id] = int(shortest_path_dist * or_factor)
                cost_paths[(s_node, d_node)] = shortest_path
    # _, energy_routes = lkh3_tsp_solver(cost_matrix)
    part1 = time.time() - part1_start
    part2_start = time.time()
    solution, routing, manager = solve_tsp_using_or_tools(cost_matrix, or_factor)
    if solution is None:
        log_ortools_failure(route_num)
        return None, None, None, None
    tsp_routes = get_tsp_routes(solution, routing, manager)
    tour, tour_energy, tour_turns = -1, -1, -1
    for routes in tsp_routes:
        path = []
        for terminal_index in range(len(routes) - 1):
            path = path + cost_paths[(terminal_list[routes[terminal_index]], terminal_list[routes[terminal_index + 1]])]
        tour = remove_consecutive_duplicates(path)
        tour_energy = get_path_length(list(tour), energy_dual)
        tour_turns = get_path_length(list(tour), direction_dual)
        # sanity_check_13(tour, terminal_set)
    log_mixintprog(part2_start, terminal_time, part1, bellman_time, route_num, alpha, beta)
    if tour == -1 or tour_energy== -1 or tour_turns == -1:
        raise ValueError("Tour not found")
    return tour, tour_energy, tour_turns, cost_paths


def log_mixintprog(part2_start: float, terminal_time: float, part1: float, bellman_time: float, route_num: int, alpha: float, beta: float) -> None:
    """
    This function logs the results of the mixintprog algorithm

    Args:
        part2_start (float): Start time of part 2
        terminal_time (float): Time taken to find the shortest paths
        part1 (float): Time taken for part 1
        bellman_time (float): Time taken for bellman ford
        route_num (int): Route number
        alpha (float): Direction weight
        beta (float): Energy weight

    Returns:
        None
    """
    part2 = time.time() - part2_start
    tot = part1 + part2
    part1_percent = round(part1 / tot * 100)
    part2_percent = round(part2 / tot * 100)
    bellman_per = round(bellman_time / part1 * 100)
    terminal_per = round(terminal_time / part1 * 100)
    line = f"{route_num},{alpha},{beta},{round(part1)},{round(part2)},{part1_percent},{part2_percent},{bellman_time},{terminal_time},{round(bellman_per)},{round(terminal_per)}\n"
    log_line(line, f'./logs/{route_num}/mixintprog.txt', False)
    log_line(line, f'./logs/mixintprog.txt', False)
    return None


def save_scalerization_results(route_num: int, bstsptw_tours: list, bstsp_tours: list, weighted_paths_dict: dict) -> None:
    """
    This function saves the results of the scalerization algorithm

    Args:
        route_num (int): Route number
        bstsptw_tours (list): BSTSPTW tours from scalerization
        bstsp_tours (list): BSTSP tours from scalerization
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals

    Returns:
        None
    """
    folder_path = f'./logs/{route_num}'
    with open(f'{folder_path}/scalerization/bstsptw_tours_{route_num}.pkl', 'wb') as f:
        pickle.dump(bstsptw_tours, f)
    with open(f'{folder_path}/scalerization/bstsp_tours_{route_num}.pkl', 'wb') as f:
        pickle.dump(bstsp_tours, f)
    with open(f'{folder_path}/scalerization/weighted_paths_dict_{route_num}.pkl', 'wb') as f:
        pickle.dump(weighted_paths_dict, f)
    # with open(f'{folder_path}/scalerization/quickest_paths_dict_{route_num}.pkl', 'wb') as f:
    #     pickle.dump(quickest_paths_dict, f)
    return None


def scalerization(energy_dual: dict, direction_dual: dict, terminal_set: set, L: networkx.classes.multidigraph.MultiDiGraph, ls_constants: dict, route_num: int, time_cost_dict: dict,
                  time_window_dict: dict, depot: tuple) -> None:
    """
    This is the main function to get the efficiency frontier by executing the local search-based scalarization algorithm.

    Args:
        energy_dual (dict): Energy cost
        direction_dual (dict): Direction cost
        terminal_set (set): Set of terminal nodes
        L (networkx.classes.multidigraph.MultiDiGraph): Line Graph
        ls_constants (dict): Constants for the local search
        route_num (int): Route number
        time_cost_dict (dict): Dictionary of travel times
        time_window_dict (dict): The dictionary containing the time windows for each terminal
        depot (tuple): Depot node

    Returns:
        None

    """
    scalerization_start_stamp = time.time()
    weighted_paths_dict = {(u, v): set() for u in terminal_set for v in terminal_set if u != v}
    scalerization_tours = []
    energy_1, energy_2, turns_1, turns_2 = None, None, None, None
    almost_zero = 0.001
    filename = f'./logs/scalerizationEndConditions.txt'
    progress_file = f"./logs/{route_num}/progressTrack.txt"
    log_line("Scalerization Enter\n", progress_file, False)

    def newtour(energy_1, energy_2, turns_1, turns_2, depth, scalerization_tours) -> None:
        """
        This function finds the mid points of the efficiency frontier between two points using recursion
        """
        if energy_1 is None or energy_2 is None or turns_1 is None or turns_2 is None:
            log_line(f"{route_num},1\n", filename, False)
            return
        if (round(energy_2, 1) == round(energy_1, 1)) or (turns_2 == turns_1):
            log_line(f"{route_num},2\n", filename, False)
            return
        if not get_time_left(ls_constants["allowed_init_time"], scalerization_start_stamp):
            log_line(f"{route_num},3\n", filename, False)
            return

        w = abs((energy_2 - energy_1) / (turns_2 - turns_1))
        optimal_paths, energy_3, turns_3, path_set = mixintprog(w / (w + 1), 1 / (w + 1), energy_dual, direction_dual, terminal_set, L, route_num, ls_constants)
        if optimal_paths is None:
            log_line(f"{route_num},4\n", filename, False)
            return
        if (round(energy_3), turns_3) == (round(energy_1), turns_1) or (round(energy_3), turns_3) == (round(energy_2), turns_2):
            log_line(f"{route_num},5\n", filename, False)
            return
        if depth > ls_constants["max_recursion_depth"]:
            log_line(f"{route_num},6\n", filename, False)
            return

        scalerization_tours.append(optimal_paths)
        for (u, v), path_ in path_set.items():
            weighted_paths_dict[(u, v)].add(tuple(path_))

        depth += 1
        newtour(energy_3, energy_2, turns_3, turns_2, depth, scalerization_tours)
        newtour(energy_1, energy_3, turns_1, turns_3, depth, scalerization_tours)

    start = time.time()
    if get_time_left(ls_constants["allowed_init_time"], scalerization_start_stamp):
        log_line("  Finding least energy tour\n", progress_file, False)
        least_energy_tour, energy_1, turns_1, paths_1 = mixintprog(almost_zero, 1, energy_dual, direction_dual, terminal_set, L, route_num, ls_constants)
        if energy_1 is not None:
            scalerization_tours.extend([least_energy_tour])
            for (u, v), path_ in paths_1.items():
                weighted_paths_dict[(u, v)].add(tuple(path_))
    least_energy_tour_time = round(time.time() - start)

    start = time.time()
    if get_time_left(ls_constants["allowed_init_time"], scalerization_start_stamp):
        log_line("  Finding least turn tour\n", progress_file, False)
        least_turn_tour, energy_2, turns_2, paths_2 = mixintprog(1, almost_zero, energy_dual, direction_dual, terminal_set, L, route_num, ls_constants)
        if energy_2 is not None:
            scalerization_tours.append(least_turn_tour)
            for (u, v), path_ in paths_2.items():
                weighted_paths_dict[(u, v)].add(tuple(path_))
    least_turn_tour_time = round(time.time() - start)

    log_endpoints(route_num, scalerization_start_stamp, scalerization_tours, least_turn_tour_time, least_energy_tour_time)
    depth = 0
    newtour(energy_1, energy_2, turns_1, turns_2, depth, scalerization_tours)
    quickest_paths_dict = {}
    bstsptw_tours, bstsp_tours, found_flag, quickest_paths_dict = scalerization_post_process(scalerization_tours, terminal_set, energy_dual, direction_dual, time_cost_dict, time_window_dict,
                                                                                             quickest_paths_dict, L, depot)
    log_scalerization(route_num, bstsptw_tours, bstsp_tours, scalerization_start_stamp)
    log_line("Scalerization Exit.\n", progress_file, False)
    save_scalerization_results(route_num, bstsptw_tours, bstsp_tours, weighted_paths_dict)
    return None


def scalerization_post_process(scalerization_tours: list, terminal_set: set, energy_dual: dict, direction_dual: dict, time_cost_dict: dict, time_window_dict: dict, quickest_paths_dict: dict,
                               L: networkx.classes.multidigraph.MultiDiGraph, depot: tuple) -> tuple:
    """
    This function is used to process the results obtained from the local search-based scalarization method.

    Args:
        scalerization_tours (list): list of tours
        terminal_set (set): Set of terminal nodes
        energy_dual (dict): Energy cost
        direction_dual (dict): Direction cost
        time_cost_dict (dict): Dictionary of travel times
        time_window_dict (dict): The dictionary containing the time windows for each terminal
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
        L (networkx.classes.multidigraph.MultiDiGraph): Line Graph
        depot (tuple): Depot node

    Returns:
        bstsptw_tours (list): BSTSPTW tours from scalerization
        bstsp_tours (list): BSTSP tours from scalerization
        found_flag (bool): Flag to indicate if the tours were found
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
    """
    found_flag = True
    if not scalerization_tours:
        found_flag = False
        scalerization_tours, quickest_paths_dict = get_minimum_time_tour(time_cost_dict, quickest_paths_dict, terminal_set, L)
    bstsptw_tours, bstsp_tours = [], []
    for path in scalerization_tours:
        path = tuple(cyclize(path, path.index(depot)))
        e, t = get_path_length(path, energy_dual), get_path_length(path, direction_dual)
        tour_penalty, is_feasible, _ = get_penalty(path, time_cost_dict, time_window_dict, depot)
        if is_feasible:
            bstsptw_tours.append((path, e, t))
        else:
            bstsp_tours.append((path, e, t, tour_penalty))
    if bstsp_tours:
        bstsp_tours = set(bstsp_tours)
        bstsp_tours = sorted(bstsp_tours, key=lambda x: (x[3]))
    if bstsptw_tours:
        bstwsptw_objs = [(path[1], path[2]) for path in bstsptw_tours]
        mask_values = get_pareto_set(bstwsptw_objs)
        bstsptw_tours = [path for path, bool_flag in zip(bstsptw_tours, mask_values) if bool_flag]
        bstsptw_tours = sorted(bstsptw_tours, key=lambda x: (x[1]))
    return bstsptw_tours, bstsp_tours, found_flag, quickest_paths_dict


def get_minimum_time_tour(time_cost_dict: dict, quickest_paths_dict: dict, terminal_set: set, L: networkx.classes.multidigraph.MultiDiGraph) -> tuple:
    """
    This function finds the minimum-time tour between the terminals.
    """
    terminal_list = list(terminal_set)
    quickest_tour = []
    for idx in range(len(terminal_set) - 1):
        seg, quickest_paths_dict = get_quickest_path(L, terminal_list[idx], terminal_list[idx + 1], time_cost_dict, quickest_paths_dict)
        quickest_tour.extend(seg)
    scalerization_tours = [remove_consecutive_duplicates(quickest_tour)]
    return scalerization_tours, quickest_paths_dict

'''
def check_negative_cycle(alpha: float, beta: float, energy_dual: dict, direction_dual: dict, L: networkx.classes.multidigraph.MultiDiGraph, route_num: int) -> None:
    """
    This function checks for negative cycles in the graph
    """
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
    line = f"{route_num},{alpha},{beta},{flag}\n"
    log_line(line, "./negative_cycle.txt", False)
    raise ValueError("Unknown function called")
'''