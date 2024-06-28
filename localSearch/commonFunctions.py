from copy import deepcopy
from collections import defaultdict, Counter

from helpers.loggers import *
from helpers.functions import *


def update_parents(parents: dict, curr_tour: list, new_path: list, operator_name: str) -> dict:
    """
    Update the parents dictionary with the new path and its parent path

    Args:
        parents (dict): Dictionary of parents for tracking lineages
        curr_tour (list): Current tour
        new_path (list): The new tour
        operator_name (str): The name of the neighborhood operator

    Returns:
        dict: The updated parents dictionary
    """
    if len(new_path) == 0:
        return parents
    new_path_tup = tuple(new_path)
    if new_path_tup in parents.keys():
        parents[new_path_tup].append((curr_tour, operator_name))
    else:
        parents[new_path_tup] = [(curr_tour, operator_name)]
    return parents


def get_lineages_in_set_z(set_z: list, initial_paths: set, parents: dict) -> dict:
    """
    Get the lineages of the paths in the set z

    Args:
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        initial_paths (set): The set of initial paths
        parents (dict): Dictionary of parents for tracking lineages

    Returns:
        dict: The dictionary containing the lineages of the paths in the set z
    """
    all_nbds = []
    for path, _, _ in set_z:
        parent_path = path
        while parent_path not in initial_paths:
            parent_path, nbd_used = parents[parent_path][0]
            all_nbds.append(nbd_used)
    path_nbd_counts = Counter(all_nbds)
    return path_nbd_counts


def reorder_and_fix_tour(tour: list, depot: tuple):
    """
    This function reorders a path to start and end at the depot.

    Args:
        tour (list[tuple[int, int, int]]): A path in the Dual Graph
        depot (tuple): Depot node

    Returns:
        Reordered path starting and ending at the depot
    """
    if depot != tour[0]:
        depot_index = tour.index(depot)
        if tour[0] == tour[-1]:
            return tour[depot_index:] + tour[1:depot_index + 1]
        else:
            return tour[depot_index:] + tour[:depot_index + 1]
    return tour


def initialize_operator(progress_file: str) -> str:
    """
    This function initializes an operator.

    Args:
        progress_file (str): File to write progress to

    Returns:
        str: The log string
    """
    log_line("  Enter\n", progress_file, False)
    operator_stream = ""
    return operator_stream


def post_process_operator(move_time: float, update_set_time: float, operator_call_count: dict, operator_name: str, operator_stream: str, operator_times: dict, nbd_tw_feasible_tours: dict,
                          nbd_optimal_tours: dict, common_arguments: dict) -> None:
    """
    This function runs the post processing logic associated with an operator.

    Args:
        move_time (float): The time taken to move
        update_set_time (float): The time taken to update the set
        operator_call_count (dict): Call counts of each operator
        operator_name (str): The name of the neighborhood operator
        operator_stream (str): Log string for the operator
        operator_times (dict): Times taken by each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal paths for each operator
        ls_iteration_no (int): Local search iteration number
        common_arguments (dict): Dictionary containing common arguments

    Returns:
        None
    """
    route_num = common_arguments["route_num"]
    progress_file = common_arguments["progress_file"]
    operator_call_count[operator_name] += 1
    operator_times[operator_name] += move_time
    t_time = move_time + update_set_time
    move_time = round(move_time / t_time * 100, 1)
    update_set_time = round(update_set_time / t_time * 100, 1)
    operator_stream += f"{round(t_time, 1)},{move_time},{update_set_time},{operator_call_count[operator_name]},{len(nbd_tw_feasible_tours[operator_name])},{len(nbd_optimal_tours[operator_name])},{common_arguments['ls_iteration_no']}\n"
    filename = f"./logs/{route_num}/{operator_name}.txt"
    log_line(operator_stream, filename, False)
    log_line("  Exit\n", progress_file, False)
    return None


def get_max_penalty_in_set_p(set_p: list) -> float:
    """
    This function gets the maximum penalty for tours belonging to Set P.

    Args:
        set_p (list): List of tours that violate time-windows

    Returns:
        max_penalty_in_explore_set (float): The maximum penalty in the explore set
    """
    max_penalty_in_explore_set = -np.inf
    if set_p:
        max_penalty_in_explore_set = sorted(set_p, key=lambda x: (x[3]))[-1][3]
    return max_penalty_in_explore_set


def update_sets(output_tours: list, curr_tour: list, operator_name: str, parents: dict, set_z: list, set_p: list, set_y: list, nbd_tw_feasible_tours: dict, nbd_optimal_tours: dict, no_imp_count: int,
                time_for_last_imp: float, operator_stream: str, common_arguments: dict) -> tuple:
    """
    This function updates Sets P, Y, and Z.

    Args:
        output_tours (list): The list of output tours
        curr_tour (list): The parent tour
        operator_name (str): The name of the neighborhood operator
        parents (dict): Dictionary of parents for tracking lineages
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal paths for each operator
        no_imp_count (int): Number of iterations without improvement
        time_for_last_imp (float): Time for last improvement in LS
        operator_stream (str): Log string for the operator
        common_arguments (dict): Dictionary containing common arguments

    Returns:
        tuple: The updated sets Z, P, and T
    """
    ls_constants = common_arguments["ls_constants"]
    time_cost_dict = common_arguments["time_cost_dict"]
    time_window_dict = common_arguments["time_window_dict"]
    depot = common_arguments["depot"]
    route_num = common_arguments["route_num"]
    progress_file = common_arguments["progress_file"]
    ls_start_time = common_arguments["ls_start_time"]

    # log_line(" U1-", progress_file, False)
    start = time.time()
    initial_count = len(output_tours)
    if output_tours:
        output_tours = [(new_tour, *get_path_length_using_energy_turns(new_tour, common_arguments["combined_dict"])) for new_tour in output_tours]
    # log_line(" U1.1-", progress_file, False)

    max_penalty_in_explore_set = get_max_penalty_in_set_p(set_p)
    # log_line(" U1.2-", progress_file, False)
    tw_feasible_paths_count, tw_infeasible_paths_count, pareto_paths_count, non_pareto_paths_count, explore_path_count, non_explore_path_count = 0, 0, 0, 0, 0, 0
    tw_feasible_paths = []
    init_time = time.time() - start
    start = time.time()
    # log_line("U2-", progress_file, False)
    ministart = time.time()
    output_tours = [(reorder_and_fix_tour(new_tour, common_arguments["depot"]), e, t) for (new_tour, e, t) in output_tours]
    reorder_time = time.time() - ministart
    ministart = time.time()
    output_tours = [(new_tour, e, t, *get_penalty(new_tour, time_cost_dict, time_window_dict, depot)) for (new_tour, e, t) in output_tours]
    penalty_time = time.time() - ministart
    # print([len(new_tour) for (new_tour, e, t, _penalty, is_feasible, _) in output_tours])
    for (new_tour, e, t, _penalty, is_feasible, _) in output_tours:
        parents = update_parents(parents, curr_tour, new_tour, operator_name)
        if is_feasible:
            tw_feasible_paths_count += 1
            tw_feasible_paths.append((new_tour, e, t))
            nbd_tw_feasible_tours[operator_name].append((new_tour, e, t))
        else:
            tw_infeasible_paths_count += 1
            if _penalty < max_penalty_in_explore_set:
                max_penalty_in_explore_set = _penalty
                set_p.insert(0, (new_tour, e, t, _penalty))
                explore_path_count += 1
            elif len(set_p) < ls_constants["set_p_limit"]:
                set_p.append((new_tour, e, t, _penalty))
                explore_path_count += 1
            else:
                non_explore_path_count += 1
    twtime = time.time() - start
    # log_line("U3-", progress_file, False)
    start = time.time()
    new_ef_set = set_z + tw_feasible_paths
    if new_ef_set:
        obj_tuples = [(e, t) for _, e, t in new_ef_set]
        mask_values = get_pareto_set(obj_tuples)
        if False in mask_values[0: len(set_z)] or True in mask_values[len(set_z):]:
            # Either an old path is knocked out or a new path is added
            no_imp_count = 0
            time_for_last_imp = time.time() - ls_start_time
        for i, flag in enumerate(mask_values):
            if i < len(set_z): continue
            if flag:
                pareto_paths_count += 1
                nbd_optimal_tours[operator_name].append(new_ef_set[i])
            else:
                if operator_name != "FixedPerm":
                    set_y.append(new_ef_set[i])
                non_pareto_paths_count += 1
        set_z = [path for path, flag in zip(new_ef_set, mask_values) if flag]
        set_z = sorted(set_z, key=lambda x: (x[1]))
    # log_line("U4-", progress_file, False)
    # Pareto front on set t
    # obj_tuples = [(e, t) for _, e, t in set_y]
    # mask_values = get_pareto_set(obj_tuples)
    # set_y = [path for path, flag in zip(set_y, mask_values) if flag]
    # if len(set_y) > ls_constants["set_y_limit"]:
    #     set_y = get_random_sample(set_y, ls_constants["set_y_limit"])
    if len(set_y) >= ls_constants["set_y_limit"]:
        i = get_random([1, 2], [])
        set_y = sorted(set_y, key=lambda x: (x[i]))[:ls_constants["set_y_limit"]]
    if len(set_p) >= ls_constants["set_p_limit"]:
        set_p = sorted(set_p, key=lambda x: (x[3]))[:ls_constants["set_p_limit"]]
    operator_stream += f"{initial_count},{len(tw_feasible_paths)},{tw_infeasible_paths_count},{pareto_paths_count},{non_pareto_paths_count},{explore_path_count},{non_explore_path_count},"
    eftime = time.time() - start
    update_set_time = init_time + twtime + eftime
    log_string = f"{operator_name},{initial_count},{round(init_time)},{round(twtime)},{round(reorder_time)},{round(penalty_time)},{round(eftime)},{round(update_set_time)}\n"
    log_line(log_string, f"./logs/{route_num}/updateSet.txt", False)
    # log_line("U5 ", progress_file, False)
    # print(log_string)
    # if update_set_time > 250: exit()
    return set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream, update_set_time


def get_quickest_path(L: networkx.classes.multidigraph.MultiDiGraph, source: int, destination: int, time_cost_dict: dict, quickest_paths_dict: dict) -> tuple:
    """
    This function gets the path with the least travel time between two nodes.

    Args:
        L (networkx.classes.multidigraph.MultiDiGraph): Line Graph
        source (int): Source node
        destination (int): Destination node
        time_cost_dict (dict): Dictionary of travel times
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals

    Returns:
        path (list): The quickest path
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
    """
    try:
        path = quickest_paths_dict[(source, destination)]
    except KeyError:
        def timecost(u, v, d):
            return time_cost_dict[u][v]

        path = nx.astar_path(L, source, destination, weight=timecost)
        quickest_paths_dict[(source, destination)] = path
    return path, quickest_paths_dict


def get_time_left(max_time: float, start_stamp: float) -> bool:
    """
    Check if there is time left

    Args:
        max_time (float): The maximum time allowed
        start_stamp (float): The start time

    Returns:
        value (bool): True if there is time left, False otherwise
    """
    return time.time() - start_stamp < max_time


def initialize_ls(energy_dual: dict, direction_dual: dict, L: networkx.classes.multidigraph.MultiDiGraph, bstsptw_tours: list, bstsp_tours: list, ls_constants: dict, time_window_dict: dict,
                  route_num: int) -> tuple:
    """
    Initialize the local search

    Args:
        energy_dual (dict): Energy cost
        direction_dual (dict): Direction cost
        L (networkx.classes.multidigraph.MultiDiGraph): Line Graph
        bstsptw_tours (list): BSTSPTW tours from scalerization
        bstsp_tours (list): BSTSP tours from scalerization
        ls_constants (dict): Constants for the local search
        time_window_dict (dict): The dictionary containing the time windows for each terminal
        route_num (int): Route number

    Returns:
        value (tuple): The initial values for the local search

    """
    progress_file = f"./logs/{route_num}/progressTrack.txt"
    log_line("Initializing LS\n", progress_file, False)

    no_imp_count = -1
    ls_iteration_no = 0

    combined_dict = {i: {j: (energy_dual[i][j], direction_dual[i][j]) for j in L.neighbors(i)} for i in L.nodes()}

    operator_call_count = {'FixedPerm': 0, 'S3opt': 0, 'S3optTW': 0, 'Quad': 0, 'GapRepair': 0, "RandomPermute": 0, "SimpleCycle": 0}
    operator_times = {'FixedPerm': 0, 'S3opt': 0, 'S3optTW': 0, 'Quad': 0, 'GapRepair': 0, "RandomPermute": 0, "SimpleCycle": 0}
    nbd_tw_feasible_tours = {'FixedPerm': [], 'S3opt': [], 'S3optTW': [], 'Quad': [], 'GapRepair': [], "RandomPermute": [], "SimpleCycle": []}
    nbd_optimal_tours = {'FixedPerm': [], 'S3opt': [], 'S3optTW': [], 'Quad': [], 'GapRepair': [], "RandomPermute": [], "SimpleCycle": []}

    ls_start_time, time_for_last_imp = time.time(), time.time()

    parents = defaultdict(list)

    set_z = deepcopy(bstsptw_tours)
    set_p = deepcopy(bstsp_tours)
    set_y = []
    initial_paths1 = set([path for path, _, _ in set_z])
    initial_paths2 = set([path for path, _, _, _ in set_p])
    initial_paths = initial_paths1.union(initial_paths2)
    max_penalty_in_explore_set = get_max_penalty_in_set_p(set_p)
    common_arguments_temp = {"ls_constants": ls_constants, "route_num": route_num, "ls_iteration_no" : 0, "progress_file": progress_file}
    log_ls_iteration(set_z, set_p, set_y, 0, max_penalty_in_explore_set, common_arguments_temp)
    terminal_pair_locked = {(i, j): False for i in time_window_dict.keys() for j in time_window_dict.keys() if i != j}
    allowed_ls_time = ls_constants["allowed_ls_time"]
    return (no_imp_count, ls_iteration_no, combined_dict, ls_start_time, time_for_last_imp, operator_call_count, operator_times,
            nbd_tw_feasible_tours, nbd_optimal_tours, parents, set_z, set_p, set_y, initial_paths, terminal_pair_locked, progress_file, allowed_ls_time)
