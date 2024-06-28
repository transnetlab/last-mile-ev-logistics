"""
This file contains functions related to FixedPerm operator
"""
from itertools import product

from helpers.functions import *
from localSearch.commonFunctions import get_time_left
from localSearch.s3opt import update_weighted_paths_dict
from helpers.shortestPath import call_with_timeout, biobj_label_correcting


def fixed_perm_operator(weighted_paths_dict: dict, curr_tour: list, operator_stream: str, terminal_pair_locked: dict, move_start_stamp: float, common_arguments: dict) -> tuple:
    """
    This function runs a single iteration of the Fixed Perm operator.
    
    Args:
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        curr_tour (list): Current tour
        operator_stream (str): Log string for the operator
        curr_tour The log string for the neighborhood operator
        ls_iteration_no (int): Local search iteration number
        terminal_pair_locked (dict): Dictionary of terminal pairs that are locked
        move_start_stamp (float): Time the move started
        common_arguments (dict): Common arguments for the local search

    Returns:
        tuple: Tuple containing the new paths, the weighted paths dictionary, the log string, and the terminal pair locked dictionary
    """
    energy_dual = common_arguments["energy_dual"]
    direction_dual = common_arguments["direction_dual"]
    combined_dict = common_arguments["combined_dict"]

    if not curr_tour:
        operator_stream += f"{0},{0},{0},{0},{0},{0},{0},"
        return [], weighted_paths_dict, operator_stream, terminal_pair_locked
    max_nb_runtime = common_arguments["ls_constants"]['max_nb_runtime']
    ls_sp_timeout_limit = common_arguments["ls_constants"]['ls_sp_timeout_limit']
    fp_flip_count = common_arguments["ls_constants"]['fp_flip_count']
    fp_terminal_count = common_arguments["ls_constants"]['fp_terminal_count']

    obj_selected = get_random(['energy', 'turns'], [] )
    terminal_indices = get_terminal_position(curr_tour, common_arguments["terminal_set"])
    pr = get_fixed_perm_terminal_priorities(terminal_indices, curr_tour, common_arguments, str(obj_selected))
    flag = "normal"
    if pr == {}:
        flag = "forced"
        # This happens if obj_selected == turns and the number of turns in curr_tour is zero.
        obj_selected = 'energy'
        pr = get_fixed_perm_terminal_priorities(terminal_indices, curr_tour, common_arguments, "energy", )
    if pr == {}:
        # Required for Ip. This happens in fpm : Use mean for curr path just
        mean_energy_dual = sum([energy_dual[curr_tour[node]][curr_tour[node + 1]] for node in range(len(curr_tour) - 1)]) / (len(curr_tour) - 1)
        mean_turn_dual = sum([direction_dual[curr_tour[node]][curr_tour[node + 1]] for node in range(len(curr_tour) - 1)]) / (len(curr_tour) - 1)
        pr = get_fixed_perm_terminal_priorities(terminal_indices, curr_tour, combined_dict, mean_energy_dual, mean_turn_dual, "energy")
    if pr == {}:
        operator_stream += f"{0},{0},{0},{0},{0},{0},{0},"
        return [], weighted_paths_dict, operator_stream, terminal_pair_locked
    pr_items = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    terminals_pair_sorted = [ter_pair for ter_pair, prob in pr_items]
    flipped_count, fetched_count, timeout_count, tcount = 0, 0, 0, 0
    p_set = [[] for _ in range(len(terminal_indices) - 1)]
    terminal_pair_mapping = {(terminal_indices[i], terminal_indices[i + 1]): i for i in range(len(terminal_indices) - 1)}
    start = time.time()
    # log_line(" 0 ", progress_file, False)
    terminals_with_two_paths = 0
    for (u_idx, u, v_idx, v) in terminals_pair_sorted:
        tcount += 1
        fetch_it = True
        if get_time_left(max_nb_runtime, move_start_stamp) and terminals_with_two_paths < fp_terminal_count:
            if not terminal_pair_locked[u, v]:
                new_paths, _ = call_with_timeout(ls_sp_timeout_limit, biobj_label_correcting, combined_dict, u, v)
                terminal_pair_locked[u, v] = True
                weighted_paths_dict = update_weighted_paths_dict(weighted_paths_dict, u, v, new_paths)
            else:
                new_paths = weighted_paths_dict[u, v]
            if new_paths:
                flipped_count += 1
                new_paths_shuffled = custom_shuffle(list(new_paths))[:fp_flip_count]
                if len(new_paths_shuffled) == 2:
                    terminals_with_two_paths += 1
                for newpath in new_paths_shuffled:
                    p_set[terminal_pair_mapping[u_idx, v_idx]].append(newpath)
                fetch_it = False
        if fetch_it:
            fetched_count += 1
            p_set[terminal_pair_mapping[u_idx, v_idx]].append(curr_tour[u_idx: v_idx + 1])
    # log_line("      1 ", progress_file, False)
    swaptime = time.time() - start
    start = time.time()

    product_list = product(*p_set)
    tourset = []
    for one_path in product_list:
        tour = []
        for segment in one_path:
            for node in segment:
                tour.append(node)
        tourset.append(tour)
    # log_line("      2 ", progress_file, False)
    tourset = [remove_consecutive_duplicates(path) for path in tourset]
    # log_line("      3 ", progress_file, False)
    # sanity_check_2(tourset)
    evaluatetime = time.time() - start
    tottime = swaptime + evaluatetime
    swaptime, evaluatetime = round(swaptime / tottime * 100), round(evaluatetime / tottime * 100)

    flipped_count_per, fetched_count_per, timeout_count_per = round(flipped_count / tcount * 100, 1), round(fetched_count / tcount * 100, 1), round(timeout_count / tcount * 100, 1)
    operator_stream += f"{flipped_count_per},{fetched_count_per},{timeout_count_per},{obj_selected},{flag},{swaptime},{evaluatetime},"
    # log_line("      4 ", progress_file, False)
    return tourset, weighted_paths_dict, operator_stream, terminal_pair_locked


def get_fixed_perm_terminal_priorities(terminal_indices: list, curr_tour: list, common_arguments: dict, obj_selected: str) -> dict:
    """
    This function gets the priorities for adjacent terminals while applying the Fixed Perm operator.

    Args:
        terminal_indices (list): List of terminal indices
        curr_tour (list): Current tour
        obj_selected (str): The selected objective
        common_arguments (dict): Common arguments for the local search

    Returns:
        dict: Dictionary of probabilities
    """
    combined_dict = common_arguments["combined_dict"]
    mean_energy_dual = common_arguments["mean_energy_dual"]
    mean_turn_dual = common_arguments["mean_turn_dual"]
    pr = {}
    for idx in range(len(terminal_indices) - 1):
        u = terminal_indices[idx]
        v = terminal_indices[idx + 1]
        if curr_tour[u] == curr_tour[v]:
            continue  # This happens rarely, but it was observed in one of the instances.
        sub_path_term = get_subpath(curr_tour, u, v, False)
        length_term = len(sub_path_term)
        e_consumption, t_consumption = get_path_length_using_energy_turns(sub_path_term, combined_dict)
        if obj_selected == 'energy':
            pr[u, curr_tour[u], v, curr_tour[v]] = max(e_consumption - length_term * mean_energy_dual, 0)
        else:
            pr[u, curr_tour[u], v, curr_tour[v]] = max(t_consumption - length_term * mean_turn_dual, 0)
    tour_consumption = sum(pr.values())
    if tour_consumption == 0:
        # This happens when the turns in curr_tour is zero.
        return pr
    else:
        pr = {terminal_tuple: consumption / tour_consumption for terminal_tuple, consumption in pr.items()}
        return pr


def get_random_skewed_tour(set_z: list, operator_stream: str, iteration_no: int, common_arguments: dict) -> tuple:
    """
    This function gets a random skewed tour in the FixedPerm operator

    Args:
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        operator_stream (str): Log string for the operator
        iteration_no (int): The current iteration number
        common_arguments (dict): Common arguments for the local search

    Returns:
        tour (list): Selected tour
        e (float): The energy costs of the path
        t (float): The turn costs of the path
        operator_stream (str): Log string for the operator
    """
    fp_std_dev_factor = common_arguments["ls_constants"]['fp_std_dev_factor']
    lower_energy_bound = common_arguments["mean_energy_dual"] - fp_std_dev_factor * common_arguments["stdev_energy_dual"]
    upper_energy_bound = common_arguments["mean_energy_dual"] + fp_std_dev_factor * common_arguments["stdev_energy_dual"]
    lower_turn_bound = common_arguments["mean_turn_dual"] - fp_std_dev_factor * common_arguments["stdev_turn_dual"]
    upper_turn_bound = common_arguments["mean_turn_dual"] + fp_std_dev_factor * common_arguments["stdev_turn_dual"]
    skewed_tours = []

    for path, e, t in set_z:
        edge_count = len(path) - 1
        if (lower_energy_bound * edge_count < e < upper_energy_bound * edge_count) and (lower_turn_bound * edge_count < t < upper_turn_bound * edge_count):
            continue
        skewed_tours.append((path, e, t))
    operator_stream += f"{len(skewed_tours)},{len(set_z)},"
    try:
        return *get_random(skewed_tours, []), operator_stream
    except ValueError:
        return [], None, None, operator_stream
