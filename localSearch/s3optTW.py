"""
This file contains functions related to s3optTW operator
"""
from helpers.functions import *
from localSearch.commonFunctions import get_time_left
from localSearch.s3opt import get_segment_to_be_removed, get_segment_to_be_added


def initialize_s3_opt_tw_step_1(curr_tour: list, common_arguments: dict) -> tuple:
    """
    This functions initializes the variables for Step 1 of the s3OptTW operator.

    Args:
        curr_tour (list): Current tour
        common_arguments (dict): Common arguments for the local search

    Returns:
        best_gain (float): Best gain
        return_list (list): List of tuples
        max_nb_runtime (int): Maximum number of runtime
        all_terminal_indices (list): List of terminal indices
        terminal_pairs (list): List of terminal pairs
        gain_trend (list): List of gains
    """
    best_gain = -np.inf
    return_list = []
    max_nb_runtime = common_arguments["ls_constants"]['max_nb_runtime']
    all_terminal_indices = get_terminal_position(curr_tour, common_arguments["terminal_set"])
    terminal_pairs = []
    gain_trend = []
    return best_gain, return_list, max_nb_runtime, all_terminal_indices, terminal_pairs, gain_trend


def get_node_v2(curr_tour: list, common_arguments: dict) -> list:
    """
    Get the node v2

    Args:
        curr_tour (list): Current tour
        common_arguments (dict): Common arguments for the local search

    Returns:
        violated_terminal_indices (list): List of nodes
    """
    _, _, unsatisfied_terminals_dict = get_penalty(curr_tour, common_arguments["time_cost_dict"], common_arguments["time_window_dict"], common_arguments["depot"])
    unsatisfied_terminals = unsatisfied_terminals_dict.keys()
    violated_terminal_indices = get_terminal_position(curr_tour, unsatisfied_terminals)
    return violated_terminal_indices


def is_tw_satisfied_for_v4(curr_tour: list, v1: int, q1_segment: list, common_arguments: dict) -> bool:
    """
    Check if the time window is satisfied for v4

    Args:
        curr_tour (list): Current tour
        v1 (int): Node v1
        q1_segment (list): Segment q1
        common_arguments (dict): Common arguments for the local search

    Returns:
        tw_satisfied_till_v4 (bool): True if the time window is satisfied, False otherwise
    """
    path_till_v4 = get_subpath(curr_tour, 0, v1, False) + q1_segment
    path_till_v4 = remove_consecutive_duplicates(path_till_v4)
    _, tw_satisfied_till_v4, _, = get_penalty(path_till_v4, common_arguments["time_cost_dict"], common_arguments["time_window_dict"], common_arguments["depot"])
    return tw_satisfied_till_v4


def get_candidate_s3opt_tw(v3_v4_slots: list, curr_tour: list, v1: int, v2: int) -> list:
    """
    This function gets the candidate nodes for the s3OptTW operator.

    Args:
        v3_v4_slots (list): List of nodes
        curr_tour (list): Current tour
        v1 (int): Node v1
        v2 (int): Node v2

    Returns:
        final_candidates (list): List of nodes
    """
    final_candidates = []
    for idx in range(len(v3_v4_slots) - 1):
        v3 = v3_v4_slots[idx]
        v4 = v3_v4_slots[idx + 1]
        v3_name = curr_tour[v3]
        v4_name = curr_tour[v4]
        if v3_name == v4_name:
            # v3 and v4 can be depot since curr_tour starts and ends at the depot node
            continue
        final_candidates.append((v1, v2, v3, v4))
    return final_candidates


def execute_s3_opt_tw_step_1(curr_tour: list, curr_tour_energy: float, curr_tour_turns: float, weighted_paths_dict: dict, operator_stream: str, move_start_stamp: float,
                             common_arguments: dict) -> tuple:
    """
    This function executes the first step of the S3OptTW operator.

    Args:
        curr_tour (list): Current tour
        curr_tour_energy (float): Current tour energy
        curr_tour_turns (float): Current tour turns
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        operator_stream (str): Log string for the operator
        move_start_stamp (float): Time the move started
        common_arguments (dict): Common arguments for the local search

    Returns:
        return_list (list): List of tuples
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        operator_stream (str): Log string for the operator
    """
    best_gain, return_list, max_nb_runtime, all_terminal_indices, terminal_pairs, gain_trend = initialize_s3_opt_tw_step_1(curr_tour, common_arguments)
    v2_list = get_node_v2(curr_tour, common_arguments)
    combined_dict = common_arguments["combined_dict"]
    for v2 in v2_list:
        v1 = all_terminal_indices[all_terminal_indices.index(v2) - 1]
        # There should be atleast two nodes and last index is not counted
        last_v4_index = all_terminal_indices[all_terminal_indices.index(v1) - 2 - 1]
        # Starting from v2 (ignore v2) till end. Starting from depot, till last_v4_index plus 1 (last index is not counted)
        v3_v4_slots = all_terminal_indices[all_terminal_indices.index(v2) + 1:] + all_terminal_indices[: all_terminal_indices.index(last_v4_index) + 1]
        terminal_pairs.extend(get_candidate_s3opt_tw(v3_v4_slots, curr_tour, v1, v2))
    final_candidates = list(set(terminal_pairs))
    # final_candidates = custom_shuffle(terminal_pairs)

    # Maximize gain over the terminals
    for (v1, v2, v3, v4) in final_candidates:
        if not get_time_left(max_nb_runtime, move_start_stamp):
            break
        # Remove segment v1_v2
        v1_v2_seg, v1_v2_seg_energy, v1_v2_seg_turns = get_segment_to_be_removed(curr_tour, v1, v2, combined_dict)
        # Remove segment v3_v4
        v3_v4_seg, v3_v4_seg_energy, v3_v4_seg_turns = get_segment_to_be_removed(curr_tour, v3, v4, combined_dict)
        # Add segment v1_v4
        q1_segment, q1_seg_energy, q1_seg_turns = get_segment_to_be_added(curr_tour, v1, v4, combined_dict, weighted_paths_dict)
        if not q1_segment: continue
        tw_satisfied_till_v4 = is_tw_satisfied_for_v4(curr_tour, v1, q1_segment, common_arguments)
        if not tw_satisfied_till_v4: continue
        new_energy = curr_tour_energy + q1_seg_energy - v1_v2_seg_energy - v3_v4_seg_energy
        new_turns = curr_tour_turns + q1_seg_turns - v1_v2_seg_turns - v3_v4_seg_turns

        return_list.append((v1, v2, v3, v4, q1_segment, new_energy, new_turns))
    operator_stream = post_process_s3_opt_tw_step1(return_list, final_candidates, terminal_pairs, gain_trend, operator_stream)
    return return_list, weighted_paths_dict, operator_stream


def post_process_s3_opt_tw_step1(return_list: list, final_candidates: list, terminal_pairs: list, gain_trend: list, operator_stream: str) -> str:
    """
    This function processes the results obtained from applying Step 1 of the S3OptTW operator.

    Args:
        return_list (list): List of tuples
        final_candidates (list): List of tuples
        terminal_pairs (list): List of tuples
        gain_trend (list): List of gains
        operator_stream (str): Log string for the operator

    Returns:
        operator_stream (str): Log string for the operator
    """
    per = round((len(terminal_pairs) - len(final_candidates)) / (len(terminal_pairs)) * 100)
    if return_list:
        operator_stream += f"{len(return_list)},{len(terminal_pairs)},{len(final_candidates)},{per},{len(gain_trend)},{0},{0},"
    else:
        operator_stream += f"0,{len(terminal_pairs)},{len(final_candidates)},{per},Nan,Nan,Nan,"
    return operator_stream


def initialize_s3_opt_tw_step_2(ls_constants: dict) -> tuple:
    """
    This functions initializes the variables for Step 2 of the s3OptTW operator.

    Args:
        ls_constants (dict): Constants for the local search

    Returns:
        new_tours (list): List of tours
        max_nb_runtime (int): Maximum number of runtime
        start (float): Start time
    """
    new_tours = []
    max_nb_runtime = ls_constants["max_nb_runtime"]
    start = time.time()
    return new_tours, max_nb_runtime, start


def execute_s3_opt_tw_step_2(curr_tour: list, listof_cycle_subpath: list, weighted_paths_dict: dict, move_start_stamp: float, operator_stream: str, common_arguments: dict) -> tuple:
    """
    This function executes the Step 2 of the S3OptTW operator.

    Args:
        curr_tour (list): Current tour
        listof_cycle_subpath (list): List of cycles from the first step
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        move_start_stamp (float): Time the move started
        operator_stream (str): Log string for the operator
        common_arguments (dict): Common arguments for the local search

    Returns:
        new_tours (list): List of tours
        status (int): Status code
        operator_stream (str): Log string for the operator

    """
    terminal_set = common_arguments["terminal_set"]
    combined_dict = common_arguments["combined_dict"]
    depot = common_arguments["depot"]

    if not listof_cycle_subpath:
        operator_stream += "0,0,Nan,"
        return [], 1, operator_stream

    new_tours, max_nb_runtime, _ = initialize_s3_opt_tw_step_2(common_arguments["ls_constants"])
    # log_line(f"  /{len(listof_cycle_subpath)}/ ", progress_file, False)
    for [v1, v2, v3, v4, v1_v4, energy_val, turns_val] in reversed(listof_cycle_subpath):
        if not get_time_left(max_nb_runtime, move_start_stamp):
            break

        # Break the cycles to get the tour
        # Change the indices w.r.t curr_tour
        terminals_in_v4v1_seg = [(x + v4) if x + v4 < len(curr_tour) else (x + v4) % len(curr_tour) + 1 for x in get_terminal_position(get_subpath(curr_tour, v4, v1, False), terminal_set)]
        # sanity_check_1(terminals_in_v4v1_seg)

        for v5_idx in range(len(terminals_in_v4v1_seg) - 2):
            # v5 cannot be equal to v4 or # v5 cannot be equal to v1 or v6 cannot be equal to v1
            if v5_idx == 0 or v5_idx == len(terminals_in_v4v1_seg) - 1 or v5_idx == len(terminals_in_v4v1_seg) - 2: continue

            # Remove segment v5, v6
            v5, v6 = terminals_in_v4v1_seg[v5_idx], terminals_in_v4v1_seg[v5_idx + 1]

            # Add segment v3, v6
            v3_v6, v3_v6_energy, v3_v6_turns = get_segment_to_be_added(curr_tour, v3, v6, combined_dict, weighted_paths_dict)
            if not v3_v6: continue
            # Add segment v5, v2
            v5_v2, v5_v2_energy, v5_v2_turns = get_segment_to_be_added(curr_tour, v5, v2, combined_dict, weighted_paths_dict)
            if not v5_v2: continue

            v4_v5 = get_subpath(curr_tour, v4, v5, False)
            v2_v3 = get_subpath(curr_tour, v2, v3, False)
            v6_v1 = get_subpath(curr_tour, v6, v1, False)
            v6_v1 = list(remove_consecutive_duplicates(v6_v1))  # The depot node occurs twice consecutively
            new_path = remove_consecutive_duplicates(v1_v4 + v4_v5 + v5_v2 + v2_v3 + v3_v6 + v6_v1)

            tour_energy, tour_turns = energy_val + v3_v6_energy + v5_v2_energy, turns_val + v3_v6_turns + v5_v2_turns
            new_tours.append((tuple(cyclize(new_path, new_path.index(depot))), tour_energy, tour_turns))
    if new_tours:
        init_len = len(new_tours)
        # log_line(f"  /{init_len}/ ", progress_file, False)
        start = time.time()
        mask_values = get_pareto_set([(e, t) for _, e, t in new_tours])
        new_tours = [path for (path, e, t), flag in zip(new_tours, mask_values) if flag]
        final_len = len(new_tours)
        operator_stream += f"{init_len},{final_len},{round(time.time() - start)},"
    else:
        operator_stream += "0,0,Nan,"
    return new_tours, 2, operator_stream


"""
def update_filter_var_s3optTW(set_z: list, energy_dual: dict, direction_dual: dict, ls_constants: dict, ls_start_time: float) -> tuple:
    allowed_ls_time = ls_constants["allowed_ls_time"]
    if set_z:
        set_z_energies = [energy_dual[path[node]][path[node + 1]] for path, _, _ in set_z for node in range(len(path) - 1)]
        set_z_turns = [direction_dual[path[node]][path[node + 1]] for path, _, _ in set_z for node in range(len(path) - 1)]
        filter_var = sum(set_z_energies) / len(set_z_energies), sum(set_z_turns) / len(set_z_turns)
        # filter_reducer = ls_constants["filterReducerUp"] - ls_constants["filterReducerDown"] * (time.time() - ls_start_time) / allowed_ls_time
        # filter_var = (filter_var[0] * filter_reducer, filter_var[1] * filter_reducer)
        filter_var = (filter_var[0], filter_var[1])
    else:
        energies_graph = [y for x in energy_dual.values() for y in x.values()]
        turns_graph = [y for x in direction_dual.values() for y in x.values()]
        filter_var = sum(energies_graph) / len(energies_graph), sum(turns_graph) / len(turns_graph)
        # filter_var = (-np.inf, -np.inf)
    return filter_var
"""
