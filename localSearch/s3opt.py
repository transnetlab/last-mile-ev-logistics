"""
This file contains functions related to s3opt operator
"""
from collections import Counter
from shapely.geometry import Polygon

from helpers.functions import *
from localSearch.commonFunctions import get_time_left


def update_filter_var_s3opt(set_z: list, common_arguments: dict) -> tuple:
    """
    This function updates the candidate filter for the S3opt operator.

    Args:
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        common_arguments (dict): Dictionary containing common arguments

    Returns:
        filter_var (tuple): Tuple containing the energy and turn filter variables
    """
    ls_constants = common_arguments["ls_constants"]
    energy_dual = common_arguments["energy_dual"]
    direction_dual = common_arguments["direction_dual"]
    ls_start_time = common_arguments["ls_start_time"]

    if set_z:
        set_z_energies = [energy_dual[path[node]][path[node + 1]] for path, _, _ in set_z for node in range(len(path) - 1)]
        set_z_turns = [direction_dual[path[node]][path[node + 1]] for path, _, _ in set_z for node in range(len(path) - 1)]
        filter_var = sum(set_z_energies) / len(set_z_energies), sum(set_z_turns) / len(set_z_turns)
        # filter_reducer = ls_constants["filterReducerUp"] - ls_constants["filterReducerDown"] * (time.time() - ls_start_time) / ls_constants["allowed_ls_time"]
        # filter_var = (filter_var[0] * filter_reducer, filter_var[1] * filter_reducer)
        filter_var = (filter_var[0], filter_var[1])
    else:
        energies_graph = [y for x in energy_dual.values() for y in x.values()]
        turns_graph = [y for x in direction_dual.values() for y in x.values()]
        filter_var = sum(energies_graph) / len(energies_graph), sum(turns_graph) / len(turns_graph)
        # filter_var = (-np.inf, -np.inf)
    return filter_var


def post_process_s3_opt_step1(return_list: list, final_candidates: list, terminal_pairs: list, gain_trend: list, operator_stream: str) -> str:
    """
    This function processes the results obtained from applying Step 1 of the S3opt operator.

    Args:
        return_list (list): The list of terminal pairs
        final_candidates (list): The list of final candidates
        terminal_pairs (list): The list of terminal pairs
        gain_trend (list): The list of gains
        operator_stream (str): The log string

    Returns:
        operator_stream (str): The updated log string

    """
    per = round((len(terminal_pairs) - len(final_candidates)) / (len(terminal_pairs)) * 100)
    if return_list:
        operator_stream += f"{len(terminal_pairs)},{len(final_candidates)},{per},{len(gain_trend)},{gain_trend[0]},{gain_trend[-1]},"
    else:
        operator_stream += f"{len(terminal_pairs)},{len(final_candidates)},{per},Nan,Nan,Nan,"
    return operator_stream


def get_segment_to_be_removed(curr_tour: list, va: int, vb: int, combined_dict: dict) -> tuple:
    """
    This function gets the segment that is to be removed from a tour.

    Args:
        curr_tour (list): Current tour
        va (int): Node index
        vb (int): Node index
        combined_dict (dict): The dictionary containing the combined energy and direction values

    Returns:
        tuple: Tuple containing the segment, energy, and turns
    """
    va_vb_seg = get_subpath(curr_tour, va, vb, False)
    va_vb_seg_energy, va_vb_seg_turns = get_path_length_using_energy_turns(va_vb_seg, combined_dict)
    return va_vb_seg, va_vb_seg_energy, va_vb_seg_turns


def get_segment_to_be_added(curr_tour: list, va: int, vb: int, combined_dict: dict, weighted_paths_dict: dict) -> tuple:
    """
    This function gets the segment that is to be added to a tour.

    Args:
        curr_tour (list): Current tour
        va (int): Node index
        vb (int): Node index
        combined_dict (dict): The dictionary containing the combined energy and direction values
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals

    Returns:
        tuple: Tuple containing the segment, energy, and turns

    """
    va_name = curr_tour[va]
    vb_name = curr_tour[vb]
    if va_name == vb_name:
        return [], 0, 0
    ab_segment = list(get_random(list(weighted_paths_dict[va_name, vb_name]), []))
    ab_seg_energy, ab_seg_turns = get_path_length_using_energy_turns(ab_segment, combined_dict)
    return ab_segment, ab_seg_energy, ab_seg_turns


def update_weighted_paths_dict(weighted_paths_dict: dict, va_name: int, vb_name: int, new_paths: list) -> dict:
    """
    This function updates the weighted paths dictionary.

    Args:
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        va_name (str): Node name
        vb_name (str): Node name
        new_paths (list): List of paths

    Returns:
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
    """
    if new_paths:
        weighted_paths_dict[va_name, vb_name].update(map(tuple, new_paths))
    return weighted_paths_dict


def select_tour_using_kde(set_z: list, common_arguments: dict) -> int:
    """
    This function selects a node using the KDE density.

    Args:
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        common_arguments (dict): Dictionary containing common arguments

    Returns:
        selected_index (int): The selected index
    """
    ls_iteration_no = common_arguments["ls_iteration_no"]
    # log_line("  KD-1-", progress_file, False)
    consumption_values = [(e, t) for p, e, t in set_z]
    data_x, data_y = zip(*consumption_values)
    if len(data_x) <= 1:
        return 0

    # Combine the x and y values into a 2D array
    data = np.vstack([data_x, data_y])
    # log_line("  2-", progress_file, False)
    try:
        # Evaluate the probability density function at the data points
        density = gaussian_kde(data).evaluate(data)
        density = density / np.sum(density)

        pi_d = [(1 - val) for val in density]
        pi_sum = np.sum(pi_d)
        probabilities = [val / pi_sum for val in pi_d]
        # Randomly pick an index from the list using the computed probabilities
        rnd = get_random(data_x, probabilities)
        selected_index = data_x.index(rnd)
        if selected_index == 0:
            selected_index = 1
        if selected_index == len(data_x) - 1:
            selected_index = len(data_x) - 2
    except np.linalg.LinAlgError:
        # Singular matrix error is raised if set_z is of length 2
        selected_index = get_random(list(range(len(data_x) - 1)), [])
    # log_line("  4-", progress_file, False)
    return selected_index


def get_scaled_number(data_list: list) -> list:
    """
    Scales a list of data between 0 and 1

    Args:
        data_list (list): List of data

    Returns:
        result (list): List of scaled data
    """
    minimum = min(data_list)
    maximum = max(data_list)

    if maximum != minimum:
        result = [(x - minimum) / (maximum - minimum) for x in data_list]
        return result
    else:
        return data_list


def can_tour_be_added_to_pareto_set(list_of_tuples: list[tuple[float, float]], new_point: tuple[float, float]) -> np.ndarray:
    """
    This function checks if a new point can be added to a set of Pareto-optimal points.

    Args:
        list_of_tuples (list[tuple[float, float]]): List of tuples
        new_point (tuple[float, float]): New point to be added

    Returns:
        mask_values[-1] (int): 1 if the new point can be added to the Pareto set; 0 otherwise
    """
    new_list = list_of_tuples + [new_point]
    mask_values = get_pareto_set(new_list)
    return mask_values[-1]


def get_s3opt_gain(energy_val: float, turn_val: float, current_ef: list, progress_file: str) -> tuple:
    """
    This function gets the gain from applying the S3opt operator.

    Args:
        energy_val (float): Energy value
        turn_val (float): Turn value
        current_ef (list): Current EF
        progress_file (str): File to write progress to

    Returns:
        tuple: Tuple containing the gain and a boolean value
    """
    # energy_val, turn_val = new_energy, new_turns
    if can_tour_be_added_to_pareto_set(current_ef, (energy_val, turn_val)):
        y_old, x_old = zip(*current_ef)
        y_old, x_old = list(y_old), list(x_old)
        # Add the axis points to start and end
        x_old.insert(0, max(x_old))
        y_old.insert(0, 0)
        x_old.append(min(x_old))
        y_old.append(0)
        # sanity_check_5(x_old)
        area_old = Polygon(zip(get_scaled_number(x_old), get_scaled_number(y_old))).area

        # New area after adding point. First add the new point and then remove dominated points
        current_ef_copy = current_ef + [(energy_val, turn_val)]
        mask_values = get_pareto_set(current_ef_copy)
        if Counter(mask_values)[True] == 1:
            # Exception case. The new point dominates the whole EF. Add exception/logic if this happens below.
            return np.inf, True
            # raise ValueError(f"The new point dominates whole EF. This is a rare case. Check and if everything is okay, add logic to account for this case. current_ef= {current_ef}, new_point= {(energy_val, turn_val)}")
        # Sort the EF after adding the new point
        current_ef_copy = [path for path, flag in zip(current_ef_copy, mask_values) if flag]
        # current_ef_copy= current_ef
        y_new, x_new = zip(*current_ef_copy)
        sorted_lists = sorted(zip(y_new, x_new))
        y_new, x_new = zip(*sorted_lists)
        y_new, x_new = list(y_new), list(x_new)

        # Add the axis points to start and end
        x_new.insert(0, max(x_new))
        y_new.insert(0, 0)
        x_new.append(min(x_new))
        y_new.append(0)
        area_new = Polygon(zip(get_scaled_number(x_new), get_scaled_number(y_new))).area

        area_diff = area_new - area_old
        return abs(area_diff), False
    else:
        return -np.inf, False


def execute_s3_opt_step_2(curr_tour: list, cycle_subpath: list, weighted_paths_dict: dict, move_start_stamp: float, common_arguments: dict) -> tuple:
    """
    This function executes the second step of the S3opt operator.

    Args:
        curr_tour (list): Current tour
        cycle_subpath (list): The cycle subpath
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        move_start_stamp (float): Time the move started
        common_arguments (dict): Dictionary containing common arguments

    Returns:
        new_tours (list): List of new tours
        status (int): The number of iterations
    """
    if not cycle_subpath:
        return [], 1
    start = time.time()
    [v1, v2, v3, v4, v1_v4] = cycle_subpath

    new_tours = []

    # Break the cycles to get the tour
    v4_v1_segment = get_subpath(curr_tour, v4, v1, False)
    terminals_in_v4v1_seg = get_terminal_position(v4_v1_segment, common_arguments["terminal_set"])
    # Change the indices w.r.t curr_tour
    terminals_in_v4v1_seg = [(x + v4) if x + v4 < len(curr_tour) else (x + v4) % len(curr_tour) + 1 for x in terminals_in_v4v1_seg]
    # sanity_check_1(terminals_in_v4v1_seg)
    max_nb_runtime = common_arguments["ls_constants"]["max_nb_runtime"]
    for v5_idx in range(len(terminals_in_v4v1_seg) - 2):
        if not get_time_left(max_nb_runtime, move_start_stamp):
            break
        if v5_idx == 0: continue  # v5 cannot be equal to v4
        if v5_idx == len(terminals_in_v4v1_seg) - 1: continue  # # v5 cannot be equal to v1
        if v5_idx == len(terminals_in_v4v1_seg) - 2: continue  # # v6 cannot be equal to v1

        # Remove segment v5, v6
        v5 = terminals_in_v4v1_seg[v5_idx]
        v6 = terminals_in_v4v1_seg[v5_idx + 1]

        # Add segment v3, v6
        v3_v6, v3_v6_energy, v3_v6_turns = get_segment_to_be_added(curr_tour, v3, v6, common_arguments["combined_dict"], weighted_paths_dict)
        if not v3_v6: continue
        # Add segment v5, v2
        v5_v2, v5_v2_energy, v5_v2_turns = get_segment_to_be_added(curr_tour, v5, v2, common_arguments["combined_dict"], weighted_paths_dict)
        if not v5_v2: continue

        v4_v5 = get_subpath(curr_tour, v4, v5, False)
        v2_v3 = get_subpath(curr_tour, v2, v3, False)
        v6_v1 = get_subpath(curr_tour, v6, v1, False)
        v6_v1 = list(remove_consecutive_duplicates(v6_v1))  # The depot node occurs twice consecutively
        new_path = v1_v4 + v4_v5 + v5_v2 + v2_v3 + v3_v6 + v6_v1
        # sanity_check_14(v1_v4, v4_v5, v5_v2, v2_v3, v3_v6, v6_v1)
        new_path = remove_consecutive_duplicates(new_path)

        new_path = tuple(cyclize(new_path, new_path.index(common_arguments["depot"])))
        new_tours.append(new_path)
    return new_tours, 2


def apply_candidate_filter(terminal_pairs: list, curr_tour: list, combined_dict: dict, filter_var: tuple) -> list:
    """
    This function applies the candidate filter to terminal pairs.

    Args:
        terminal_pairs (list): List of terminal pairs
        curr_tour (list): Current tour
        combined_dict (dict): The dictionary containing the combined energy and direction values
        filter_var (tuple): Filter variable

    Returns:
        final_candidates (list): List of terminal pairs

    """

    final_candidates = []
    for (v1, v2, v3, v4) in terminal_pairs:
        energy_v1_v2, turns_v1_v2 = get_path_length_using_energy_turns(get_subpath(curr_tour, v1, v2, False), combined_dict)
        energy_v3_v4, turns_v3_v4 = get_path_length_using_energy_turns(get_subpath(curr_tour, v3, v4, False), combined_dict)
        if energy_v1_v2 < filter_var[0] * (v2 - v1) or turns_v1_v2 < filter_var[1] * (v2 - v1):
            continue
        if energy_v3_v4 < filter_var[0] * (v4 - v3) or turns_v3_v4 < filter_var[1] * (v4 - v3):
            continue
        final_candidates.append((v1, v2, v3, v4))
    final_candidates = custom_shuffle(final_candidates)
    return final_candidates


def execute_s3_opt_step_1(curr_tour: list, curr_tour_energy: float, curr_tour_turns: float, current_ef: list, filter_var: tuple, weighted_paths_dict: dict,
                          operator_stream: str, move_start_stamp: float, common_arguments: dict) -> tuple:
    """
    This function executes the first step of the S3opt operator.

    Args:
        curr_tour (list): Current tour
        curr_tour_energy (float): Current tour energy
        curr_tour_turns (float): Current tour turns
        current_ef (list): Current EF
        filter_var (tuple): Filter variable
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        operator_stream (str): Log string for the operator
        move_start_stamp (float): Time the move started
        common_arguments (dict): Dictionary containing common arguments

    Returns:
        return_list (list): List of terminal pairs
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        operator_stream (str): Log string for the operator
    """
    best_gain = -np.inf
    return_list = []

    terminal_indices = get_terminal_position(curr_tour, common_arguments["terminal_set"])
    terminal_pairs = get_s3_opt_terminal_pairs(terminal_indices)
    final_candidates = apply_candidate_filter(terminal_pairs, curr_tour, common_arguments["combined_dict"], filter_var)

    gain_trend = []
    # log_line(f"  /{len(final_candidates)}/ ", progress_file, False)
    max_nb_runtime = common_arguments["ls_constants"]["max_nb_runtime"]
    for (v1, v2, v3, v4) in final_candidates:
        if not get_time_left(max_nb_runtime, move_start_stamp):
            break
        # Remove segment v1_v2
        v1_v2_seg, v1_v2_seg_energy, v1_v2_seg_turns = get_segment_to_be_removed(curr_tour, v1, v2, common_arguments["combined_dict"])

        # Remove segment v3_v4
        v3_v4_seg, v3_v4_seg_energy, v3_v4_seg_turns = get_segment_to_be_removed(curr_tour, v3, v4, common_arguments["combined_dict"])

        # Add segment v1_v4
        q1_segment, q1_seg_energy, q1_seg_turns = get_segment_to_be_added(curr_tour, v1, v4, common_arguments["combined_dict"], weighted_paths_dict)
        if not q1_segment: continue

        new_energy = curr_tour_energy + q1_seg_energy - v1_v2_seg_energy
        new_turns = curr_tour_turns + q1_seg_turns - v1_v2_seg_turns

        curr_gain, forced = get_s3opt_gain(new_energy, new_turns, current_ef, common_arguments["progress_file"])
        if curr_gain > best_gain or forced:
            best_gain = curr_gain
            return_list = [v1, v2, v3, v4, q1_segment]
            gain_trend.append(best_gain)
        if forced: break
    operator_stream = post_process_s3_opt_step1(return_list, final_candidates, terminal_pairs, gain_trend, operator_stream)
    return return_list, weighted_paths_dict, operator_stream


def get_s3_opt_terminal_pairs(terminal_indices: list) -> list:
    """
    This function gets the terminal pairs for the S3opt operator.

    Args:
        terminal_indices (list): List of terminal indices

    Returns:
        list: List of terminal pairs
    """
    total_terminals = len(terminal_indices)
    pairs = []
    # -1 because we need to leave the last element for v4
    for v3_index in range(total_terminals - 1):
        v3 = terminal_indices[v3_index]
        v4 = terminal_indices[v3_index + 1]
        starting_v1_index = v3_index + 4  # Jumps three (v4,v5,v6) nodes ahead
        list_of_v1_indices = get_v1_indices(starting_v1_index, v3_index, total_terminals)[:-1]  # the last element has to be reserved for v2
        for v1_index in list_of_v1_indices:
            # v1_index == total_terminals - 1 means v1 is the last node w.r.t original tour (which is depot). Thus, the next node will also be depot which makes v1==v2.l
            if v1_index == total_terminals - 1: continue
            v1 = terminal_indices[v1_index]
            v2 = terminal_indices[(v1_index + 1) % total_terminals]
            # When v1 is the last element in the list, v2 is the first element in the terminal_indices
            pairs.append((v1, v2, v3, v4))
    return pairs


def get_v1_indices(starting_v1_index: int, v3_index: int, total_terminals: int) -> list:
    """
    Get the indices of v1

    Args:
        starting_v1_index (int): The starting index of v1
        v3_index (int): The index of v3
        total_terminals (int): The total number of terminals

    Returns:
        list: List of indices of v1
    """
    if total_terminals >= starting_v1_index:
        return list(range(starting_v1_index, total_terminals)) + list(range(0, v3_index))
    else:
        shift = starting_v1_index % total_terminals
        return list(range(shift, v3_index))


'''
def get_weighted_graph(L: networkx.classes.multidigraph.MultiDiGraph, alpha: float, combined_dict: dict) -> networkx.classes.multidigraph.MultiDiGraph:
    """
    Get the weighted graph
    """
    L_weighted = L.copy()
    for u, v, d in L_weighted.edges(data=True):
        e, t = combined_dict[u][v]
        d['length'] = alpha * e + (1 - alpha) * t
    return L_weighted


def set_alpha_s3opt(set_z: list, selected_index: int, seed: int) -> float:
    """
    Set the alpha value for the S3opt operator
    """
    if len(set_z) > 1:
        _, e1, t1 = set_z[selected_index - 1]
        _, e2, t2 = set_z[selected_index + 1]
        try:
            ratio = abs((e2 - e1) / (t1 - t2))
        except ZeroDivisionError:
            ratio = 0.5
        alpha = ratio / (ratio + 1)
    else:
        alpha = get_uniform_random_number(0, 1, seed + 3)
    return alpha

'''
