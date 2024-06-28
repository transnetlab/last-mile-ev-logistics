"""
This file contains functions related to GapRepair operator
"""
from itertools import product

from helpers.functions import *
from localSearch.commonFunctions import get_quickest_path, get_time_left


def can_terminal_be_added_between_terminals(unode: int, vnode: int, candidate_terminal: int, arrival_time_at_u: float, time_window_dict: dict, time_cost_dict: dict, quickest_paths_dict: dict,
                                            L: networkx.classes.multidigraph.MultiDiGraph) -> tuple:
    """
    This function checks whether a terminal can be added between two terminals in a path.

    Args:
        unode (int): U node
        vnode (int): V node
        candidate_terminal (int): Candidate terminal to be added
        arrival_time_at_u (float): Arrival time at u
        time_window_dict (dict): The dictionary containing the time windows for each terminal
        time_cost_dict (dict): Dictionary of travel times
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
        L (networkx.classes.multidigraph.MultiDiGraph): Dual graph

    Returns:
        tuple: Tuple containing a flag and the quickest paths dictionary
    """
    lower_tw_at_v, upper_tw_at_v = time_window_dict[vnode]
    _, upper_tw_at_candidate = time_window_dict[candidate_terminal]
    if candidate_terminal == vnode or candidate_terminal == unode:
        return False, quickest_paths_dict
    # if tw of the candidate itself doesn't allow
    if upper_tw_at_candidate <= arrival_time_at_u:
        return False, quickest_paths_dict
    # if time window allows but there's no shortest path from u to candidate violates tw at u, i.e., cannot reach u in time
    path, quickest_paths_dict = get_quickest_path(L, unode, candidate_terminal, time_cost_dict, quickest_paths_dict)
    u_to_candidate_time = get_path_length(path, time_cost_dict)
    if upper_tw_at_candidate < arrival_time_at_u + u_to_candidate_time:
        return False, quickest_paths_dict
    # not possible to reach final terminal in time window
    path, quickest_paths_dict = get_quickest_path(L, candidate_terminal, vnode, time_cost_dict, quickest_paths_dict)
    candidate_to_v_time = get_path_length(path, time_cost_dict)
    if upper_tw_at_v < arrival_time_at_u + u_to_candidate_time + candidate_to_v_time:
        return False, quickest_paths_dict
    return True, quickest_paths_dict


def get_segments_to_be_removed(unsatisfied_terminals_idx: list, terminal_indices: list) -> list:
    """
    This function gives the segments to be removed

    Args:
        unsatisfied_terminals_idx (list): List of unsatisfied terminal indices
        terminal_indices (list): List of terminal indices

    Returns:
        list: List of segments to be removed

    """
    pointer_i, pointer_j = 0, 0
    segments_tobe_removed = []
    for node_idx in terminal_indices:
        if node_idx not in unsatisfied_terminals_idx:
            if pointer_j == pointer_i:
                pointer_j = node_idx
            else:
                ch0 = pointer_i
                ch1 = node_idx
                segments_tobe_removed.append((ch0, ch1))
                pointer_j = node_idx
            pointer_i = node_idx
        if node_idx in unsatisfied_terminals_idx:
            pointer_j = node_idx
    return segments_tobe_removed


def get_cycle_q(curr_tour: list, segments_tobe_removed: list, timecost: dict, L: networkx.classes.multidigraph.MultiDiGraph, quickest_paths_dict: dict) -> tuple:
    """
    Get cycle q

    Args:
        curr_tour (list): Current tour
        segments_tobe_removed (list): List of segments to be removed
        timecost (dict): Dictionary of time costs
        L (networkx.classes.multidigraph.MultiDiGraph): Dual graph
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals

    Returns:
        cycle_q (list): Cycle q
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
    """
    cycle_q = []
    for idx in range(len(segments_tobe_removed)):
        if idx == 0:
            cycle_q += get_subpath(curr_tour, 0, segments_tobe_removed[idx][0], False)
        path, quickest_paths_dict = get_quickest_path(L, curr_tour[segments_tobe_removed[idx][0]], curr_tour[segments_tobe_removed[idx][1]], timecost, quickest_paths_dict)
        cycle_q += path
        if idx == len(segments_tobe_removed) - 1:
            cycle_q += get_subpath(curr_tour, segments_tobe_removed[idx][1], len(curr_tour) - 1, False)
        else:
            cycle_q += get_subpath(curr_tour, segments_tobe_removed[idx][1], segments_tobe_removed[idx + 1][0], False)
    cycle_q = remove_consecutive_duplicates(tuple(cycle_q))
    return cycle_q, quickest_paths_dict


def get_arrival_time_dict(cycle_q: list, time_cost_dict: dict, terminals_in_cycleq: list, time_window_dict: dict) -> dict:
    """
    Get arrival time dictionary

    Args:
        cycle_q (list): Cycle q
        time_cost_dict (dict): Dictionary of travel times
        terminals_in_cycleq (list): List of terminals in cycle q
        time_window_dict (dict): The dictionary containing the time windows for each terminal

    Returns:
        arrival_time_dict (dict): Arrival time dictionary
    """
    arrival_time_sofar = 0
    tour_waiting = 0
    terminal_remaining = set(terminals_in_cycleq) - {cycle_q[0]}
    arrival_time_dict = {0: 0}
    for node_idx, tail_node in enumerate(cycle_q[:-1]):
        head_node = cycle_q[node_idx + 1]
        travel_time = time_cost_dict[tail_node][head_node]
        arrival_time_sofar += travel_time
        arrival_time_dict[node_idx + 1] = arrival_time_sofar
        if head_node in terminal_remaining:
            terminal_satisfied, waiting_penalty = tw_satisfy(arrival_time_sofar, time_window_dict[head_node])
            if terminal_satisfied:
                terminal_remaining.remove(head_node)
                arrival_time_sofar += waiting_penalty
                tour_waiting += waiting_penalty
    return arrival_time_dict


def gap_repair_operator(curr_tour: list, operator_stream: str, quickest_paths_dict: dict, weighted_paths_dict: dict, common_arguments: dict) -> tuple:
    """
    This function runs a single iteration of the Gap Repair operator in which we repair the tour by removing segments and adding terminals.

    Args:
        curr_tour (list): Current tour
        operator_stream (str): Log string for the operator
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        common_arguments (dict): Common arguments for the local search

    Returns:
        newtours (list): List of new tours
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
        operator_stream (str): Log string for the operator
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
    """
    terminal_set = common_arguments["terminal_set"]
    ls_constants = common_arguments["ls_constants"]
    time_cost_dict = common_arguments["time_cost_dict"]
    time_window_dict = common_arguments["time_window_dict"]
    depot = common_arguments["depot"]
    max_nb_runtime = common_arguments["ls_constants"]["max_nb_runtime"]
    L = common_arguments["L"]
    ls_iteration_no = common_arguments["ls_iteration_no"]

    destroy_time_start = time.time()

    terminal_indices = get_terminal_position(curr_tour, terminal_set)
    _penalty, is_feasible, unsatisfied_terminals_dict = get_penalty(curr_tour, time_cost_dict, time_window_dict, depot)
    unsatisfied_terminals = tuple(unsatisfied_terminals_dict.keys())
    if len(unsatisfied_terminals) == len(terminal_indices):
        operator_stream += f"{-1},{-1},{-1},{-1},{-1},{0},"
        return [curr_tour], quickest_paths_dict, operator_stream, weighted_paths_dict
    unsatisfied_terminals_idx = [idx for idx, node in enumerate(curr_tour) if node in unsatisfied_terminals]

    segments_tobe_removed = get_segments_to_be_removed(unsatisfied_terminals_idx, terminal_indices)

    cycle_q, quickest_paths_dict = get_cycle_q(curr_tour, segments_tobe_removed, time_cost_dict, L, quickest_paths_dict)

    # If cycle q is BSTSPTW feasible then return q, it is already tw feasible
    missing_terminals = terminal_set - set(cycle_q)
    _penalty, is_feasible, _ = get_penalty(cycle_q, time_cost_dict, time_window_dict, depot)
    if len(missing_terminals) == 0 and is_feasible:
        operator_stream += f"{-1},{-1},{-1},{-1},{-1},{1},"
        return [cycle_q], quickest_paths_dict, operator_stream, weighted_paths_dict
    # log_line("  1 ", progress_file, False)
    # All terminals in cycle_q satisfy time windows
    try:
        terminals_in_cycleq_idx, terminals_in_cycleq = zip(*[(idx, node) for idx, node in enumerate(cycle_q) if node in terminal_set])
    except:
        # Sanity check
        print(unsatisfied_terminals_idx)
        print(terminal_set)
        print(curr_tour)
        print(cycle_q)
        print(terminal_set)
        print(segments_tobe_removed)
        raise ValueError("Error in getting terminals in cycle q")

    terminals_in_cycleq = list(set(terminals_in_cycleq))
    arrival_time_dict = get_arrival_time_dict(cycle_q, time_cost_dict, terminals_in_cycleq, time_window_dict)

    if len(terminals_in_cycleq_idx) < 2:
        operator_stream += f"{-1},{-1},{-1},{-1},{-1},{1.1},"
        return [], quickest_paths_dict, operator_stream, weighted_paths_dict

    # Check the tw unsatisfied terminals that can be added
    width_terms = {}
    for unode_idx, vnode_idx in zip(terminals_in_cycleq_idx, terminals_in_cycleq_idx[1:]):
        unode = cycle_q[unode_idx]
        vnode = cycle_q[vnode_idx]
        width_terms[(unode, vnode)] = []
        arrival_time_at_u = arrival_time_dict[unode_idx]
        for candidate_terminal_idx in unsatisfied_terminals_idx:  # Check which terminals can be added
            candidate_terminal = curr_tour[candidate_terminal_idx]
            flag, quickest_paths_dict = can_terminal_be_added_between_terminals(unode, vnode, candidate_terminal, arrival_time_at_u, time_window_dict, time_cost_dict, quickest_paths_dict, L)
            if flag:
                width_terms[(unode, vnode)].append(candidate_terminal_idx)  # save which terminals could be added

    # Get terminals to be deleted. Break ties randomly
    max_length = max(len(values) for values in width_terms.values())
    if max_length == 0:
        operator_stream += f"{-1},{-1},{-1},{-1},{-1},{0},"
        return [], quickest_paths_dict, operator_stream, weighted_paths_dict
    terminal_pairs_with_max_width = [item for item, term_list in width_terms.items() if len(term_list) == max_length]
    u_star, v_star = get_random(terminal_pairs_with_max_width, [])
    remove_terminals_idxs = width_terms[(u_star, v_star)]
    # n = get_random(list(range(1, len(remove_terminals_idxs) + 1)), [])
    # remove_terminals_idxs = get_random_sample(remove_terminals_idxs, n)

    # log_line("  2 ", progress_file, False)

    # Delete the terminals in remove_terminals from curr_tour
    path_to_repair = curr_tour[:]
    terminal_after = path_to_repair[-1]

    for terminal_idx_idx in reversed(range(len(terminal_indices))):
        # Iterate in reverse order
        terminal_idx = terminal_indices[terminal_idx_idx]
        ptr_before_len = len(path_to_repair)
        if terminal_idx in remove_terminals_idxs:
            # If the terminal is to be removed, then find the path between the terminal before and after the terminal to be removed
            terminal_before_idx = terminal_indices[terminal_idx_idx - 1]
            terminal_before = path_to_repair[terminal_before_idx]
            path, quickest_paths_dict = get_quickest_path(L, terminal_before, terminal_after, time_cost_dict, quickest_paths_dict)
            if terminal_after_idx < len(path_to_repair):
                # If tour is of length 10, and terminals are 0, 7, 10 indexes, and 7th node is to be removed, then terminal_after_idx = 10. Now if the shorest path from 7th node to
                # 10th node is of length 2, then path_to_repair = [0:7] + len(2). Thus 9. len(path_to_repair) - 1 = 8 and terminal_after_idx = 10. So we need to add the remaining path.
                path_to_repair = get_subpath(path_to_repair, 0, terminal_before_idx, False) + path + get_subpath(
                    path_to_repair, terminal_after_idx, len(path_to_repair) - 1, False)
            else:
                path_to_repair = get_subpath(path_to_repair, 0, terminal_before_idx, False) + path
        else:
            terminal_after_idx = terminal_idx
            terminal_after = path_to_repair[terminal_idx]
        path_to_repair = list(remove_consecutive_duplicates(path_to_repair))
        ptr_after_len = len(path_to_repair)
        terminal_after_idx = terminal_after_idx + (ptr_after_len - ptr_before_len)
        terminal_after = path_to_repair[terminal_after_idx]

    destry_time = time.time() - destroy_time_start
    # log_line("  3 ", progress_file, False)
    ###############################################################################################################################################
    # Repair step
    repair_time_start = time.time()
    path_to_repair = remove_consecutive_duplicates(path_to_repair)  # Remove repeated nodes
    path_to_repair = list(path_to_repair)
    if len(path_to_repair) == 1:
        path_to_repair.append(path_to_repair[0])
    if depot != path_to_repair[-1]:
        raise ValueError("depot not in end of path_to_repair")

    terminals_tobe_added = list(terminal_set - set(path_to_repair))
    if not terminals_tobe_added:
        operator_stream += f"{max_length},{-1},{-1},{-1},{-1},{2},"
        return [path_to_repair], quickest_paths_dict, operator_stream, weighted_paths_dict

    quickest_tours, weighted_tours, prefetched_tours = [], [], []
    u_star_list = [idx for idx, node in enumerate(path_to_repair) if node == u_star]
    path_to_repair_terminalidx = get_terminal_position(path_to_repair, terminal_set)
    start = time.time()
    # log_line("  4 ", progress_file, False)
    # log_line(f"  {len(u_star_list)} ", progress_file, False)
    for u_star_idx in u_star_list:
        # log_line(f"  {count}-", progress_file, False)
        if not get_time_left(max_nb_runtime, start):
            break
        if u_star_idx == len(path_to_repair) - 1: continue
        try:
            v_star_idx = path_to_repair_terminalidx[path_to_repair_terminalidx.index(u_star_idx) + 1]
            v_star = path_to_repair[v_star_idx]
        except IndexError:
            continue

        terminals_tobe_added_copy = terminals_tobe_added[:]
        best_sequence_path, best_sequence = [], []
        source = u_star
        while terminals_tobe_added_copy:
            otm = []
            for terminal in terminals_tobe_added_copy:
                path, quickest_paths_dict = get_quickest_path(L, source, terminal, time_cost_dict, quickest_paths_dict)
                otm.append((terminal, path))
            otm = [(node, path, get_path_length(path, time_cost_dict)) for node, path in otm]
            source, path, length = sorted(otm, key=lambda x: x[2])[0]
            best_sequence_path += path
            terminals_tobe_added_copy.remove(source)
            best_sequence.append(source)
        path, quickest_paths_dict = get_quickest_path(L, best_sequence_path[-1], v_star, time_cost_dict, quickest_paths_dict)
        best_sequence_path += path
        quickest_tour = get_subpath(path_to_repair, 0, u_star_idx, False) + best_sequence_path + get_subpath(path_to_repair, v_star_idx, len(path_to_repair) - 1, False)
        quickest_tour = remove_consecutive_duplicates(tuple(quickest_tour))
        quickest_tours.append(quickest_tour)

        # terminals_tobe_added_copy = terminals_tobe_added[:]
        # best_sequence_path, best_sequence = [], []
        # source = u_star
        # while terminals_tobe_added_copy:
        #     otm = []
        #     for terminal in terminals_tobe_added_copy:
        #         path = nx.astar_path(L, source, terminal, weight=weight_weighted)
        #         otm.append((terminal, path))
        #         weighted_paths_dict = update_weighted_paths_dict(weighted_paths_dict, source, terminal, [path])
        #     otm = [(node, path, sum([weight_weighted(path[i], path[i + 1], 0) for i in range(len(path) - 1)])) for node, path in otm]
        #     source, path, length = sorted(otm, key=lambda x: x[2])[0]
        #     best_sequence_path += path
        #     terminals_tobe_added_copy.remove(source)
        #     best_sequence.append(source)
        # path, quickest_paths_dict = get_quickest_path(L, best_sequence_path[-1], v_star, time_cost_dict, quickest_paths_dict)
        # best_sequence_path += path
        # weighted_tour = get_subpath(path_to_repair, 0, u_star_idx, False) + best_sequence_path + get_subpath(path_to_repair, v_star_idx, len(path_to_repair) - 1, False)
        # weighted_tour = remove_consecutive_duplicates(weighted_tour)
        # weighted_tours.append(weighted_tour)

        if 2 <= len(best_sequence) < ls_constants["gr_terminal_count"]:
            ustarr_to_first = []
            if u_star != best_sequence[0]:
                ustarr_to_first = get_random_sample(list(weighted_paths_dict[(u_star, best_sequence[0])]), ls_constants["gr_flip_count"])
            subtours = [get_random_sample(list(weighted_paths_dict[(best_sequence[ter_idx], best_sequence[ter_idx + 1])]), ls_constants["gr_flip_count"]) for ter_idx in
                        range(len(best_sequence) - 1)]
            last_to_vstarr = []
            if v_star != best_sequence[-1]:
                last_to_vstarr = get_random_sample(list(weighted_paths_dict[(best_sequence[-1], v_star)]), ls_constants["gr_flip_count"])
            subtours = [ustarr_to_first] + subtours + [last_to_vstarr]
            product_list = product(*subtours)
            subtours = []
            for one_path in product_list:
                tour = []
                for segment in one_path:
                    for node in segment:
                        tour.append(node)
                subtours.append(tour)
            for subtour in subtours:
                tour = get_subpath(list(path_to_repair), 0, u_star_idx, False) + subtour + get_subpath(list(path_to_repair), v_star_idx, len(path_to_repair) - 1, False)
                tour = remove_consecutive_duplicates(tuple(tour))
                prefetched_tours.append(tour)

    repair_time = time.time() - repair_time_start

    newtours = quickest_tours + weighted_tours + prefetched_tours

    # log_line("  4.1 ", progress_file, False)
    # sanity_check_18(terminal_set, newtours, depot)
    # sanity_check_19(energy_dual, quickest_tours, direction_dual)
    # log_line("  4.2 ", progress_file, False)
    # sanity_check_19(energy_dual, weighted_tours, direction_dual)
    # log_line(f"  4.3({len(prefetched_tours)}) ", progress_file, False)
    # for tour in prefetched_tours:
    #     get_path_length(tour, energy_dual)
    #     get_path_length(tour, direction_dual)

    # sanity_check_19(energy_dual, prefetched_tours, direction_dual)
    # log_line("  4.4 ", progress_file, False)

    destry_time = round((destry_time / (destry_time + repair_time)) * 100)
    repair_time = round((repair_time / (destry_time + repair_time)) * 100)
    operator_stream += f"{max_length},{destry_time},{repair_time},{True},{True},{3},"
    # log_line(f"  start({len(prefetched_tours)}) ", progress_file, False)
    # log_line(f"  {prefetched_tours} ", progress_file, False)
    # log_line(f"  end ", progress_file, False)
    # log_line("  5 ", progress_file, False)
    return newtours, quickest_paths_dict, operator_stream, weighted_paths_dict
