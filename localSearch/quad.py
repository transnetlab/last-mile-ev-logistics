"""
This file contains functions related to Quad operator
"""
from helpers.functions import *
from helpers.loggers import log_line
from localSearch.s3opt import update_weighted_paths_dict
from helpers.shortestPath import call_with_timeout, biobj_label_correcting


def quad_operator(curr_tour: list, weighted_paths_dict: dict, operator_stream: str, terminal_pair_locked: dict, iteration_no: int, common_arguments: dict) -> tuple:
    """
    This function executes a single iteration of the quad operator.

    Args:
        curr_tour (list): Current tour
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        operator_stream (str): Log string for the operator
        terminal_pair_locked (dict): Dictionary of terminal pairs that are locked
        iteration_no (int): The current iteration number
        common_arguments (dict): Common arguments for the local search

    Returns:
        quad_paths (list): List of tours
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        operator_stream (str): Log string for the operator
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals

    """
    ls_sp_timeout_limit = common_arguments["ls_constants"]['ls_sp_timeout_limit']
    terminal_indices = get_terminal_position(curr_tour, common_arguments["terminal_set"])
    combined_dict = common_arguments["combined_dict"]

    terminals_used = set()
    terminal_idx_selected = []

    # Randomly select four pairs of unique adjacent terminals
    term_pairs = list(zip(terminal_indices, terminal_indices[1:]))
    term_pairs.append((terminal_indices[-1], terminal_indices[0]))
    term_pairs = custom_shuffle(term_pairs)

    for pair in term_pairs:
        if len(terminal_idx_selected) == 4:
            break
        if len(set([curr_tour[x] for x in pair]) - terminals_used) == 2:
            terminal_idx_selected.append(pair)
            terminals_used.add(curr_tour[pair[0]])
            terminals_used.add(curr_tour[pair[1]])
    selected_terminals = [elem for pair in terminal_idx_selected for elem in pair] # Flatten the list
    if len(selected_terminals) != 8:
        log_line("  No unique adjacent terminals ", common_arguments["progress_file"], False)
        operator_stream += f'{None},{None},{None},{None},'
        return [], weighted_paths_dict, operator_stream, weighted_paths_dict
    # sanity_check_20(selected_terminals, curr_tour, route_num)

    p12 = get_subpath(curr_tour, selected_terminals[1], selected_terminals[2], False)
    p34 = get_subpath(curr_tour, selected_terminals[3], selected_terminals[4], False)
    p56 = get_subpath(curr_tour, selected_terminals[5], selected_terminals[6], False)
    p70 = get_subpath(curr_tour, selected_terminals[7], selected_terminals[0], False)

    flag27 = "Precomputed"
    flag05 = "Precomputed"
    flag63 = "Precomputed"
    flag41 = "Precomputed"

    if not terminal_pair_locked[curr_tour[selected_terminals[2]], curr_tour[selected_terminals[7]]]:
        p27, _ = call_with_timeout(ls_sp_timeout_limit, biobj_label_correcting, combined_dict, curr_tour[selected_terminals[2]], curr_tour[selected_terminals[7]])
        terminal_pair_locked[curr_tour[selected_terminals[2]], curr_tour[selected_terminals[7]]] = True
        weighted_paths_dict = update_weighted_paths_dict(weighted_paths_dict, curr_tour[selected_terminals[2]], curr_tour[selected_terminals[7]], p27)
        flag27 = "Biobjective"
    # log_line("  27 ", progress_file, False)

    if not terminal_pair_locked[curr_tour[selected_terminals[0]], curr_tour[selected_terminals[5]]]:
        p05, _ = call_with_timeout(ls_sp_timeout_limit, biobj_label_correcting, combined_dict, curr_tour[selected_terminals[0]], curr_tour[selected_terminals[5]])
        terminal_pair_locked[curr_tour[selected_terminals[0]], curr_tour[selected_terminals[5]]] = True
        weighted_paths_dict = update_weighted_paths_dict(weighted_paths_dict, curr_tour[selected_terminals[0]], curr_tour[selected_terminals[5]], p05)
        flag05 = "Biobjective"
    # log_line("  05 ", progress_file, False)

    if not terminal_pair_locked[curr_tour[selected_terminals[6]], curr_tour[selected_terminals[3]]]:
        p63, _ = call_with_timeout(ls_sp_timeout_limit, biobj_label_correcting, combined_dict, curr_tour[selected_terminals[6]], curr_tour[selected_terminals[3]])
        terminal_pair_locked[curr_tour[selected_terminals[6]], curr_tour[selected_terminals[3]]] = True
        weighted_paths_dict = update_weighted_paths_dict(weighted_paths_dict, curr_tour[selected_terminals[6]], curr_tour[selected_terminals[3]], p63)
        flag63 = "Biobjective"
    # log_line("  63 ", progress_file, False)

    if not terminal_pair_locked[curr_tour[selected_terminals[4]], curr_tour[selected_terminals[1]]]:
        p41, _ = call_with_timeout(ls_sp_timeout_limit, biobj_label_correcting, combined_dict, curr_tour[selected_terminals[4]], curr_tour[selected_terminals[1]])
        terminal_pair_locked[curr_tour[selected_terminals[4]], curr_tour[selected_terminals[1]]] = True
        weighted_paths_dict = update_weighted_paths_dict(weighted_paths_dict, curr_tour[selected_terminals[4]], curr_tour[selected_terminals[1]], p41)
        flag41 = "Biobjective"
    # log_line("  41 ", progress_file, False)

    p27 = [list(x) for x in weighted_paths_dict[curr_tour[selected_terminals[2]], curr_tour[selected_terminals[7]]]]
    p05 = [list(x) for x in weighted_paths_dict[curr_tour[selected_terminals[0]], curr_tour[selected_terminals[5]]]]
    p63 = [list(x) for x in weighted_paths_dict[curr_tour[selected_terminals[6]], curr_tour[selected_terminals[3]]]]
    p41 = [list(x) for x in weighted_paths_dict[curr_tour[selected_terminals[4]], curr_tour[selected_terminals[1]]]]

    # sanity_check_9(p27, p05, p63, p41)
    # log_line("  join ", progress_file, False)
    quad_paths = [remove_consecutive_duplicates(p12 + q2 + p70 + q8 + p56 + q6 + p34 + q4) for q2 in p27 for q8 in p05 for q6 in p63 for q4 in p41]
    operator_stream += f'{flag27},{flag05},{flag63},{flag41},'
    return quad_paths, weighted_paths_dict, operator_stream, weighted_paths_dict
