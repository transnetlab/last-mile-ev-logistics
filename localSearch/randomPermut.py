"""
This file contains functions related to RandomPermute operator

"""
from itertools import product

from helpers.functions import *
from localSearch.commonFunctions import get_time_left
from localSearch.s3opt import update_weighted_paths_dict
from helpers.shortestPath import call_with_timeout, biobj_label_correcting


def random_permut_operator(weighted_paths_dict: dict, operator_stream: str, terminal_pair_locked: dict, move_start_stamp: float, common_arguments: dict) -> tuple:
    """
    This function runs a single iteration of the Random Permute operator.

    Args:
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        operator_stream (str): Log string for the operator
        terminal_pair_locked (dict): Dictionary of terminal pairs that are locked
        move_start_stamp (float): Time the move started
        common_arguments (dict): Common arguments for the local search

    Returns:
        tourset (list): List of tours
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        operator_stream (str): Log string for the operator
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals

    """
    terminal_set = common_arguments["terminal_set"]
    max_nb_runtime = common_arguments["ls_constants"]['max_nb_runtime']
    combined_dict = common_arguments["combined_dict"]
    rm_path_count = common_arguments["ls_constants"]['rm_path_count']
    shuffled_terminals = custom_shuffle(list(terminal_set))[:]
    shuffled_terminals.append(shuffled_terminals[0])
    newpaths = [[] for x in range(len(shuffled_terminals) - 1)]
    ls_sp_timeout_limit = common_arguments["ls_constants"]['ls_sp_timeout_limit']
    total_terminals, terminals_flipped = 0, 0
    for idx in range(len(shuffled_terminals) - 1):
        total_terminals += 1
        terminal1 = shuffled_terminals[idx]
        terminal2 = shuffled_terminals[idx + 1]
        if get_time_left(max_nb_runtime, move_start_stamp) and not terminal_pair_locked[terminal1, terminal2]:
            newPaths, _ = call_with_timeout(ls_sp_timeout_limit, biobj_label_correcting, combined_dict, terminal1, terminal2)
            terminal_pair_locked[terminal1, terminal2] = True
            weighted_paths_dict = update_weighted_paths_dict(weighted_paths_dict, terminal1, terminal2, newPaths)
            terminals_flipped += 1
        newpaths[idx] = custom_shuffle(list(weighted_paths_dict[terminal1, terminal2]))[:rm_path_count]
    product_list = list(product(*newpaths))
    tourset = []
    for one_path in product_list:
        tour = []
        for segment in one_path:
            for node in segment:
                tour.append(node)
        tour = remove_consecutive_duplicates(tour)
        depot_index = tour.index(common_arguments["depot"])
        tourset.append(tuple(cyclize(tour, depot_index)))
    terminals_flipped_per = round(terminals_flipped / total_terminals * 100, 1)
    operator_stream += f"{total_terminals},{terminals_flipped},{terminals_flipped_per},"
    return tourset, weighted_paths_dict, operator_stream, weighted_paths_dict, terminal_pair_locked
