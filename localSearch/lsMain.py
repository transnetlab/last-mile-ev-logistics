"""
This file contains the main functions for the local search
"""
from tqdm import tqdm

from localSearch.quad import *
from localSearch.s3opt import *
from localSearch.s3optTW import *
from localSearch.simpleCycle import *
from localSearch.fixedPerm import *
from localSearch.gapRepair import *
from localSearch.randomPermut import *
from localSearch.commonFunctions import *


def initialize_ls_iteration(no_imp_count: int, common_arguments: dict) -> tuple[float, int, int]:
    """
    This function initializes a local search iteration.

    Args:
        no_imp_count (int): Number of iterations without improvement
        common_arguments (dict): Dictionary containing common arguments

    Returns:
        iteration_start_time (float): Time the iteration started
        no_imp_count (int): Number of iterations without improvement
        ls_iteration_no (int): Local search iteration number
    """
    ls_iteration_no = common_arguments["ls_iteration_no"]
    log_line("Iteration started\n", common_arguments["progress_file"], False)
    iteration_start_time = time.time()
    no_imp_count += 1
    ls_iteration_no += 1
    return iteration_start_time, no_imp_count, ls_iteration_no


def call_local_search(bstsptw_tours: list, bstsp_tours: list, terminal_set: set, L: networkx.classes.multidigraph.MultiDiGraph, energy_dual: dict, direction_dual: dict, time_cost_dict: dict,
            time_window_dict: dict, mean_energy_dual: float, mean_turn_dual: float, weighted_paths_dict: dict, stdev_energy_dual: float, stdev_turn_dual: float, route_num: int, depot: tuple,
            ls_constants: dict) -> tuple:
    """
    This function is the main function for solving a Bi-level Steiner TSP using a local search algorithm.

    Args:
        bstsptw_tours (list): BSTSPTW tours from scalerization
        bstsp_tours (list): BSTSP tours from scalerization
        terminal_set (set): Set of terminal nodes
        L (networkx.classes.multidigraph.MultiDiGraph): Dual graph
        energy_dual (dict): Energy cost
        direction_dual (dict): Direction cost
        time_cost_dict (dict): Dictionary of travel times
        time_window_dict (dict): The dictionary containing the time windows for each terminal
        mean_energy_dual (float): Mean energy cost for the dual graph
        mean_turn_dual (float): Mean turn cost for the dual graph
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        stdev_energy_dual (float): Standard deviation of energy costs
        stdev_turn_dual (float): Standard deviation of turn costs
        route_num (int): Route number
        depot (tuple): Depot node
        ls_constants (dict): Constants for the local search

    Returns:
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        time_for_last_imp (float): Time for last improvement in LS
        parents (dict): Dictionary of parents for tracking lineages
        total_ls_time (float): Total time taken for the local search
        set_p (list): List of tours that violate time-windows
        ls_iteration_no (int): Local search iteration number

    """
    (no_imp_count, ls_iteration_no, combined_dict, ls_start_time, time_for_last_imp, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, parents, set_z, set_p, set_y,
     initial_paths, terminal_pair_locked, progress_file, allowed_ls_time) = (initialize_ls(energy_dual, direction_dual, L, bstsptw_tours, bstsp_tours, ls_constants, time_window_dict, route_num))

    quickest_paths_dict, warm_start_time = warm_start_quickest_paths(terminal_set, L, time_cost_dict)

    common_arguments = {"terminal_set": terminal_set, "time_cost_dict": time_cost_dict, "time_window_dict": time_window_dict, "combined_dict": combined_dict, "ls_start_time": ls_start_time,
                        "depot": depot, "ls_constants": ls_constants, "progress_file": progress_file, "mean_energy_dual": mean_energy_dual, "mean_turn_dual": mean_turn_dual,
                        "stdev_energy_dual": stdev_energy_dual, "stdev_turn_dual": stdev_turn_dual, "route_num": route_num, "energy_dual": energy_dual, "direction_dual": direction_dual, "L": L,
                        "ls_iteration_no": ls_iteration_no}

    with (tqdm(total=allowed_ls_time, desc="Processing") as pbar):
        while get_time_left(allowed_ls_time, ls_start_time):
            iteration_start_time, no_imp_count, common_arguments["ls_iteration_no"] = (initialize_ls_iteration(no_imp_count, common_arguments))
            filter_var = update_filter_var_s3opt(set_z, common_arguments)

            log_line("S3opt\n", progress_file, False)
            if ls_constants["nbd_flags"]["S3opt"] and len(set_z) > 1 and get_time_left(allowed_ls_time, ls_start_time) and len(terminal_set) >= 6:
                (set_z, set_p, set_y, no_imp_count, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours) = (
                    call_s3_opt(no_imp_count, set_z, set_p, set_y, parents, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours,
                                common_arguments, filter_var))

            log_line("FixedPerm\n", progress_file, False)
            if ls_constants["nbd_flags"]["FixedPerm"] and len(set_z) > 0 and get_time_left(allowed_ls_time, ls_start_time):
                (set_z, set_p, set_y, no_imp_count, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, terminal_pair_locked) = (
                    call_fixed_perm(no_imp_count, set_z, set_p, set_y, parents, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours,
                                    common_arguments, terminal_pair_locked, 'z'))

            log_line("RandomPermute\n", progress_file, False)
            if ls_constants["nbd_flags"]["RandomPermute"] and get_time_left(allowed_ls_time, ls_start_time):
                (set_z, set_p, set_y, no_imp_count, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, terminal_pair_locked) = (
                    call_random_permute(no_imp_count, set_z, set_p, set_y, parents, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours,
                                        nbd_optimal_tours, common_arguments, terminal_pair_locked))

            log_line("Quad\n", progress_file, False)
            if ls_constants["nbd_flags"]["Quad"] and len(terminal_set) >= 8 and len(set_z) > 0 and get_time_left(allowed_ls_time, ls_start_time):
                (set_z, set_p, set_y, no_imp_count, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, terminal_pair_locked) = (
                    call_quad(no_imp_count, set_z, set_p, set_y, parents, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours,
                              common_arguments, terminal_pair_locked))

            log_line("S3optTW\n", progress_file, False)
            if ls_constants["nbd_flags"]["S3optTW"] and len(set_p) > 0 and get_time_left(allowed_ls_time, ls_start_time) and len(terminal_set) >= 6:
                (set_z, set_p, set_y, no_imp_count, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours) = (
                    call_s3_opt_tw(no_imp_count, set_z, set_p, set_y, parents, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours,
                                   common_arguments))

            log_line("GapRepair\n", progress_file, False)
            if ls_constants["nbd_flags"]["GapRepair"] and len(set_p) > 0 and get_time_left(allowed_ls_time, ls_start_time):
                (set_z, set_p, set_y, no_imp_count, time_for_last_imp, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, quickest_paths_dict, weighted_paths_dict) = (
                    call_gap_repair(no_imp_count, set_z, set_p, set_y, parents, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours,
                                    common_arguments, quickest_paths_dict))

            log_line("FixedPerm\n", progress_file, False)
            if ls_constants["nbd_flags"]["FixedPerm"] and len(set_y) > 0 and get_time_left(allowed_ls_time, ls_start_time):
                (set_z, set_p, set_y, no_imp_count, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, terminal_pair_locked) = (
                    call_fixed_perm(no_imp_count, set_z, set_p, set_y, parents, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours,
                                    common_arguments, terminal_pair_locked, 't'))

            # log_line("SimpleCycle\n", progress_file, False)
            # if ls_constants["nbd_flags"]["SimpleCycle"] and len(set_z) > 0 and get_time_left(allowed_ls_time, ls_start_time):
            #     (set_z, set_p, set_y, no_imp_count, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours) = (
            #         call_simple_cycle(no_imp_count, set_z, set_p, set_y, parents, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours,
            #                           common_arguments))

            iteration_time = time.time() - iteration_start_time
            max_penalty_in_explore_set = get_max_penalty_in_set_p(set_p)
            log_ls_iteration(set_z, set_p, set_y, iteration_time, max_penalty_in_explore_set, common_arguments)
            pbar.update(iteration_time)

    total_ls_time = time.time() - ls_start_time
    log_final_ls(operator_call_count, nbd_tw_feasible_tours, nbd_optimal_tours, operator_times, common_arguments)

    log_line("LS Exiting\n", progress_file, False)
    return set_z, time_for_last_imp, parents, total_ls_time, set_p, common_arguments["ls_iteration_no"]


def call_s3_opt(no_imp_count: int, set_z: list, set_p: list, set_y: list, parents: dict, time_for_last_imp: float, weighted_paths_dict: dict, operator_call_count: dict, operator_times: dict,
                nbd_tw_feasible_tours: dict, nbd_optimal_tours: dict, common_arguments: dict, filter_var: tuple) -> tuple:
    """
    This function calls the S3opt operator.

    Args:
        no_imp_count (int): Number of iterations without improvement
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        parents (dict): Dictionary of parents for tracking lineages
        time_for_last_imp (float): Time for last improvement in LS
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        operator_call_count (dict): Call counts of each operator
        operator_times (dict): Times taken by each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal tours for each operator
        filter_var (tuple): Filter variable for the S3opt operator
        common_arguments (dict): Dictionary containing common arguments

    Returns:
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        no_imp_count (int): Number of iterations without improvement
        time_for_last_imp (float): Time for last improvement in LS
        operator_call_count (dict): Call counts of each operator
        operator_times (dict): Times taken by each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal tours for each operator
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
    """
    operator_name = 'S3opt'
    move_start_stamp = time.time()
    operator_stream = initialize_operator(common_arguments["progress_file"])

    selected_index = select_tour_using_kde(set_z, common_arguments)
    curr_tour, curr_tour_energy, curr_tour_turns = set_z[selected_index]
    operator_stream += f"{selected_index},{0},"
    current_ef = [(e, t) for p, e, t in set_z]
    log_line("  S1-E ", common_arguments["progress_file"], False)
    cycle_subpath, weighted_paths_dict, operator_stream = execute_s3_opt_step_1(curr_tour, curr_tour_energy, curr_tour_turns, current_ef, filter_var, weighted_paths_dict,
                                                                                operator_stream, move_start_stamp, common_arguments)
    log_line("  S2-E ", common_arguments["progress_file"], False)
    step1_time, step2_start_stamp = time.time() - move_start_stamp, time.time()
    output_tours, returnFlag = execute_s3_opt_step_2(curr_tour, cycle_subpath, weighted_paths_dict, move_start_stamp, common_arguments)

    step2_time = time.time() - step2_start_stamp
    move_time = time.time() - move_start_stamp
    step1_time, step2_time = round(step1_time / move_time * 100, 1), round(step2_time / move_time * 100, 1)
    operator_stream += f"{round(step1_time)},{round(step2_time)},{returnFlag},"
    log_line("  US-E ", common_arguments["progress_file"], False)
    (set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream, update_set_time) = (
        update_sets(output_tours, curr_tour, operator_name, parents, set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream,
                    common_arguments))

    post_process_operator(move_time, update_set_time, operator_call_count, operator_name, operator_stream, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, common_arguments)
    return set_z, set_p, set_y, no_imp_count, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours


def call_s3_opt_tw(no_imp_count: int, set_z: list, set_p: list, set_y: list, parents: dict, time_for_last_imp: float, weighted_paths_dict: dict, operator_call_count: dict, operator_times: dict,
                   nbd_tw_feasible_tours: dict, nbd_optimal_tours: dict, common_arguments: dict) -> tuple:
    """
    This function calls the S3optTW operator.

    Args:
        no_imp_count (int): Number of iterations without improvement
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        parents (dict): Dictionary of parents for tracking lineages
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        time_for_last_imp (float): Time for last improvement in LS
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        operator_call_count (dict): Call counts of each operator
        operator_times (dict): Times taken by each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal tours for each operator
        common_arguments (dict): Dictionary containing common arguments

    Returns:
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        no_imp_count (int): Number of iterations without improvement
        time_for_last_imp (float): Time for last improvement in LS
        operator_call_count (dict): Call counts of each operator
        operator_times (dict): Times taken by each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal tours for each operator
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
    """
    operator_name = 'S3optTW'
    move_start_stamp = time.time()
    operator_stream = initialize_operator(common_arguments["progress_file"])

    selected_index = get_random(list(range(len(set_p))), [])
    curr_tour, curr_tour_energy, curr_tour_turns, _ = set_p[selected_index]
    operator_stream += f"{selected_index},{0},"
    log_line("  S1-E ", common_arguments["progress_file"], False)
    listof_cycle_subpath, weighted_paths_dict, operator_stream = execute_s3_opt_tw_step_1(curr_tour, curr_tour_energy, curr_tour_turns, weighted_paths_dict, operator_stream, move_start_stamp,
                                                                                          common_arguments)
    step1_time, step2_start_stamp = time.time() - move_start_stamp, time.time()
    log_line("  S2-E ", common_arguments["progress_file"], False)
    output_tours, returnFlag, operator_stream = execute_s3_opt_tw_step_2(curr_tour, listof_cycle_subpath, weighted_paths_dict, move_start_stamp, operator_stream, common_arguments)
    step2_time = time.time() - step2_start_stamp
    move_time = time.time() - move_start_stamp
    step1_time, step2_time = round(step1_time / move_time * 100, 1), round(step2_time / move_time * 100, 1)
    operator_stream += f"{round(step1_time)},{round(step2_time)},{returnFlag},"
    log_line("  US-E ", common_arguments["progress_file"], False)
    (set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream, update_set_time) = (
        update_sets(output_tours, curr_tour, operator_name, parents, set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream,
                    common_arguments))

    post_process_operator(move_time, update_set_time, operator_call_count, operator_name, operator_stream, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, common_arguments)
    return set_z, set_p, set_y, no_imp_count, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours


def call_gap_repair(no_imp_count: int, set_z: list, set_p: list, set_y: list, parents: dict, time_for_last_imp: float, weighted_paths_dict: dict, operator_call_count: dict, operator_times: dict,
                    nbd_tw_feasible_tours: dict, nbd_optimal_tours: dict, common_arguments: dict, quickest_paths_dict: dict) -> tuple:
    """
    This function calls the GapRepair operator.

    Args:
        no_imp_count (int): Number of iterations without improvement
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        parents (dict): Dictionary of parents for tracking lineages
        time_for_last_imp (float): Time for last improvement in LS
        operator_call_count (dict): Call counts of each operator
        operator_times (dict): Times taken by each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal tours for each operator
        common_arguments (dict): Dictionary containing common arguments
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals

    Returns:
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        no_imp_count (int): Number of iterations without improvement
        time_for_last_imp (float): Time for last improvement in LS
        operator_call_count (dict): Call counts of each operator
        operator_times (dict): Times taken by each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal tours for each operator
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
    """

    operator_name = 'GapRepair'
    gr_max_tries = common_arguments["ls_constants"]["gr_max_tries"]
    max_nb_runtime = common_arguments["ls_constants"]["max_nb_runtime"]
    progress_file = common_arguments["progress_file"]

    tours_analyzed = []
    move_start_stamp = time.time()
    for x in range(gr_max_tries):
        start = time.time()
        if not get_time_left(max_nb_runtime, move_start_stamp):
            break
        selected_index = get_random(list(range(len(set_p))), [])
        operator_stream = initialize_operator(progress_file)
        curr_tour, e, t, p = set_p[selected_index]
        if curr_tour in tours_analyzed:
            continue
        tours_analyzed.append(curr_tour)
        operator_stream += f"{selected_index},{0},"

        output_tours, quickest_paths_dict, operator_stream, weighted_paths_dict = gap_repair_operator(curr_tour, operator_stream, quickest_paths_dict, weighted_paths_dict, common_arguments)

        move_time = time.time() - start
        (set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream, update_set_time) = (
            update_sets(output_tours, curr_tour, operator_name, parents, set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream,
                        common_arguments))

        post_process_operator(move_time, update_set_time, operator_call_count, operator_name, operator_stream, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, common_arguments)

    return set_z, set_p, set_y, no_imp_count, time_for_last_imp, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, quickest_paths_dict, weighted_paths_dict


def call_quad(no_imp_count: int, set_z: list, set_p: list, set_y: list, parents: dict, time_for_last_imp: float, weighted_paths_dict: dict, operator_call_count: dict, operator_times: dict,
              nbd_tw_feasible_tours: dict, nbd_optimal_tours: dict, common_arguments: dict, terminal_pair_locked: dict) -> tuple:
    """
    This function calls the Quad operator.

    Args:
        no_imp_count (int): Number of iterations without improvement
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        parents (dict): Dictionary of parents for tracking lineages
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        time_for_last_imp (float): Time for last improvement in LS
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        operator_call_count (dict): Call counts of each operator
        operator_times (dict): Times taken by each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict) Dictionary containing optimal tours for each operator
        common_arguments (dict): Dictionary containing common arguments
        terminal_pair_locked (dict): Dictionary of locked terminal pairs

    Returns:
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        no_imp_count (int): Number of iterations without improvement
        time_for_last_imp (float): Time for last improvement in LS
        operator_call_count (dict): Call counts of each operator
        operator_times (dict): Times taken by each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal tours for each operator
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
    """
    quad_max_tries = common_arguments["ls_constants"]["quad_max_tries"]
    max_nb_runtime = common_arguments["ls_constants"]["max_nb_runtime"]
    operator_name = 'Quad'
    tours_analyzed = []
    move_start_stamp = time.time()
    for x in range(quad_max_tries):
        start = time.time()
        if not get_time_left(max_nb_runtime, move_start_stamp):
            break
        operator_stream = initialize_operator(common_arguments["progress_file"])
        curr_tour, e, t = get_random(set_z, [])
        if curr_tour in tours_analyzed: continue
        tours_analyzed.append(curr_tour)
        selected_index = set_z.index((curr_tour, e, t))
        operator_stream += f"{selected_index},"
        output_tours, weighted_paths_dict, operator_stream, terminal_pair_locked = quad_operator(curr_tour, weighted_paths_dict, operator_stream, terminal_pair_locked, x, common_arguments)
        move_time = time.time() - start

        (set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream, update_set_time) = (
            update_sets(output_tours, curr_tour, operator_name, parents, set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream,
                        common_arguments))

        post_process_operator(move_time, update_set_time, operator_call_count, operator_name, operator_stream, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, common_arguments)

    return set_z, set_p, set_y, no_imp_count, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, terminal_pair_locked


def call_fixed_perm(no_imp_count: int, set_z: list, set_p: list, set_y: list, parents: dict, time_for_last_imp: float, weighted_paths_dict: dict, operator_call_count: dict, operator_times: dict,
                    nbd_tw_feasible_tours: dict, nbd_optimal_tours: dict, common_arguments: dict, terminal_pair_locked: dict, fp_type: str):
    """
    This function calls the FixedPerm operator.

    Args:
        no_imp_count (int): Number of iterations without improvement
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        parents (dict): Dictionary of parents for tracking lineages
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        time_for_last_imp (float): Time for last improvement in LS
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        operator_call_count (dict): Call counts of each operator
        operator_times (dict): Times taken by each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal tours for each operator
        terminal_pair_locked (dict): Dictionary of locked terminal pairs
        fp_type (str): Type of FixedPerm operator
        common_arguments (dict): Dictionary containing common arguments

    Returns:
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        no_imp_count (int): Number of iterations without improvement
        time_for_last_imp (float): Time for last improvement in LS
        operator_call_count (dict): Call counts of each operator
        operator_times (dict): Times taken by each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal tours for each operator
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
    """
    operator_name = 'FixedPerm'
    tours_analyzed = []
    move_start_stamp = time.time()
    fp_max_tries = common_arguments["ls_constants"]["fp_max_tries"]
    max_nb_runtime = common_arguments["ls_constants"]["max_nb_runtime"]
    for x in range(fp_max_tries):
        start = time.time()
        if not get_time_left(max_nb_runtime, move_start_stamp):
            break
        operator_stream = initialize_operator(common_arguments["progress_file"])
        if fp_type == 'z':
            curr_tour, e, t, operator_stream = get_random_skewed_tour(set_z, operator_stream, x, common_arguments)
            if curr_tour:
                selected_index = set_z.index((curr_tour, e, t))
                operator_stream += f"{selected_index},"
            else:
                operator_stream += f"NaN,"
        else:
            curr_tour, e, t = get_random(set_y, [])
            selected_index = set_y.index((curr_tour, e, t))
            operator_stream += f"0,0,{selected_index},"
        if curr_tour in tours_analyzed:
            continue
        tours_analyzed.append(curr_tour)
        output_tours, weighted_paths_dict, operator_stream, terminal_pair_locked = fixed_perm_operator(weighted_paths_dict, curr_tour, operator_stream, terminal_pair_locked, start, common_arguments)
        move_time = time.time() - start
        (set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream, update_set_time) = (
            update_sets(output_tours, curr_tour, operator_name, parents, set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream,
                        common_arguments))

        post_process_operator(move_time, update_set_time, operator_call_count, operator_name, operator_stream, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, common_arguments)

    return set_z, set_p, set_y, no_imp_count, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, terminal_pair_locked


def call_random_permute(no_imp_count: int, set_z: list, set_p: list, set_y: list, parents: dict, time_for_last_imp: float, weighted_paths_dict: dict, operator_call_count: dict, operator_times: dict,
                        nbd_tw_feasible_tours: dict, nbd_optimal_tours: dict, common_arguments: dict, terminal_pair_locked: dict) -> tuple:
    """
    This function calls the RandomPermute operator.

    Args:
        no_imp_count (int): Number of iterations without improvement
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        parents (dict): Dictionary of parents for tracking lineages
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        time_for_last_imp (float): Time for last improvement in LS
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        operator_call_count (dict): Call counts of each operator
        operator_times (dict): Times taken by each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal tours for each operator
        common_arguments (dict): Dictionary containing common arguments
        terminal_pair_locked (dict): Dictionary of locked terminal pairs

    Returns:
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        no_imp_count (int): Number of iterations without improvement
        time_for_last_imp (float): Time for last improvement in LS
        operator_call_count (dict): Call counts of each operator
        operator_times (dict): Times taken by each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal tours for each operator
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
    """
    operator_name = 'RandomPermute'
    rm_tries = common_arguments["ls_constants"]["rm_tries"]
    max_nb_runtime = common_arguments["ls_constants"]["max_nb_runtime"]
    move_start_stamp = time.time()
    try:
        curr_tour, _, _ = set_z[get_random(list(range(len(set_z))), [])]
    except:
        curr_tour = []
    for x in range(rm_tries):
        start = time.time()
        if not get_time_left(max_nb_runtime, move_start_stamp):
            break
        operator_stream = initialize_operator(common_arguments["progress_file"])
        output_tours, weighted_paths_dict, operator_stream, weighted_paths_dict, terminal_pair_locked = random_permut_operator(weighted_paths_dict, operator_stream, terminal_pair_locked, start,
                                                                                                                               common_arguments)
        move_time = time.time() - start
        (set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream, update_set_time) = (
            update_sets(output_tours, curr_tour, operator_name, parents, set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream,
                        common_arguments))
        post_process_operator(move_time, update_set_time, operator_call_count, operator_name, operator_stream, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, common_arguments)

    return set_z, set_p, set_y, no_imp_count, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, terminal_pair_locked


def call_simple_cycle(no_imp_count: int, set_z: list, set_p: list, set_y: list, parents: dict, time_for_last_imp: float, weighted_paths_dict: dict, operator_call_count: dict, operator_times: dict,
                      nbd_tw_feasible_tours: dict, nbd_optimal_tours: dict, common_arguments: dict) -> tuple:
    """
    This function calls the SimpleCycle operator.

    Args:
        parents (dict): Dictionary of parents for tracking lineages
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal tours for each operator
        no_imp_count (int): Number of iterations without improvement
        time_for_last_imp (float): Time for last improvement in LS
        operator_call_count (dict): Call counts of each operator
        operator_times (dict): Times taken by each operator
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
        common_arguments (dict): Dictionary containing common arguments

    Returns:
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        no_imp_count (int): Number of iterations without improvement
        time_for_last_imp (float): Time for last improvement in LS
        operator_call_count (dict): Call counts of each operator
        operator_times (dict): Times taken by each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal tours for each operator
        quickest_paths_dict (dict): Dictionary of shortest paths between all pairs of terminals
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
    """
    operator_name = 'SimpleCycle'
    move_start_stamp = time.time()
    operator_stream = initialize_operator(common_arguments["progress_file"])

    curr_tour, e, t = get_random(set_z, [])

    output_tours, operator_stream = simple_cycle_operator(curr_tour, operator_stream, move_start_stamp, common_arguments)
    move_time = time.time() - move_start_stamp

    (set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream, update_set_time) = (
        update_sets(output_tours, curr_tour, operator_name, parents, set_z, set_p, set_y, nbd_tw_feasible_tours, nbd_optimal_tours, no_imp_count, time_for_last_imp, operator_stream, common_arguments))

    post_process_operator(move_time, update_set_time, operator_call_count, operator_name, operator_stream, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours, common_arguments)

    return set_z, set_p, set_y, no_imp_count, time_for_last_imp, weighted_paths_dict, operator_call_count, operator_times, nbd_tw_feasible_tours, nbd_optimal_tours
