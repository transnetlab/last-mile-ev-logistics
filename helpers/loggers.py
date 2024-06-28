"""
This module contains functions for logging the results of the local search and IP algorithms
"""
import os
import time
import shutil
import pickle
import networkx


def final_log_file(G: networkx.classes.multidigraph.MultiDiGraph, L: networkx.classes.multidigraph.MultiDiGraph, terminal_set: set, bstsptw_tours: list, bstsp_tours: list, set_z: list, parents: dict,
                   route_num: int, time_for_last_imp: float, ls_constants: dict, tw_term_count: int, total_ls_time: float, set_p: list, warm_start_time: float, ls_iteration_no: int) -> None:
    """
    This function logs the results after applying the Local Search operator to a problem.

    Args:
        G (networkx.classes.multidigraph.MultiDiGraph): Primal graph
        L (networkx.classes.multidigraph.MultiDiGraph): Dual graph
        terminal_set (set): Set of terminal nodes
        bstsptw_tours (list): BSTSPTW tours from scalerization
        bstsp_tours (list): BSTSP tours from scalerization
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        parents (dict): Dictionary of parents for tracking lineages
        route_num (int): Route number
        time_for_last_imp (float): Time for last improvement in LS
        ls_constants (dict): Constants for the local search
        tw_term_count (int): Count of terminal nodes with time windows
        total_ls_time (float): Total time taken for the local search
        set_p (list): List of tours that violate time-windows
        warm_start_time (float): Time taken for warm start
        ls_iteration_no (int): Local search iteration number

    Returns:
        None
    """
    folder_path = f'./logs/final_LSlog_file.txt'

    try:
        _, bstsptw_energy, bstsptw_turns = zip(*bstsptw_tours)
    except ValueError:
        bstsptw_energy, bstsptw_turns = [], []
    try:
        _, bstsp_energy, bstsp_turns, _ = zip(*bstsp_tours)
    except ValueError:
        bstsp_energy, bstsp_turns = [], []
    try:
        _, ls_energy, ls_turns = zip(*set_z)
    except ValueError:
        ls_energy, ls_turns = [], []

    parents2 = dict(parents)
    root_nodes = [p for p, e, t in bstsptw_tours] + [p for p, e, t, penalty in bstsp_tours] + [tuple()]
    ls_paths_final = [tuple(p) for p, _, _ in set_z]
    move_counter = {'FixedPerm': 0, 'S3opt': 0, 'S3optTW': 0, 'Quad': 0, 'GapRepair': 0, 'RandomPermute': 0, 'SimpleCycle': 0}
    for parent_tour in ls_paths_final:
        while parent_tour not in root_nodes:
            parent_tour2, move = parents2[parent_tour][0]
            move_counter[move] += 1
            parent_tour2 = tuple(parent_tour2)
            parent_tour = parent_tour2

    save_ls_results(route_num, set_z, time_for_last_imp, move_counter, total_ls_time, set_p)

    line = f"{route_num},{len(G.nodes)},{len(G.edges)},{len(L.nodes)},{len(L.edges)},{len(terminal_set)},{tw_term_count},{len(bstsp_turns)},{len(bstsptw_turns)},{True},{len(ls_turns) - len(bstsptw_turns)},{len(set_p)},{min(round(time_for_last_imp + ls_constants['allowed_init_time'], 1), ls_constants['allowed_time'])},{ls_constants['allowed_init_time']},{ls_constants['allowed_time']},{move_counter['S3opt']},{move_counter['S3optTW']},{move_counter['GapRepair']},{move_counter['FixedPerm']},{move_counter['Quad']},{move_counter['RandomPermute']},{move_counter['SimpleCycle']},{warm_start_time},{ls_iteration_no}\n"
    log_line(line, folder_path, False)
    return None


def make_log_dirs(folder_path: str) -> None:
    """
    This function creates directories for saving the log files.

    Args:
        folder_path (str): Folder path

    Returns:
        None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(f'{folder_path}/scalerization/'):
        os.makedirs(f'{folder_path}/scalerization/')
    if not os.path.exists(f'{folder_path}/plots/'):
        os.makedirs(f'{folder_path}/plots/')
    if not os.path.exists(f'{folder_path}/PrecomputedWeightedSP/'):
        os.makedirs(f'{folder_path}/PrecomputedWeightedSP/')
    if not os.path.exists(f'{folder_path}/PrecomputedQuickestSP/'):
        os.makedirs(f'{folder_path}/PrecomputedQuickestSP/')
    return None


def clean_mip_logs() -> None:
    """
    This function cleans the logs created by the MIP processing.

    Returns:
        None
    """
    path_for_ip_results = './logs/IPRes'
    try:
        shutil.rmtree(path_for_ip_results)
    except:
        pass
    if not os.path.exists(path_for_ip_results):
        os.makedirs(path_for_ip_results)

    path_for_ip_pickle_files = './logs/IPOutput'
    if not os.path.exists(path_for_ip_pickle_files):
        os.makedirs(path_for_ip_pickle_files)

    file_path = f'./logs/IPscalerizationEndConditions.txt'
    write_line = "Route_num,Condition\n"
    log_line(write_line, file_path, True)
    return None


def clean_summary_logs() -> None:
    """
    This function cleans the summary logs from a previous run.

    Returns:
        None
    """
    os.makedirs('./logs/', exist_ok=True)
    write_line = "RouteNum,G_nodes,G_edges,L_nodes,L_edges,Terminals,Terminals_withTW,BSTSTP_paths,BSTSTPTW_paths,ScalarizationFoundSol,OnlyLS_paths,SetP,TimeForLastImp,AllowedInitTime,TotalTime,S3opt,S3optTW,GapRepair,FixedPerm,Quad,RandomPermute,SimpleCycle,warm_start_time,LSiteration\n"
    file_path = f'./logs/final_LSlog_file.txt'
    log_line(write_line, file_path, True)

    write_line = "Route_num,Time,LeastTurnTourTime,LeastEnergyTourTime,Points\n"
    file_path = f'./logs/endPoints.txt'
    log_line(write_line, file_path, True)

    write_line = "Route_num\n"
    file_path = f'./logs/googleORtools_failed.txt'
    log_line(write_line, file_path, True)

    write_line = "Route_num\n"
    file_path = f'./logs/johnsan_failed.txt'
    log_line(write_line, file_path, True)

    write_line = "RouteNum,alpha,beta,JhonsanTime,GoogleOR,JhonsanTime%,GoogleOR%,Bellman,Dijkstra,Bellman%,Dijkstra%\n"
    file_path = f'./logs/mixintprog.txt'
    log_line(write_line, file_path, True)

    write_line = "Route_num\n"
    file_path = f'./logs/Routes_withNoTW.txt'
    log_line(write_line, file_path, True)

    write_line = "Route_num,Time,TotalPoints,BSTSPTW,BSTSP\n"
    file_path = f'./logs/scalerization.txt'
    log_line(write_line, file_path, True)

    file_path = f'./logs/scalerizationEndConditions.txt'
    write_line = "Route_num,Condition\n"
    log_line(write_line, file_path, True)
    return None


def clear_collective_stats() -> None:
    """
    This function clears the summary files from previous post processing steps and recreates the directories.

    Returns:
        None
    """
    folder_path = "./logs/summary"
    try:
        shutil.rmtree(folder_path)
    except:
        pass
    if not os.path.exists(f'{folder_path}/'):
        os.makedirs(f'{folder_path}/')
    if not os.path.exists(f'{folder_path}/S3opt'):
        os.makedirs(f'{folder_path}/S3opt')
    if not os.path.exists(f'{folder_path}/final'):
        os.makedirs(f'{folder_path}/final')
    if not os.path.exists(f'{folder_path}/S3optTW'):
        os.makedirs(f'{folder_path}/S3optTW')
    if not os.path.exists(f'{folder_path}/GapRepair'):
        os.makedirs(f'{folder_path}/GapRepair')
    if not os.path.exists(f'{folder_path}/FixedPerm'):
        os.makedirs(f'{folder_path}/FixedPerm')
    if not os.path.exists(f'{folder_path}/Quad'):
        os.makedirs(f'{folder_path}/Quad')
    if not os.path.exists(f'{folder_path}/RandomPermute'):
        os.makedirs(f'{folder_path}/RandomPermute')
    if not os.path.exists(f'{folder_path}/SimpleCycle'):
        os.makedirs(f'{folder_path}/SimpleCycle')
    if not os.path.exists(f'{folder_path}/collective'):
        os.makedirs(f'{folder_path}/collective')
    return None


def clean_route_logs(route_num: int) -> None:
    """
    This method cleans the logs for a particular route.

    Args:
        route_num (int): Route number

    Returns:
        None
    """
    folder_path = f'./logs/{route_num}'
    try:
        shutil.rmtree(folder_path)
    except:
        pass
    make_log_dirs(folder_path)

    write_line = "NbdName,Tours,InitTime,TWTime,reorderTime,penaltyTime,ParetoTime,TotalTime\n"
    file_path = f'{folder_path}/updateSet.txt'
    log_line(write_line, file_path, True)

    write_line = "RouteNum,alpha,beta,JhonsanTime,GoogleOR,JhonsanTime%,GoogleOR%,Bellman,Dijkstra,Bellman%,Dijkstra%\n"
    file_path = f'{folder_path}/mixintprog.txt'
    log_line(write_line, file_path, True)

    file_path = f'./{folder_path}/ls_iteration.txt'
    write_line = "Iteration,Z,P,T,S3opt,S3optTW,GapRepair,FixedPerm,Quad,RandomPermute,SimpleCycle,Time,MaxPenalty\n"
    log_line(write_line, file_path, True)

    file_path = f'{folder_path}/ls_final.txt'
    write_line = "Method,TW_Count,Optimal_Count,Call_Count,Total_Time\n"
    log_line(write_line, file_path, True)

    file_path = f'{folder_path}/S3opt.txt'
    write_line = "SelectedIndex,Alpha,Init_Candidate,Final_candidate,Reduction,GainImproved,LowestGain,BestGain,Step1Time,Step2Time,ReturnFlag,PathsFound,TWPaths,NonTWPaths,ParetoPaths,NonParetoPaths,ExplorePaths,NonExplorePaths,Time,Move_time,update_set_time,CallCount,OverallTWCount,OverallOptimalCount,LSiterationNo\n"
    log_line(write_line, file_path, True)

    file_path = f'{folder_path}/S3optTW.txt'
    write_line = "SelectedIndex,Alpha,NumberOfCycles,Init_Candidate,Final_candidate,Reduction,GainImproved,LowestGain,BestGain,Step1Time,InitTours,FinalTours,ParetoTime,Step2Time,ReturnFlag,PathsFound,TWPaths,NonTWPaths,ParetoPaths,NonParetoPaths,ExplorePaths,NonExplorePaths,Time,Move_time,update_set_time,CallCount,OverallTWCount,OverallOptimalCount,LSiterationNo\n"
    log_line(write_line, file_path, True)

    file_path = f'{folder_path}/GapRepair.txt'
    write_line = "SelectedIndex,Alpha,MaxWidth,DestroyTimePer,RepairTimePer,QuickestTWFlag,WeightTWFlag,ReturnFlag,PathsFound,TWPaths,NonTWPaths,ParetoPaths,NonParetoPaths,ExplorePaths,NonExplorePaths,Time,Move_time,update_set_time,CallCount,OverallTWCount,OverallOptimalCount,LSiterationNo\n"
    log_line(write_line, file_path, True)

    file_path = f'{folder_path}/FixedPerm.txt'
    write_line = "SetZ,FilteredSetZ,SelectedIndex,T_flipped,T_fetched,T_timeout,Obj,Flag,SwapTime%,EvaluateTime%,PathsFound,TWPaths,NonTWPaths,ParetoPaths,NonParetoPaths,ExplorePaths,NonExplorePaths,Time,Move_time,update_set_time,CallCount,OverallTWCount,OverallOptimalCount,LSiterationNo\n"
    log_line(write_line, file_path, True)

    file_path = f'{folder_path}/Quad.txt'
    write_line = "SelectedIndex,Flag27,Flag05,Flag63,Flag41,PathsFound,TWPaths,NonTWPaths,ParetoPaths,NonParetoPaths,ExplorePaths,NonExplorePaths,Time,Move_time,update_set_time,CallCount,OverallTWCount,OverallOptimalCount,LSiterationNo\n"
    log_line(write_line, file_path, True)

    file_path = f'{folder_path}/RandomPermute.txt'
    write_line = "TotalTerminals,Flipped,FlippedPer,PathsFound,TWPaths,NonTWPaths,ParetoPaths,NonParetoPaths,ExplorePaths,NonExplorePaths,Time,Move_time,update_set_time,CallCount,OverallTWCount,OverallOptimalCount,LSiterationNo\n"
    log_line(write_line, file_path, True)

    file_path = f'{folder_path}/SimpleCycle.txt'
    write_line = "CycleCount,PathsFound,TWPaths,NonTWPaths,ParetoPaths,NonParetoPaths,ExplorePaths,NonExplorePaths,Time,Move_time,update_set_time,CallCount,OverallTWCount,OverallOptimalCount,LSiterationNo\n"
    log_line(write_line, file_path, True)

    file_path = f'{folder_path}/endPoints.txt'
    write_line = "Route_num,Time,LeastTurnTourTime,LeastEnergyTourTime,Points\n"
    log_line(write_line, file_path, True)

    file_path = f'{folder_path}/scalerization.txt'
    write_line = "Route_num,Time,TotalPoints,BSTSPTW,BSTSP\n"
    log_line(write_line, file_path, True)

    file_path = f'{folder_path}/johnsan_failed.txt'
    write_line = "Route_num\n"
    log_line(write_line, file_path, True)

    file_path = f'{folder_path}/googleORtools_failed.txt'
    write_line = "Route_num\n"
    log_line(write_line, file_path, True)

    file_path = f'{folder_path}/Routes_withNoTW.txt'
    write_line = "Route_num\n"
    log_line(write_line, file_path, True)

    file_path = f'{folder_path}/progressTrack.txt'
    write_line = "\n"
    log_line(write_line, file_path, True)
    return None


def log_line(line: str, filename: str, clean: bool) -> None:
    """
    This method logs a line of text to a file.

    Args:
        line (str): Line to log
        filename (str): File path
        clean (bool): Clean the file before writing

    Returns:
        None
    """
    with open(filename, 'a') as file:
        if clean:
            file.truncate(0)
        file.write(line)
    return None


def log_endpoints(route_num: int, start_time: float, scalerization_tours: list, least_turn_tour_time: float, least_energy_tour_time: float) -> None:
    """
    This method logs to a file information on scalarization end points.

    Args:
        route_num (int): Route number
        start_time (float): Start time
        scalerization_tours (list): list of Scalerization tours
        least_turn_tour_time (float): Time required to find the least turn tour
        least_energy_tour_time (float) Time required to find the least energy tour

    Returns:
        None
    """
    t = round(time.time() - start_time)
    line = f"{route_num},{t},{least_turn_tour_time},{least_energy_tour_time},{len(scalerization_tours)}\n"
    folder_path = f'./logs/{route_num}/endPoints.txt'
    log_line(line, folder_path, False)
    folder_path = f'./logs/endPoints.txt'
    log_line(line, folder_path, False)
    return None


def log_ortools_failure(route_num: int) -> None:
    """
    This method logs information corresponding to the failure of Google OR Tools.

    Args:
        route_num (int): Route number

    Returns:
        None
    """
    write_line = f"{route_num}" + '\n'
    folder_path = f'./logs/{route_num}/googleORtools_failed.txt'
    log_line(write_line, folder_path, False)
    folder_path = f'./logs/googleORtools_failed.txt'
    log_line(write_line, folder_path, False)
    return None


def log_johnson(route_num: int) -> None:
    """
    This method logs information of Johnson's Method.

    Args:
        route_num (int): Route number

    Returns:
        None
    """
    write_line = f"{route_num}" + '\n'
    folder_path = f'./logs/{route_num}/johnsan_failed.txt'
    log_line(write_line, folder_path, False)
    folder_path = f'./logs/johnsan_failed.txt'
    log_line(write_line, folder_path, False)
    return None


def log_scalerization(route_num: int, bstsptw_tours: list, bstsp_tours: list, scalerization_start_stamp: float) -> None:
    """
    This method logs to a file information on a single iteration of the scalarization method.

    Args:
        route_num (int): Route number
        bstsptw_tours (list): BSTSPTW tours from scalerization
        bstsp_tours (list): BSTSP tours from scalerization
        scalerization_start_stamp (float): Start time of scalerization

    Returns:
        None
    """
    t = round(time.time() - scalerization_start_stamp)
    line = f"{route_num},{t},{len(bstsptw_tours) + len(bstsp_tours)},{len(bstsptw_tours)},{len(bstsp_tours)}\n"
    folder_path = f'./logs/{route_num}/scalerization.txt'
    log_line(line, folder_path, False)
    folder_path = f'./logs/scalerization.txt'
    log_line(line, folder_path, False)
    return None


def log_routes_with_no_tw(route_num: int) -> None:
    """
    This method logs information on routes with no time windows.

    Args:
        route_num (int): Route number

    Returns:
        None
    """
    line = f"{route_num}\n"
    folder_path = f'./logs/{route_num}/Routes_withNoTW.txt'
    log_line(line, folder_path, False)
    folder_path = f'./logs/Routes_withNoTW.txt'
    log_line(line, folder_path, False)
    return None


def log_ls_iteration(set_z: list, set_p: list, set_y: list, iteration_time: float, max_penalty_in_explore_set: float, common_arguments: dict) -> None:
    """
    This method logs to a file information on a single iteration of the Local Search method.

    Args:
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        set_p (list): List of tours that violate time-windows
        set_y (list): List of BSTSPTW feasible, non-Pareto-optimal tours
        iteration_time (float): Time taken for the iteration
        max_penalty_in_explore_set (float): Maximum penalty in the explore set
        common_arguments (dict): Common arguments for the local search

    Returns:
        None
    """
    ls_constants = common_arguments["ls_constants"]
    route_num = common_arguments["route_num"]
    ls_iteration_no = common_arguments["ls_iteration_no"]
    progress_file = common_arguments["progress_file"]

    line = ""
    if ls_iteration_no == 0:
        line += "Starting LS\n"
    nbd_flags = ls_constants["nbd_flags"]
    line += f"""
    Iteration: {ls_iteration_no}
        Z {len(set_z)}
        P: {len(set_p)}
        S3opt: {nbd_flags["S3opt"]}
        S3optTW: {nbd_flags["S3optTW"]}
        GapRepair: {nbd_flags["GapRepair"]}
        FixedPerm: {nbd_flags["FixedPerm"]}
        Quad: {nbd_flags["Quad"]}
        RandomPermute: {nbd_flags["RandomPermute"]}
        SimpleCycle: {nbd_flags["SimpleCycle"]}
        Time: {round(iteration_time, 1)}
        Max penalty in P: {max_penalty_in_explore_set}\n
    """

    write_line = f'{ls_iteration_no},{len(set_z)},{len(set_p)},{len(set_y)},{nbd_flags["S3opt"]},{nbd_flags["S3optTW"]},{nbd_flags["GapRepair"]},{nbd_flags["FixedPerm"]},{nbd_flags["Quad"]},{nbd_flags["RandomPermute"]},{nbd_flags["SimpleCycle"]},{round(iteration_time)},{max_penalty_in_explore_set}\n'
    filename = f"./logs/{route_num}/ls_iteration.txt"
    log_line(write_line, filename, False)
    log_line("Iteration Finished\n", progress_file, False)
    return None


def log_final_ls(operator_call_count: dict, nbd_tw_feasible_tours: dict, nbd_optimal_tours: dict, operator_times: dict, common_arguments: dict) -> None:
    """
    This method logs to a file results from the Local Search method.

    Args:
        operator_call_count (dict): Call counts of each operator
        nbd_tw_feasible_tours (dict): Dictionary containing time window feasible tours for each operator
        nbd_optimal_tours (dict): Dictionary containing optimal paths for each operator
        operator_times (dict): Times taken by each operator
        common_arguments (dict): Common arguments for the local search

    Returns:
        None
    """
    route_num = common_arguments["route_num"]
    ls_iteration_no = common_arguments["ls_iteration_no"]

    writeLine = f"S3opt,{len(nbd_tw_feasible_tours['S3opt'])},{len(nbd_optimal_tours['S3opt'])},{operator_call_count['S3opt']},{round(operator_times['S3opt'])}\n"
    writeLine += f"S3optTW,{len(nbd_tw_feasible_tours['S3optTW'])},{len(nbd_optimal_tours['S3optTW'])},{operator_call_count['S3optTW']},{round(operator_times['S3optTW'])}\n"
    writeLine += f"GR,{len(nbd_tw_feasible_tours['GapRepair'])},{len(nbd_optimal_tours['GapRepair'])},{operator_call_count['GapRepair']},{round(operator_times['GapRepair'])}\n"
    writeLine += f"FP,{len(nbd_tw_feasible_tours['FixedPerm'])},{len(nbd_optimal_tours['FixedPerm'])},{operator_call_count['FixedPerm']},{round(operator_times['FixedPerm'])}\n"
    writeLine += f"Quad,{len(nbd_tw_feasible_tours['Quad'])},{len(nbd_optimal_tours['Quad'])},{operator_call_count['Quad']},{round(operator_times['Quad'])}\n"
    writeLine += f"RP,{len(nbd_tw_feasible_tours['RandomPermute'])},{len(nbd_optimal_tours['RandomPermute'])},{operator_call_count['RandomPermute']},{round(operator_times['RandomPermute'])}\n"
    writeLine += f"CR,{len(nbd_tw_feasible_tours['SimpleCycle'])},{len(nbd_optimal_tours['SimpleCycle'])},{operator_call_count['SimpleCycle']},{round(operator_times['SimpleCycle'])}\n"
    writeLine += f"Iteration,{ls_iteration_no}\n"

    filename = f"./logs/{route_num}/ls_final.txt"
    log_line(writeLine, filename, False)

    return None


def get_scalarization_results(route_num: int) -> tuple:
    """
    This method gets the result from the scalarization method.

    Args:
        route_num (int): Route number

    Returns:
        bstsptw_tours (list): BSTSPTW tours from scalerization
        bstsp_tours (list): BSTSP tours from scalerization
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
    """
    folder_path = f'./logs/{route_num}'
    with open(f'{folder_path}/scalerization/bstsptw_tours_{route_num}.pkl', 'rb') as f:
        bstsptw_tours = pickle.load(f)
    with open(f'{folder_path}/scalerization/bstsp_tours_{route_num}.pkl', 'rb') as f:
        bstsp_tours = pickle.load(f)
    with open(f'{folder_path}/scalerization/weighted_paths_dict_{route_num}.pkl', 'rb') as f:
        weighted_paths_dict = pickle.load(f)
    # with open(f'{folder_path}/scalerization/quickest_paths_dict_{route_num}.pkl', 'rb') as f:
    #     quickest_paths_dict = pickle.load(f)
    return bstsptw_tours, bstsp_tours, weighted_paths_dict


def get_scalarization_results_single_route(route_num: int) -> tuple:
    """
    This method gets the scalarization results for a single operator (used in Operator Analysis).

    Args:
        route_num (int): Route number

    Returns:
        bstsptw_tours (list): BSTSPTW tours from scalerization
        bstsp_tours (list): BSTSP tours from scalerization
        weighted_paths_dict (dict): Dictionary of weighted paths between terminals
    """
    folder_path = f'./oneRoute/logs/{route_num}/S3opt/'
    with open(f'{folder_path}/scalerization/bstsptw_tours_{route_num}.pkl', 'rb') as f:
        bstsptw_tours = pickle.load(f)
    with open(f'{folder_path}/scalerization/bstsp_tours_{route_num}.pkl', 'rb') as f:
        bstsp_tours = pickle.load(f)
    with open(f'{folder_path}/scalerization/weighted_paths_dict_{route_num}.pkl', 'rb') as f:
        weighted_paths_dict = pickle.load(f)
    # with open(f'{folder_path}/scalerization/quickest_paths_dict_{route_num}.pkl', 'rb') as f:
    #     quickest_paths_dict = pickle.load(f)
    return bstsptw_tours, bstsp_tours, weighted_paths_dict


def save_ls_results(route_num: int, set_z: list, time_for_last_imp: float, move_counter: dict, total_ls_time: float, set_p: list) -> None:
    """
    This method saves the results obtained from applying the Local Search method.

    Args:
        route_num (int): Route number
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        time_for_last_imp (float): Time for last improvement in LS
        move_counter (dict): Dictionary of move counters
        total_ls_time (float): Total time taken for the local search
        set_p (list): List of tours that violate time-windows

    Returns:
        None
    """
    tmep = route_num, set_z, time_for_last_imp, move_counter, total_ls_time, set_p
    folder_path = f'./logs/{route_num}'
    with open(f'{folder_path}/ls_output_{route_num}.pkl', 'wb') as f:
        pickle.dump(tmep, f)
    return None


def get_results_ls(route_num: int) -> tuple:
    """
    This method gets the results obtained from applying the Local Search method.

    Args:
        route_num (int): Route number

    Returns:
        value (tuple): Results from the local search
    """
    folder_path = f'./logs/{route_num}'
    with open(f'{folder_path}/ls_output_{route_num}.pkl', 'rb') as f:
        tmep = pickle.load(f)
    return tmep
