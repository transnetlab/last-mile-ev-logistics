"""
This file is used to run the local search algorithm one operator at a time
"""
import multiprocessing

from helpers.loggers import *
from helpers.functions import *
from localSearch.lsMain import call_local_search
from localSearch.lsInitial import scalerization
from helpers.graphBuilder import get_route_information


def main(route_num: int) -> None:
    """
    This function is the main function for solving a Bi-level Steiner TSP using a local search algorithm.

    Args:
        route_num (int): Route number

    Returns:
        None
    """
    clean_route_logs(route_num)
    (G, L, terminal_set, time_window_dict, time_cost_dict, energy_dual, direction_dual, mean_energy_dual, mean_turn_dual, tw_term_count,
     stdev_energy_dual, stdev_turn_dual, depot) = get_route_information(route_num)

    try:
        bstsptw_tours, bstsp_tours, weighted_paths_dict = get_scalarization_results(route_num)
    except:
        scalerization(energy_dual, direction_dual, terminal_set, L, ls_constants, route_num, time_cost_dict, time_window_dict, depot)
        bstsptw_tours, bstsp_tours, weighted_paths_dict = get_scalarization_results(route_num)

    set_z, time_for_last_imp, parents, total_ls_time, set_p, ls_iteration_no = call_local_search(bstsptw_tours, bstsp_tours, terminal_set, L, energy_dual, direction_dual, time_cost_dict,
                                                                                                 time_window_dict, mean_energy_dual, mean_turn_dual, weighted_paths_dict, stdev_energy_dual,
                                                                                                 stdev_turn_dual, route_num, depot, ls_constants)
    # (route_num, set_z, time_for_last_imp, move_counter, total_ls_time, set_p) = get_results_ls(route_num)
    test_solutions(set_z, terminal_set, time_cost_dict, time_window_dict, depot, route_num)

    final_log_file(G, L, terminal_set, bstsptw_tours, bstsp_tours, set_z, parents, route_num, time_for_last_imp, ls_constants, tw_term_count, total_ls_time, set_p, 0, ls_iteration_no)
    return None


def call_main(route_num: int) -> None:
    """
    Update the ls_constants and calls the main function

    Args:
        route_num (int): Route number

    Returns:
        None
    """
    try:
        shutil.rmtree(f"./logs/oneRoute/{route_num}")
    except:
        pass
    os.makedirs(f"./logs/oneRoute/{route_num}", exist_ok=True)
    all_operators = ["S3opt", "S3optTW", "GapRepair", "FixedPerm", "Quad", "RandomPermute", "SimpleCycle", "all"]
    for operator in all_operators:
        if operator == "all":
            for op in all_operators:
                ls_constants["nbd_flags"][op] = True
        else:
            for op in all_operators:
                ls_constants["nbd_flags"][op] = False
            ls_constants["nbd_flags"][operator] = True

        clean_summary_logs()
        main(route_num)

        source_folder = f"./logs/{route_num}"
        os.makedirs(f"./logs/oneRoute/{route_num}/{operator}", exist_ok=True)
        destination_folder = f"./logs/oneRoute/{route_num}/{operator}"
        copy_folder(source_folder, destination_folder)
    return None


if __name__ == '__main__':

    # ls_constants is a dictionary that contains the parameters for the local search algorithm
    ls_constants = {"allowed_time": 60 * 60 * 2,  # Total allowed time in seconds
                    "cores": 30,  # Number of cores to use
                    "parallelize": True,  # Whether to parallelize the local search
                    "generate_log_file": False,  # Whether to generate a log file
                    "set_p_limit": 100,  # Maximum number of tours in set P
                    "set_y_limit": 100,  # Number of tours in set Y
                    "ls_sp_timeout_limit": 30,  # Timeout limit for shortest path computation
                    "or_factor": 1000,  # Factor for Google-OR tools as it handles only integers
                    "max_recursion_depth": 50,  # Maximum recursion depth in scalarization
                    "nbd_flags": {"S3opt": True, "S3optTW": True, "GapRepair": True, "FixedPerm": True, "Quad": True, "RandomPermute": True, "SimpleCycle": True},
                    # Flags for different operators. True means that the operator is used
                    "max_nb_runtime": 60 * 5,  # Maximum runtime for neighborhood operators
                    "fp_std_dev_factor": 0.1,  # Standard deviation factor for tour selection FPM operator
                    "fp_terminal_count": 8,  # Number of terminals adjacent terminals analysed in FPM operator
                    "fp_flip_count": 2,  # Maximum number of adjacent terminal flipped in FPM operator
                    "rm_path_count": 1,  # Number of paths selected in RM operator
                    "simple_cycle_time": 10,  # Maximum time for each call of SimpleCycle operator is applied
                    "gr_terminal_count": 8,  # Number of terminals adjacent terminals analysed in GR operator
                    "gr_flip_count": 3,  # Maximum number of adjacent terminal flipped in GR operator
                    "rm_tries": 1,  # Number of times RandomPermute operator is applied
                    "gr_max_tries": 1,  # Maximum number of tries for GR operator
                    "fp_max_tries": 1,  # Maximum number of tries for FPM operator
                    "quad_max_tries": 1  # Maximum number of tries for Quad operator
                    }
    ls_constants["allowed_init_time"] = ls_constants["allowed_time"] / 5  # Allowed time for initialization in seconds
    ls_constants["allowed_ls_time"] = ls_constants["allowed_time"] - ls_constants["allowed_init_time"]  # Allowed time for LS in seconds

    print_experiment_details(ls_constants)
    all_routes = pd.read_csv(f"./lsInputs/working_amazon_routes.csv").sort_values(by=['TWTerminals'], ascending=False)
    all_routes = all_routes.Route_num.tolist()
    if ls_constants["parallelize"]:
        with multiprocessing.Pool(processes=ls_constants["cores"]) as pool:
            pool.map(call_main, all_routes)
    else:
        for route_num in all_routes:
            call_main(route_num)
