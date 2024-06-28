"""
This file is used to run the benchmarking for the IP and LS algorithms
"""
import multiprocessing
from tqdm import tqdm

from milp.milp import *
from helpers.functions import write_ip_ls_stats, plot_ip_figures


def main(input_case: tuple[str, int]) -> None:
    """
    This is the main function used for solving a Bi-level Steiner TSP by first using a mixed integer program and then by using a local search algorithm.

    Args:
        input_case (tuple): instance_id, terminal_percent

    Returns:
        None
    """
    instance_id, terminal_percent = input_case
    route_id, instance_file = parse_route_id(instance_id, terminal_percent)
    clean_route_logs(route_id)
    L, energy_cost, turn_cost, time_cost_dict, time_windows, terminal_list, depot = generate_network_for_problem_instance(ip_constants, instance_file, terminal_percent)

    if L is None or len(terminal_list) < 2:
        return None

    # _, IP_first_point_time = get_mip_results(route_id)
    atlaest_one_point_found = mip_scalerization(L, energy_cost, turn_cost, time_cost_dict, time_windows, terminal_list, ip_constants, route_id, depot)
    ip_tours, ip_time = get_mip_results(route_id)

    # if atlaest_one_point_found:
    set_z, time_for_last_imp, len_bstsptw_tours, len_bstsp_tours = initiate_local_search(L, energy_cost, turn_cost, time_cost_dict, time_windows, terminal_list, depot, route_id, ls_constants)
    # route_num, set_z, time_for_last_imp, move_counter, total_ls_time, set_p = get_results_ls(route_id)
    write_ip_ls_stats(terminal_percent, route_id, L, ip_time, time_for_last_imp, len_bstsp_tours, len_bstsptw_tours, terminal_list, ip_tours, set_z)
    plot_ip_figures(ip_tours, set_z, instance_id, route_id, terminal_percent)
    return None


if __name__ == '__main__':
    # Test instances sorted by number of nodes
    # names = ['204.1', '206.4', '208.1', '203.3', '206.2', '208.3', '205.3', '207.1', '201.3', '202.1', '203.2', '204.2', '207.3', '207.2', '202.3', '202.4', '205.4', '208.2', '201.2', '201.4',
    #          '205.2', '204.3', '206.3', '201.1', '203.1', '202.2', '203.4', '205.1', '207.4', '206.1']
    # [(x, 30) for x in names]
    # instance_list = [('test', 10)]
    instance_list = [('204.1', 10), ('206.4', 10), ('208.1', 10), ('203.3', 10), ('206.2', 10), ('208.3', 10), ('204.1', 20), ('206.4', 20), ('208.1', 20), ('203.3', 20), ('206.2', 20), ('208.3', 20)]
    high_value = 1000

    ls_constants = {"allowed_time": 60 * 60 * 2,  # Total allowed time in LS in seconds
                    "cores": 4,  # Number of cores to use
                    "parallelize": True,  # Whether to parallelize the LS
                    "generate_log_file": False,  # Whether to generate a log file
                    "set_p_limit": 5000,  # Maximum number of tours in set P
                    "set_y_limit": 5000,  # Number of tours in set Y
                    "ls_sp_timeout_limit": 300,  # Timeout limit for shortest path computation
                    "or_factor": high_value,  # Factor for Google-OR tools as it handles only integers
                    "max_recursion_depth": 50,  # Maximum recursion depth in scalarization
                    "nbd_flags": {"S3opt": True, "S3optTW": True, "GapRepair": True, "FixedPerm": True, "Quad": True, "RandomPermute": True, "SimpleCycle": False},
                    # Flags for different operators. True means that the operator is used
                    "max_nb_runtime": 60 * 3,  # Maximum runtime for neighborhood operators
                    "fp_std_dev_factor": 0,  # Standard deviation factor for tour selection FPM operator
                    "fp_terminal_count": 12,  # Number of terminals in FPM operator
                    "fp_flip_count": 2,  # Maximum number of adjacent terminal flipped in FPM operator
                    "rm_path_count": high_value,  # Number of paths selected in RM operator
                    "simple_cycle_time": 0,  # Maximum time for each call of SimpleCycle operator is applied
                    "gr_terminal_count": high_value,  # Number of terminals in GR operator
                    "gr_flip_count": high_value,  # Maximum number of adjacent terminal flipped in GR operator
                    "gr_max_tries": 10,  # Maximum number of tries for GR operator
                    "fp_max_tries": 10,  # Maximum number of tries for FPM operator
                    "quad_max_tries": 100,  # Maximum number of tries for Quad operator
                    "rm_tries": 50,  # Number of times RandomPermute operator is applied
                    }

    ls_constants["allowed_init_time"] = ls_constants["allowed_time"] / 5  # Allowed time for initialization in seconds
    ls_constants["allowed_ls_time"] = ls_constants["allowed_time"] - ls_constants["allowed_init_time"]  # Allowed time for LS in seconds

    ip_constants = {"seed": 10,  # Seed for random number generator
                    "lower_bound_energy": 0,  # Lower bound for energy used while generating random instances
                    "upper_bound_energy": 10,  # Upper bound for energy used while generating random instances
                    "revisits": 4,  # Number of revisits for each terminal
                    "allowed_time": 60 * 60 * 4,  # Total allowed time in IP in seconds
                    "max_recursion_depth": 50  # Maximum recursion depth in scalarization
                    }

    clean_mip_logs()

    if ls_constants["generate_log_file"]:
        if ls_constants["parallelize"]:
            log_file_name = f"console_p_{ls_constants['cores']}_{ls_constants['ls_allowed_init_time']}_{ls_constants['ls_allowed_ls_time']}"
        else:
            log_file_name = f"console_s_{0}_{ls_constants['ls_allowed_init_time']}_{ls_constants['ls_allowed_ls_time']}"
        sys.stdout = open(f'./logs/{log_file_name}.txt', 'w')
    if ls_constants["generate_log_file"]: sys.stdout.close()

    if ls_constants["parallelize"]:
        with multiprocessing.Pool(processes=ls_constants["cores"]) as pool:
            pool.map(main, instance_list)
    else:
        for input_case in tqdm(instance_list):
            main(input_case)
