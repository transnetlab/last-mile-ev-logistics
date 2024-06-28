import sys
import math
import cplex
import random
import statistics
import numpy as np
import networkx as nx
from collections import defaultdict

from helpers.loggers import *
from localSearch.lsMain import call_local_search
from localSearch.lsInitial import scalerization
from helpers.shortestPath import biobj_label_correcting
from helpers.functions import get_pareto_set, test_solutions


def parse_route_id(route_id: str, terminal_percent: int) -> tuple:
    """
    This function takes in the problem instance Id and returns the route Id and the name of the instance file.

    Args:
        route_id (str): instance_id
        terminal_percent (int): percentage of terminals

    Returns:
        route_id (str): route_id
        instance_file (str): instance_file
    """
    instance_file = f'./milpInputs/rc_{route_id}.txt'

    route_id = instance_file.split('_')[-1].split('.txt')[0]
    route_id = route_id.replace('.', '_')
    route_id = str(terminal_percent) + "_" + route_id
    return route_id, instance_file


def convert_to_nested_dict(tuple_dict: dict) -> dict:
    """
    This function converts a dictionary of tuples into a nested dictionary.

    Args:
        tuple_dict (dict): dictionary of tuples

    Returns:
        nested_dict (dict): nested dictionary
    """
    nested_dict = defaultdict(dict)
    for (vi, vj), value in tuple_dict.items():
        nested_dict[vi][vj] = value
    return nested_dict


def get_time_left(max_time: int, start_stamp: float) -> bool:
    """
    This function checks whether time is left for further processing.

    Args:
        max_time (int): maximum time
        start_stamp (float): start time

    Returns:
        value (bool): True if time is left, False otherwise
    """
    return time.time() - start_stamp < max_time


def get_test_network_from_paper() -> tuple:
    """
    This function gets information on the test network described in the paper.

    Returns:
        L (networkx.classes.multidigraph.MultiDiGraph): Dual graph
        energy_cost (dict): Energy cost dict
        turn_cost (dict): Turn cost dict
        time_cost_dict (dict): Dictionary of travel times
        time_windows (dict): Time windows dict
        dual_terminals (list): Terminal nodes
        depot (tuple): Depot node
    """
    L = nx.DiGraph()
    edges = [(0, 1), (1, 2), (1, 10), (1, 9), (9, 0), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0), (7, 5), (8, 7), (3, 8), (8, 1), (0, 7), (10, 11), (11, 3)]
    turn_energy_list = [(0, 2), (1, -1), (0, 2), (0, 3), (0, 1), (0, 2), (1, 4), (1, 4), (0, 2), (1, 1), (1, -1), (1, 2), (1, -1), (1, 1), (1, 2), (0, 2), (1, 2)]
    L.add_edges_from(edges)
    dual_terminals = [0, 3, 5]
    turn_cost = {x: y[0] for x, y in zip(edges, turn_energy_list)}
    energy_cost = {x: y[1] for x, y in zip(edges, turn_energy_list)}
    time_cost_dict = {x: 10 for x in edges}
    depot = 0
    time_windows = {}
    for v in L.nodes():
        time_windows[v] = (0, 10000000)
    time_windows.update({0: (0, 10000000), 3: (30, 120), 5: (0, 50)})
    return L, energy_cost, turn_cost, time_cost_dict, time_windows, dual_terminals, depot

def get_bounds_on_departure_time(L: networkx.classes.multidigraph.MultiDiGraph, time_cost_dict: dict, time_windows: dict, energy_cost: dict, turn_cost: dict, terminal_list: list, depot: int) -> dict:
    """
    Get lower and bounds on departure time from every node. Lower bound is defined using shortest path from depot to node.
    Upper Bound is same for every node and is defined as the longest path from all the pareto-optimal from all terminals to depot.

    Args:
        L (networkx.classes.multidigraph.MultiDiGraph): Dual graph
        time_cost_dict (dict): Dictionary of travel times
        time_windows (dict): Time windows dict
        energy_cost (dict): Energy cost dict
        turn_cost (dict): Turn cost dict
        terminal_list (list): Terminal nodes
        depot (int): Depot node

    Returns:
        departure_time_bounds (dict): Lower and upper bounds on departure time
    """
    def weight_weighted(u, v, d):
        return time_cost_dict[u, v]

    longest_path = 0
    energy_dual = convert_to_nested_dict(energy_cost)
    direction_dual = convert_to_nested_dict(turn_cost)
    combined_dict = {i: {j: (energy_dual[i][j], direction_dual[i][j]) for j in L.neighbors(i)} for i in L.nodes()}
    for terminal in terminal_list:
        set_of_paths, _ = biobj_label_correcting(combined_dict, terminal, depot)
        path_durations = [sum(time_cost_dict[path[i], path[i + 1]] for i in range(len(path) - 1)) for path in set_of_paths]
        longest_path = max(longest_path, max(path_durations))
    last_departure_time = max(time_windows[terminal][1] for terminal in set(terminal_list) - {depot})
    upper_bound = last_departure_time + longest_path
    departure_time_bounds = {terminal: (math.floor(nx.shortest_path_length(L, depot, terminal, weight=weight_weighted)), math.ceil(upper_bound)) for terminal in L.nodes()}
    return departure_time_bounds

def generate_network_for_problem_instance(ip_constants: dict, instance_file: str, terminal_percent: int) -> tuple:
    """
    Create MIP instance network

    Args:
        ip_constants (dict): dictionary of constants
        instance_file (str): path to IP instance
        terminal_percent (int): percentage of terminals

    Returns:
        L (networkx.classes.multidigraph.MultiDiGraph): Dual graph
        energy_cost (dict): Energy cost dict
        turn_cost (dict): Turn cost dict
        time_cost_dict (dict): Dictionary of travel times
        time_windows (dict): Time windows dict
        dual_terminals (list): Terminal nodes
        depot (tuple): Depot node
    """
    # terminal_percent,  seed = 20, 42
    if "test" in instance_file:
        return get_test_network_from_paper()
    seed = ip_constants["seed"]
    random.seed(seed)
    np.random.seed(seed)

    time_cost_matrix, arrival, depart = get_instance_data(instance_file)

    num_nodes = len(arrival)

    # Create a directed graph
    L = nx.DiGraph()
    L.add_nodes_from(range(num_nodes))
    for source in range(num_nodes):
        for target in range(num_nodes):
            if source != target:
                L.add_edge(source, target)

    n_t = round(terminal_percent * num_nodes / 100) - 1
    if n_t <= 0:
        return None, None, None, None, None, None, None

    energy_cost = {e: np.random.uniform(ip_constants["lower_bound_energy"], ip_constants["upper_bound_energy"]) for e in L.edges}
    turn_cost = {e: np.random.choice([0, 1]) for e in L.edges}

    depot = 0
    dual_vertices = list(range(num_nodes))
    temp_nodes = [x for x in dual_vertices if x != depot]
    dual_terminals = [depot] + list(random.sample(temp_nodes, n_t - 1))

    time_windows = {}
    for v in dual_vertices:
        if v in dual_terminals:
            time_windows[v] = (arrival[v], depart[v])
        else:
            time_windows[v] = (0, 10000000)
    time_windows[depot] = (0, 10000000)

    time_cost_dict = {}
    for vi in dual_vertices:
        for vj in dual_vertices:
            if vi != vj:
                time_cost_dict[(vi, vj)] = time_cost_matrix[vi][vj]
    return L, energy_cost, turn_cost, time_cost_dict, time_windows, dual_terminals, depot


def get_instance_data(file_path: str) -> tuple:
    """
    This function reads data from an MIP problem instance file.

    Args:
        file_path (str): path to IP instance

    Returns:
        time_cost_matrix (list): time cost matrix
        arrival (list): arrival times
        depart (list): depart times
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    n = int(lines[0].strip())  # Number of nodes
    time_cost_matrix = []
    for i in range(1, n + 1):
        row = list(map(float, lines[i].split()))
        time_cost_matrix.append(row)

    arrival = []
    depart = []
    for i in range(n + 1, n * 2 + 1):
        window = list(map(int, lines[i].split()))
        arrival.append(window[0])
        depart.append(window[1])

    return time_cost_matrix, arrival, depart


def initiate_local_search(L: networkx.classes.multidigraph.MultiDiGraph, energy_cost: dict, turn_cost: dict, time_cost_dict: dict, time_windows: dict, terminal_list: list, depot: tuple,
                          route_num: int, ls_constants: dict):
    """
    This function calls the local search algorithm for solving a Bi-level Steiner TSP with time windows.

    Args:
        L (networkx.classes.multidigraph.MultiDiGraph): Dual graph
        energy_cost (dict): Energy cost dict
        turn_cost (dict): Turn cost dict
        time_cost_dict (dict): Dictionary of travel times
        time_windows (dict): Time windows dict
        terminal_list (list): Terminal nodes
        depot (tuple): Depot node
        route_num (int): Route number
        ls_constants (dict): Constants for the local search

    Returns:
        set_z (list): List of BSTSPTW, Pareto-optimal tours
        time_for_last_imp (float): Time for last improvement in LS
        value (int): Count of BSTSPTW tours
        value (int): Count of BSTSP tours
    """
    # route_num = route_id
    print(f"Running LS for route {route_num}")
    energy_dual = convert_to_nested_dict(energy_cost)
    direction_dual = convert_to_nested_dict(turn_cost)
    time_cost_dict = convert_to_nested_dict(time_cost_dict)
    depot_index = terminal_list.index(depot)
    # terminal_set = cyclize(terminal_list, depot_index)
    terminal_set = set(terminal_list)

    edge_energies = [energy_dual[i][j] for i in energy_dual.keys() for j in energy_dual[i].keys()]
    all_turns = [direction_dual[i][j] for i in direction_dual.keys() for j in direction_dual[i].keys()]
    mean_energy_dual = statistics.mean(edge_energies)
    mean_turn_dual = statistics.mean(all_turns)
    stdev_energy_dual = statistics.stdev(edge_energies)
    stdev_turn_dual = statistics.stdev(all_turns)

    tw_term_count = 0

    revised_time_windows = {}
    for nodes in time_windows.keys():
        if time_windows[nodes][1] < 1000000 or nodes == depot:
            tw_term_count += 1
            revised_time_windows[nodes] = time_windows[nodes]
    time_window_dict = revised_time_windows

    if ls_constants["generate_log_file"]:
        if ls_constants["parallelize"]:
            log_file_name = f"console_p_{ls_constants['cores']}_{ls_constants['allowed_init_time']}_{ls_constants['allowed_ls_time']}"
        else:
            log_file_name = f"console_s_{0}_{ls_constants['allowed_init_time']}_{ls_constants['allowed_ls_time']}"
        sys.stdout = open(f'./logs/{log_file_name}.txt', 'w')

    clean_summary_logs()
    scalerization(energy_dual, direction_dual, terminal_set, L, ls_constants, route_num, time_cost_dict, time_window_dict, depot)
    bstsptw_tours, bstsp_tours, weighted_paths_dict = get_scalarization_results(route_num)
    set_z, time_for_last_imp, parents, total_ls_time, set_p, ls_iteration_no = call_local_search(bstsptw_tours, bstsp_tours, terminal_set, L, energy_dual, direction_dual, time_cost_dict,
                                                                                                 time_window_dict, mean_energy_dual, mean_turn_dual, weighted_paths_dict, stdev_energy_dual,
                                                                                                 stdev_turn_dual, route_num, depot, ls_constants)
    # (route_num, set_z, time_for_last_imp, move_counter, total_ls_time, set_p) = get_results_ls(route_num)
    test_solutions(set_z, terminal_set, time_cost_dict, time_window_dict, depot, route_num)
    final_log_file(L, L, terminal_set, bstsptw_tours, bstsp_tours, set_z, parents, route_num, time_for_last_imp, ls_constants, tw_term_count, total_ls_time, set_p, 0, ls_iteration_no)
    return set_z, time_for_last_imp, len(bstsptw_tours), len(bstsp_tours)


def mip_main(alpha: float, beta: float, L: networkx.classes.multidigraph.MultiDiGraph, energy_cost: dict, turn_cost: dict, time_cost_dict: dict, time_windows: dict, terminal_list: list, revisits: int,
             progress_file: str, time_remaining: float, departure_time_bounds: dict) -> tuple:
    """
    This function is used for solving a Bi-level Steiner TSP with time windows using a mixed integer program.

    Args:
        alpha (float): Direction weight
        beta (float): Energy weight
        L (networkx.classes.multidigraph.MultiDiGraph): Dual graph
        energy_cost (dict): Energy cost dict
        turn_cost (dict): Turn cost dict
        time_cost_dict (dict): Dictionary of travel times
        time_windows (dict): Time windows dict
        terminal_list (list): Terminal nodes
        revisits (int): Number of revisits
        progress_file (str): File to write progress to
        time_remaining (float): Time left
        departure_time_bounds (dict): Lower and upper bounds on departure time

    Returns:
        list (list): energy, turn and path
        status (int): CPLEX status
    """
    # alpha, beta = 0.000001, 1
    n_t = len(terminal_list)

    objective_fun_binary = defaultdict(list)
    objective_fun_integer = defaultdict(list)
    objective_fun_continuous = defaultdict(list)
    less_than_equal_to_constraints = defaultdict(list)
    equal_to_constraints = defaultdict(list)

    # Adding x_u_i_v_j to objective function as binary variable
    for u, v in L.edges:
        coeff = beta * energy_cost[(u, v)] + alpha * turn_cost[(u, v)]
        for i in range(revisits):
            for j in range(revisits):
                objective_fun_binary['coefficient'].append(coeff)
                objective_fun_binary['variable'].append('x_{}_{}_{}_{}'.format(u, i, v, j))
                objective_fun_binary['type'].append('B')

    # Adding f_{u_i_v_j} to objective function as integer variable
    for u, v in L.edges:
        coeff = 0
        for i in range(revisits):
            for j in range(revisits):
                objective_fun_integer['coefficient'].append(coeff)
                objective_fun_integer['variable'].append('f_{}_{}_{}_{}'.format(u, i, v, j))
                objective_fun_integer['lower_bound'].append(0)
                objective_fun_integer['upper_bound'].append(n_t - 1)

    # Adding y_{u_i} to objective function as binary variable
    for u in terminal_list:
        coeff = 0
        for i in range(revisits):
            objective_fun_binary['coefficient'].append(coeff)
            objective_fun_binary['variable'].append('y_{}_{}'.format(u, i))
            objective_fun_binary['type'].append('B')

    # Adding s_{u_i} to objective function as continuous variable
    for v in L.nodes:
        for i in range(revisits):
            coeff = 0
            objective_fun_continuous['coefficient'].append(coeff)
            objective_fun_continuous['variable'].append('s_{}_{}'.format(v, i))
            objective_fun_continuous['lower_bound'].append(departure_time_bounds[v][0])
            objective_fun_continuous['upper_bound'].append(departure_time_bounds[v][1])

    # Adding flow balance at every node and every copy (constraint 2)
    for v in L.nodes:
        for i in range(revisits):
            constraint_name = [f"Flow_balance_{v}_{i}"]
            constraint = [[[], []]]
            rhs = [0.0]
            for u in L.predecessors(v):
                for j in range(revisits):
                    constraint[0][0].append('x_{}_{}_{}_{}'.format(u, j, v, i))
                    constraint[0][1].append(1.0)

            for w in L.successors(v):
                for k in range(revisits):
                    constraint[0][0].append('x_{}_{}_{}_{}'.format(v, i, w, k))
                    constraint[0][1].append(-1.0)

            equal_to_constraints['constraint_name'].extend(constraint_name)
            equal_to_constraints['constraint'].extend(constraint)
            equal_to_constraints['rhs'].extend(rhs)

    # Adding constraint one delivery at one terminal across all copies (constraint 3)
    non_depot_terminals = set(terminal_list) - {0}
    for v in non_depot_terminals:
        constraint_name = [f"Terminal_visit_once_{v}"]
        constraint = [[[], []]]
        rhs = [1.0]
        for j in range(revisits):

            for u in L.predecessors(v):
                for i in range(revisits):
                    constraint[0][0].append('f_{}_{}_{}_{}'.format(u, i, v, j))
                    constraint[0][1].append(1.0)

            for w in L.successors(v):
                for k in range(revisits):
                    constraint[0][0].append('f_{}_{}_{}_{}'.format(v, j, w, k))
                    constraint[0][1].append(-1.0)

        equal_to_constraints['constraint_name'].extend(constraint_name)
        equal_to_constraints['constraint'].extend(constraint)
        equal_to_constraints['rhs'].extend(rhs)

    # Adding no delivery at depot and other nodes (constraint 4)
    for v in L.nodes:
        if v not in terminal_list:
            constraint_name = [f"No_delivery_{v}"]
            constraint = [[[], []]]
            rhs = [0.0]
            for i in range(revisits):

                for u in L.predecessors(v):
                    for j in range(revisits):
                        constraint[0][0].append('f_{}_{}_{}_{}'.format(u, j, v, i))
                        constraint[0][1].append(1.0)

                for w in L.successors(v):
                    for k in range(revisits):
                        constraint[0][0].append('f_{}_{}_{}_{}'.format(v, i, w, k))
                        constraint[0][1].append(-1.0)

            equal_to_constraints['constraint_name'].extend(constraint_name)
            equal_to_constraints['constraint'].extend(constraint)
            equal_to_constraints['rhs'].extend(rhs)

    # Adding the upper bound of the flow variable (constraint 5)
    for u, v in L.edges:
        for i in range(revisits):
            for j in range(revisits):
                constraint_name = [f"Flow_upper_bound_{u}_{i}_{v}_{j}"]
                constraint = [
                    [['f_{}_{}_{}_{}'.format(u, i, v, j), 'x_{}_{}_{}_{}'.format(u, i, v, j)], [1.0, -1 * (n_t - 1)]]]
                rhs = [0]
                less_than_equal_to_constraints['constraint_name'].extend(constraint_name)
                less_than_equal_to_constraints['constraint'].extend(constraint)
                less_than_equal_to_constraints['rhs'].extend(rhs)

    # Adding the Nodal arrival times constraints (constraint 6)
    for u, v in L.edges:
        for i in range(revisits):
            for j in range(revisits):
                if v != 0 or j != 0:  # j starts from 0 in formulation it is 1
                    M_local = departure_time_bounds[u][1] + time_cost_dict[(u, v)] - departure_time_bounds[v][0]
                    constraint_name = [f"Nodal_arrival_{u}_{i}_{v}_{j}"]
                    constraint = [[['s_{}_{}'.format(u, i), 's_{}_{}'.format(v, j), 'x_{}_{}_{}_{}'.format(u, i, v, j)],
                                   [1.0, -1.0, M_local]]]
                    rhs = [-1 * time_cost_dict[(u, v)] + M_local]
                    less_than_equal_to_constraints['constraint_name'].extend(constraint_name)
                    less_than_equal_to_constraints['constraint'].extend(constraint)
                    less_than_equal_to_constraints['rhs'].extend(rhs)
    # (constraint 7)
    constraint_name = [f"DepotRevisit_{0}"]
    constraint = [[[], []]]
    rhs = [1.0]
    for u, v in L.edges:
        if u == 0:
            for j in range(revisits):
                constraint[0][0].append('x_{}_{}_{}_{}'.format(0, 0, v, j))
                constraint[0][1].append(1.0)
    equal_to_constraints['constraint_name'].extend(constraint_name)
    equal_to_constraints['constraint'].extend(constraint)
    equal_to_constraints['rhs'].extend(rhs)

    # Adding the time windows constraints (constraint 8)
    for v in terminal_list:
        for i in range(revisits):
            constraint_name = [f"upper_bound_y_{v}_{i}"]
            constraint = [[['y_{}_{}'.format(v, i)], [1.0]]]
            rhs = [0.0]
            for w in L.successors(v):
                for j in range(revisits):
                    constraint[0][0].append('x_{}_{}_{}_{}'.format(v, i, w, j))
                    constraint[0][1].append(-1.0)
            less_than_equal_to_constraints['constraint_name'].extend(constraint_name)
            less_than_equal_to_constraints['constraint'].extend(constraint)
            less_than_equal_to_constraints['rhs'].extend(rhs)

    # Adding the time windows constraints (constraint 9)
    for v in terminal_list:
        constraint_name = [f"Time_window_{v}"]
        constraint = [[[], []]]
        rhs = [-1.0]
        for i in range(revisits):
            constraint[0][0].append('y_{}_{}'.format(v, i))
            constraint[0][1].append(-1.0)
        less_than_equal_to_constraints['constraint_name'].extend(constraint_name)
        less_than_equal_to_constraints['constraint'].extend(constraint)
        less_than_equal_to_constraints['rhs'].extend(rhs)

    # constraint 10
    for v in terminal_list:
        for i in range(revisits):
            M1_lhs = time_windows[v][0] - departure_time_bounds[v][0]
            M2_rhs = departure_time_bounds[v][1] - time_windows[v][1]
            constraint_name = [f"lower_bound_s_{v}_{i}", f"upper_bound_s_{v}_{i}"]
            constraint = [[['y_{}_{}'.format(v, i), 's_{}_{}'.format(v, i)], [M1_lhs, -1]],
                          [['y_{}_{}'.format(v, i), 's_{}_{}'.format(v, i)], [M2_rhs, 1]]]
            rhs = [M1_lhs - time_windows[v][0], time_windows[v][1] + M2_rhs]
            less_than_equal_to_constraints['constraint_name'].extend(constraint_name)
            less_than_equal_to_constraints['constraint'].extend(constraint)
            less_than_equal_to_constraints['rhs'].extend(rhs)

    # constraint 11
    # lazy_constraints = defaultdict(list)
    # for u in L.nodes:
    #     if u == 0: continue
    #     for i in range(revisits - 1):
    #         constraint_name = [f"Ordering_{u}_{i}"]
    #         constraint = [[[], []]]
    #         rhs = [0.0]
    #         for v in L.successors(u):
    #             for j in range(revisits):
    #                 constraint[0][0].append('x_{}_{}_{}_{}'.format(u, i, v, j))
    #                 constraint[0][1].append(1.0)
    #                 constraint[0][0].append('x_{}_{}_{}_{}'.format(u, i + 1, v, j))
    #                 constraint[0][1].append(-1.0)
    #         lazy_constraints['constraint_name'].extend(constraint_name)
    #         lazy_constraints['constraint'].extend(constraint)
    #         lazy_constraints['rhs'].extend(rhs)

    # creating model
    model = cplex.Cplex()
    model.objective.set_sense(model.objective.sense.minimize)

    # Adding variables to objective function
    for i in range(len(objective_fun_binary['coefficient'])):
        model.variables.add(obj=[objective_fun_binary['coefficient'][i]], names=[objective_fun_binary['variable'][i]], types=objective_fun_binary['type'][i])

    # print("Adding integer variables...")
    for i in range(len(objective_fun_integer['coefficient'])):
        model.variables.add(obj=[objective_fun_integer['coefficient'][i]], names=[objective_fun_integer['variable'][i]],
                            lb=[objective_fun_integer['lower_bound'][i]], ub=[objective_fun_integer['upper_bound'][i]], types=['I'])

    # print("Adding continuous variables...")
    for i in range(len(objective_fun_continuous['coefficient'])):
        model.variables.add(obj=[objective_fun_continuous['coefficient'][i]], names=[objective_fun_continuous['variable'][i]],
                            lb=[objective_fun_continuous['lower_bound'][i]], ub=[objective_fun_continuous['upper_bound'][i]], types=['I'])

    # Adding constraints to model
    for i in range(len(equal_to_constraints['constraint'])):
        model.linear_constraints.add(lin_expr=[equal_to_constraints['constraint'][i]], senses=['E'],
                                     rhs=[equal_to_constraints['rhs'][i]], names=[equal_to_constraints['constraint_name'][i]])
    for i in range(len(less_than_equal_to_constraints['constraint'])):
        model.linear_constraints.add(lin_expr=[less_than_equal_to_constraints['constraint'][i]], senses=["L"],
                                     rhs=[less_than_equal_to_constraints['rhs'][i]], names=[less_than_equal_to_constraints['constraint_name'][i]])
    # for i in range(len(lazy_constraints['constraint'])):
    #     model.linear_constraints.advanced.add_lazy_constraints(lin_expr=[lazy_constraints['constraint'][i]], senses=["L"],
    #                                  rhs=[lazy_constraints['rhs'][i]], names=[lazy_constraints['constraint_name'][i]])

    # model.write(f'model_with_{alpha}_{beta}.lp')
    # Turn solve log off
    model.set_log_stream(None)
    model.set_results_stream(None)
    model.set_warning_stream(None)
    model.set_error_stream(None)
    model.parameters.mip.display.set(0)
    model.parameters.timelimit.set(time_remaining)
    model.solve()

    try:
        model.solution.get_objective_value()
    except cplex.exceptions.errors.CplexSolverError as e:
        return ([None, None, None], 0)

    # model.solution.write(f'solution_with_{alpha}_{beta}.sol')

    # Get the values of x that is 1
    active_arcs = [(u, i, v, j)
                   for u, v in L.edges
                   for i in range(revisits)
                   for j in range(revisits)
                   if model.solution.get_values('x_{}_{}_{}_{}'.format(u, i, v, j)) > 0.99]
    # Example: active_arcs = [(0, 2, 1, 0), (0, 0, 7, 2), (1, 0, 2, 0), (2, 0, 3, 2), (3, 2, 8, 2), (5, 0, 6, 2), (5, 2, 6, 0), (6, 0, 0, 2), (6, 2, 0, 0), (7, 0, 5, 0), (7, 2, 5, 2), (8, 2, 7, 0)]

    for i, arcs in enumerate(active_arcs):
        arc = active_arcs[i]
        active_arcs[i] = ((arc[0], arc[1]), (arc[2], arc[3]))
    # Example: active_arcs = [((0, 2), (1, 0)), ((0, 0), (7, 2)), ((1, 0), (2, 0)), ((2, 0), (3, 2)), ((3, 2), (8, 2)), ((5, 0), (6, 2)), ((5, 2), (6, 0)), ((6, 0), (0, 2)), ((6, 2), (0, 0)), ((7, 0), (5, 0)), ((7, 2), (5, 2)), ((8, 2), (7, 0))]
    path = [active_arcs[0][0], active_arcs[0][1]]
    active_arcs_copy = active_arcs.copy()  # Debugging purposes only
    active_arcs.remove(active_arcs[0])

    while len(active_arcs):
        next_edge_found = False
        for arcs in active_arcs:
            if path[-1] == arcs[0]:
                next_edge_found = True
                break
        if not next_edge_found:
            raise ValueError(
                f"Error: Adjacent edge not present in optimal solution of IP.\nalpha={alpha}\nbeta={beta}\npath={path}\nremaining_active_arcs={active_arcs}\nactive_arcs_copy={active_arcs_copy}")
            # return "Error: Adjacent edge not present in optimal solution of IP"
        path.append(arcs[1])
        active_arcs.remove(arcs)
    for i, _ in enumerate(path):
        path[i] = path[i][0]

    energy, turns = 0, 0
    for i, _ in enumerate(path):
        if i != len(path) - 1:
            energy += energy_cost[path[i], path[i + 1]]
            turns += turn_cost[path[i], path[i + 1]]

    # print("Objectives:", turns, energy)
    turns, energy = round(turns, 2), round(energy, 2)
    # log_line(f" Exit \n", progress_file, False)
    return [energy, turns, path], model.solution.get_status()


def mip_scalerization(L: networkx.classes.multidigraph.MultiDiGraph, energy_cost: dict, turn_cost: dict, time_cost_dict: dict, time_windows: dict, terminal_list: list, ip_constants: dict,
                      route_id: str, depot: int) -> bool:
    """
    This function gives the efficiency frontier by using an MIP-based scalarization approach, in which the points on the efficieny frontier are obtained by solving a sequence of mixed integer programs.

    Args:
        L (networkx.classes.multidigraph.MultiDiGraph): Dual graph
        energy_cost (dict): Energy cost dict
        turn_cost (dict): Turn cost dict
        time_cost_dict (dict): Dictionary of travel times
        time_windows (dict): Time windows dict
        terminal_list (list): Terminal nodes
        ip_constants (dict): Dictionary of IP constants
        route_id (str): Route id
        depot (int): Depot node

    Returns:
        value (bool): True if at least one point found, False otherwise
    """
    print(f"Running IP for Route {route_id}")
    departure_time_bounds = get_bounds_on_departure_time(L, time_cost_dict, time_windows, energy_cost, turn_cost, terminal_list, depot)
    # print(f"Departure Time bounds for Instance {route_id}:")
    # print(departure_time_bounds)

    mip_start_stamp = time.time()
    mip_tours = []
    energy_1, energy_2, turns_1, turns_2 = None, None, None, None
    almost_zero = 0.000001
    atlaest_one_point_found = False
    revisits = min(len(terminal_list), ip_constants["revisits"])
    filename = f'./logs/IPscalerizationEndConditions.txt'

    progress_file = f'./logs/{route_id}/progressTrackIP.txt'
    write_line = f"Revisit: {revisits}\n"
    log_line(write_line, progress_file, True)
    log_line("Scalerization Enter\n", progress_file, False)

    def newtour(energy_1, energy_2, turns_1, turns_2, depth, mip_tours) -> None:
        """
        This function finds the mid points of the efficiency frontier between two points using recursion
        """
        log_line(f" {depth} - ", progress_file, False)
        if energy_1 is None or energy_2 is None or turns_1 is None or turns_2 is None:
            log_line(f"{route_id},1\n", filename, False)
            return
        if (round(energy_2, 3) == round(energy_1, 3)) or (turns_2 == turns_1):
            log_line(f"{route_id},2\n", filename, False)
            return
        if get_time_left(ip_constants['allowed_time'], mip_start_stamp) is False:
            log_line(f"{route_id},3\n", filename, False)
            return

        w = abs((energy_2 - energy_1) / (turns_2 - turns_1))
        time_remaining = ip_constants['allowed_time'] - (time.time() - mip_start_stamp)
        energy_3, turns_3, path_3 = \
        mip_main(w / (w + 1), 1 / (w + 1), L, energy_cost, turn_cost, time_cost_dict, time_windows, terminal_list, revisits, progress_file, time_remaining, departure_time_bounds)[0]

        if energy_3 is None or turns_3 is None or path_3 is None:
            log_line(f"{route_id},4\n", filename, False)
            return
        if (round(energy_3, 3), turns_3) == (round(energy_1, 3), turns_1) or (round(energy_3, 3), turns_3) == (round(energy_2, 3), turns_2):
            log_line(f"{route_id},5\n", filename, False)
            return
        if depth > ip_constants['max_recursion_depth']:
            log_line(f"{route_id},6\n", filename, False)
            return

        mip_tours.append((energy_3, turns_3, path_3))

        depth += 1
        newtour(energy_3, energy_2, turns_3, turns_2, depth, mip_tours)
        newtour(energy_1, energy_3, turns_1, turns_3, depth, mip_tours)

    if get_time_left(ip_constants['allowed_time'], mip_start_stamp):
        log_line("  Finding least energy tour -", progress_file, False)
        time_remaining = ip_constants['allowed_time'] - (time.time() - mip_start_stamp)
        energy_1, turns_1, path_1 = mip_main(almost_zero, 1, L, energy_cost, turn_cost, time_cost_dict, time_windows, terminal_list, revisits, progress_file, time_remaining, departure_time_bounds)[0]
        if energy_1 is not None:
            mip_tours.append((energy_1, turns_1, path_1))
            atlaest_one_point_found = True

    if get_time_left(ip_constants['allowed_time'], mip_start_stamp):
        log_line("  Finding least turn tour\n", progress_file, False)
        time_remaining = ip_constants['allowed_time'] - (time.time() - mip_start_stamp)
        energy_2, turns_2, path_2 = mip_main(1, almost_zero, L, energy_cost, turn_cost, time_cost_dict, time_windows, terminal_list, revisits, progress_file, time_remaining, departure_time_bounds)[0]
        if energy_2 is not None:
            mip_tours.append((energy_2, turns_2, path_2))

    depth = 0
    if get_time_left(ip_constants['allowed_time'], mip_start_stamp):
        log_line(f"  depth: ", progress_file, False)
        newtour(energy_1, energy_2, turns_1, turns_2, depth, mip_tours)
    mip_tours, ip_time = post_process_mip_tours(mip_tours, mip_start_stamp, route_id)
    log_line("Scalerization Exit.\n", progress_file, False)
    save_mip_results(route_id, mip_tours, ip_time)
    return atlaest_one_point_found


def get_mip_results(route_id: str) -> tuple:
    """
    This function is used to obtain the tours generated by the MIP model.

    Args:
        route_id (str): route id

    Returns:
        tuple: mip_tours, ip_time
    """
    folder_path = f'./logs/IPOutput'
    with open(f'{folder_path}/IPResults_{route_id}.pkl', 'rb') as f:
        mip_tours, ip_time = pickle.load(f)
    return mip_tours, ip_time


def save_mip_results(route_id: str, mip_tours: list, ip_time: float) -> None:
    """
    This function saves the MIP results to a specific file.

    Args:
        route_id (str): route id
        mip_tours (list): list of MIP tours
        ip_time (float): time taken by IP

    Returns:
        None
    """
    folder_path = f'./logs/IPOutput'
    with open(f'{folder_path}/IPResults_{route_id}.pkl', 'wb') as f:
        pickle.dump((mip_tours, ip_time), f)
    return None


def post_process_mip_tours(mip_tours: list, mip_start_stamp: float, route_id: str) -> tuple:
    """
    This function post processes the results obtained from the MIP-based scalarization method.

    Args:
        mip_tours (list): list of MIP tours
        mip_start_stamp (float): start time
        route_id (str): route id

    Returns:
        ip_tours (list): list of MIP tours
        ip_time (float): time taken by MIP
    """
    if mip_tours:
        scalerization_objs = [(e, t) for e, t, _ in mip_tours]
        ip_mask = get_pareto_set(scalerization_objs)
        ip_tours = [mip_tours[i] for i in range(len(ip_mask)) if ip_mask[i]]
    else:
        ip_tours = []
    ip_time = time.time() - mip_start_stamp
    print(f"IP done for route {route_id}. Time: {round(time.time() - mip_start_stamp)} seconds")
    return ip_tours, ip_time
