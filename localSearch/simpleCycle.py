"""
This file contains functions related to SimpleCycle operator
"""
import signal
import networkx as nx

from helpers.functions import cyclize
from localSearch.commonFunctions import remove_consecutive_duplicates, get_time_left


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")


def get_simple_cycle_with_time_limit(timeout: int, graph):
    """
    This function finds simple cycles in a graph while ensuring that the total processing time does not exceed a pre-specified limit.

    Args:
        timeout (int): Timeout in seconds
        graph (nx.Graph): Graph to find cycles in

    Returns:
        list: List of cycles in the graph
    """
    # Set the timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Timeout in `timeout` seconds

    try:
        # Call the function and return the result if it completes before the timeout
        result = sorted(list(nx.simple_cycles(graph)), key=lambda x: len(x))
    except TimeoutException:
        result = []
    else:
        # Cancel the timeout alarm if the function returns before the timeout
        signal.alarm(0)
    return result


def simple_cycle_operator(curr_tour: list, operator_stream: str, move_start_stamp: float, common_arguments: dict) -> tuple:
    """
    This function runs a single iteration of the Simple Cycle operator.

    Args:
        curr_tour (list): Current tour
        operator_stream (str): Log string for the operator
        move_start_stamp (float): Time the move started
        common_arguments (dict): Common arguments for the local search

    Returns:
        list: List of cycles
        str: Log string

    """
    ls_constants = common_arguments["ls_constants"]
    depot = common_arguments["depot"]
    terminal_set = common_arguments["terminal_set"]

    found = 0
    G = nx.DiGraph()
    tour = []

    for edges in zip(curr_tour, curr_tour[1:]):
        G.add_edge(edges[0], edges[1])
    cycles = get_simple_cycle_with_time_limit(ls_constants["simple_cycle_time"], G)

    operator_stream += f"{len(cycles)},"

    for tour in cycles:
        if not get_time_left(ls_constants["max_nb_runtime"], move_start_stamp):
            break
        missing_terminals = terminal_set - set(tour)
        if len(missing_terminals) == 0:
            found = 1
            break
    if found == 1:
        tour.append(tour[0])
        tour = cyclize(tour, tour.index(depot))
        tour = remove_consecutive_duplicates(tour)
        return [tuple(tour)], operator_stream
    return [], operator_stream
