"""
This module contains helper functions for finding shortest paths in a network
"""
import networkx
import networkx as nx
import time, signal, heapq, copy
from collections import defaultdict


def is_objective_dominated(obj1: list, obj2: list) -> bool:
    """
    This function checks whether one objective is dominated by another objective.

    Args:
        obj1 (list): Objective 1. E.g. [1, 2]
        obj2 (list): Objective 2 E.g. [2, 3]

    Returns:
        bool: True if obj1 is dominated by obj2, False otherwise
    """
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] < obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    if sum_less != 0:
        return True
    return False


def add_labels(labels_dict: dict, obj_list: list, destination: int, temp_node: int, temp_obj: list) -> bool:
    """
    This function adds labels to the label dictionary.

    Args:
        labels_dict (dict): Label dictionary
        obj_list (list): List of objective values
        destination (int): Destination node
        temp_node (int): Temporary node
        temp_obj (list): Temporary objective values

    Returns:
        value (bool): True if the labels are added, False otherwise

    """
    # temp_node, temp_obj = epicenter, obj
    for index in labels_dict[temp_node]:
        if is_objective_dominated(temp_obj, obj_list[index]):
            return False
    for index in labels_dict[destination]:
        if is_objective_dominated(temp_obj, obj_list[index]):
            return False
    return True


def biobj_label_correcting(network, source, destination) -> tuple[list, list]:
    """
    This function calls the bi-objective label correcting algorithm.

    Args:
        network (dict): Network dictionary
        source (int): Source node
        destination (int): Destination node

    Returns:
        _paths_1 (list): List of paths
        _objs_1 (list): List of objective values for each path
    """
    # network = combined_dict
    # network, source, destination = test_network, 0, 4
    # Step 1. Initialization
    adj_dict = {node: list(values.keys()) for node, values in network.items()}
    obj_list = []
    path_list = []
    ni = 0
    labels_dict = {node: [] for node in network.keys()}
    queue = []
    ind = 0
    heapq.heappush(queue, (0, ind, {'path': [source], 'obj': [0, 0]}))
    ind += 1

    # Step 2. The main loop
    while queue:
        _, temp_ind, info = heapq.heappop(queue)
        path = info['path']
        obj = info['obj']
        epicenter = path[-1]
        if add_labels(labels_dict, obj_list, destination, epicenter, obj):
            labels_dict[epicenter].append(ni)
            obj_list.append(obj)
            path_list.append(path)
            ni += 1
            for node in adj_dict[epicenter]:
                if node not in path:
                    edge_weight = network[epicenter][node]
                    new_label = [obj[0] + edge_weight[0], obj[1] + edge_weight[1]]
                    temp_path = copy.deepcopy(path)
                    temp_path.append(node)
                    heapq.heappush(queue, (sum(new_label), ind, {'path': temp_path, 'obj': new_label}))
                    ind += 1

    # Step 3. Get the results
    _paths_ = [path_list[node] for node in labels_dict[destination]]
    _objs_ = [obj_list[node] for node in labels_dict[destination]]

    # Step 4.For paths with same objective values, take the one with shortest path length
    mydic = defaultdict(list)
    obj_path = list(zip(_objs_, _paths_))
    for obj, path in obj_path:
        mydic[tuple(obj)].append(path)
    mydic = dict(mydic)
    for obj, path_list in mydic.items():
        mydic[obj] = path_list[min(enumerate([len(path) for path in path_list]), key=lambda x: x[1])[0]]
    return_list = [(obj, path) for obj, path in mydic.items()]
    _objs_1, _paths_1 = zip(*return_list)
    _objs_1, _paths_1 = list(_objs_1), list(_paths_1)
    return _paths_1, _objs_1


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")


def call_with_timeout(timeout, func, net, s, d):
    """
    This method calls a function with a timeout.
    """
    # Set the timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Timeout in `timeout` seconds

    try:
        # Call the function and return the result if it completes before the timeout
        result = func(net, s, d)
    except TimeoutException:
        result = [], [[]]
    else:
        # Cancel the timeout alarm if the function returns before the timeout
        signal.alarm(0)

    return result


def bellman_ford(L: networkx.classes.multidigraph.MultiDiGraph, source: int, cost_dict: dict) -> dict:
    """
    This method finds the shortest path in a network by using the Bellman-Ford algorithm.

    Args:
        L (networkx.classes.multidigraph.MultiDiGraph): Dual graph
        source (int): Source node
        cost_dict (dict): Weighted cost dict

    Returns:
        distances (dict): Dictionary of distances from source node. Format: {node: distance}

    """
    # Initialize distance dictionary with infinity values for all nodes
    distances = {node: float('inf') for node in L.nodes()}
    distances[source] = 0
    nodes_list = list(L.nodes())
    nodes_list.remove(source)

    # Initialize a flag to keep track of relaxation
    relaxed = [source]

    # Perform relaxation of edges until no more relaxation is possible
    while relaxed:
        u = relaxed.pop()
        adj = L.adj[u]
        for v in adj:
            weight = cost_dict[u][v]
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                relaxed.append(v)
    return distances


def johnson_shortest_paths(L: networkx.classes.multidigraph.MultiDiGraph, cost_dict: dict, terminal_set: set) -> tuple:
    """
    This method finds the shortest path in a network with non-negative edge weights by using Johnson's algorithm.

    Args:
        L (networkx.classes.multidigraph.MultiDiGraph): Dual graph
        cost_dict (dict): Weighted cost dict
        terminal_set (set): Set of terminal nodes

    Returns:
        terminal_dict (dict): Format: {(terminal_source, terminal_desti): (distance, path)}
        bellman_time (int): Time taken for Bellman-Ford algorithm
        terminal_time (int): Time taken for terminal node calculation
    """
    start = time.time()
    s = 10000
    for v in L.nodes():
        cost_dict[s][v] = 0
    L.add_node(s)

    # Add edge weights
    for u, v, data in L.edges(data=True):
        data['length'] = cost_dict[u][v]
    for node in L.nodes():
        if node != s:
            L.add_edge(s, node, length=0)
    dist = bellman_ford(L, s, cost_dict)
    # Step 3: Re-weight the edges using the vertex weights
    for u, v in L.edges():
        try:
            L.get_edge_data(u, v)[0]['length'] = round(L.get_edge_data(u, v)[0]['length'] + dist[u] - dist[v], 5)
        except KeyError:
            L.get_edge_data(u, v)['length'] = round(L.get_edge_data(u, v)['length'] + dist[u] - dist[v], 5)
    L.remove_node(s)
    bellman_time = round(time.time() - start)
    start = time.time()
    terminal_dict = defaultdict(dict)
    for terminal_source in terminal_set:
        try:
            dist_dijk, path_dijk = nx.single_source_dijkstra(L, terminal_source, weight="length")
        except ValueError:
            return None, None, None
        for terminal_desti in terminal_set:
            try:
                terminal_dict[terminal_source][terminal_desti] = (dist_dijk[terminal_desti], path_dijk[terminal_desti])
            except KeyError:
                # Comes when the terminal is not reachable from the source
                terminal_dict[terminal_source][terminal_desti] = (100000000, [])
    terminal_dict = dict(terminal_dict)
    terminal_time = round(time.time() - start)
    return terminal_dict, bellman_time, terminal_time
