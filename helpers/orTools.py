"""
This module contains functions to solve TSP using Google OR tools.
"""
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


def create_or_model(cost_matrix: list[list]):
    """
    This method creates the data dictionary for a problem.

    Args:
        cost_matrix (list[list]): shortest path length between all nodes

    Returns:
        data (dict): data dict for the problem

    """
    data = {}
    data['distance_matrix'] = cost_matrix
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def print_solution_or_tools(manager, routing, solution, or_factor):
    """
    This method prints on the console the TSP solution obtained from Google OR Tools.

    Args:
        manager (obj): manager object from Google OR tools
        routing (obj): routing object from Google OR tools
        solution (obj): solution object from Google OR tools
        or_factor (int): OR factor used in printing

    Returns:
        None
    """
    # print(f'Objective: {solution.ObjectiveValue()/or_factor} miles')
    if int(solution.ObjectiveValue()) >= 100000000:
        print("Google OR tools: no Solution Exists. Error code: 3432")
        return None
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    plan_output += 'Route distance: {}miles\n'.format(route_distance)
    return 1


def solve_tsp_using_or_tools(cost_matrix: list[list], or_factor, stop_time=None) -> tuple:
    """
    This method solves a Travelling Salesman Problem using Google OR Tools.

    Args:
        cost_matrix (list[list]): shortest path length between all nodes
        or_factor (int): OR factor used in printing
        stop_time (int, optional): Defaults to None.

    Returns:
        solution (obj): solution object from Google OR tools
        routing (obj): routing object from Google OR tools
        manager (obj): manager object from Google OR tools

    """
    data = create_or_model(cost_matrix)

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    if stop_time != None:
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = stop_time

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        solution.ObjectiveValue()

        value = print_solution_or_tools(manager, routing, solution, or_factor)
        if value == None:
            return None, None, None

    return solution, routing, manager


def get_tsp_routes(solution, routing, manager) -> list:
    """
    This function gets the vehicle routes from a solution and stores them in an array.

    Args:
        solution (obj): solution object from Google OR tools
        routing (obj): routing object from Google OR tools
        manager (obj): manager object from Google OR tools
    """
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes
