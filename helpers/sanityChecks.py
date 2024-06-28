"""
This file contains functions to check the sanity of the solution from LS and IP algorithms
"""
import math


def sanity_check_1(terminals_in_v4v1_seg):
    if len(terminals_in_v4v1_seg) < 2:
        raise ValueError("length of terminals_in_v4v1_seg is <2. This should not happen")
    return None


def sanity_check_2(tourset):
    for index, newTour in enumerate(tourset):
        if newTour[0] != newTour[-1]:
            raise ValueError(f"Path {index} not cyclic in FBM Nbd. This should not happen.")
    return None


def sanity_check_3(v2):
    if v2 == 0:
        raise ValueError("Depot is violated in S3OptTW STep 1. Why is that")
    return None


def sanity_check_4(terminal_penalties):
    if len(terminal_penalties) == 0:
        raise ValueError("No terminal penalties. Check here")
    return None


def sanity_check_5(x_old):
    if x_old != sorted(x_old, reverse=True):
        raise ValueError("x_old not sorted. This should not happen. The efficiency frontir should be sorted.")


def sanity_check_6(bstsptw_tours, bstsp_tours):
    for path, e, t in bstsptw_tours:
        if path[0] != path[-1]:
            raise ValueError(
                f"Last node is not depot node. If this happens, figure out why. If this is correct, modify get_path_length function to include last edge also")
    for path, e, t, _ in bstsp_tours:
        if path[0] != path[-1]:
            raise ValueError(
                f"Last node is not depot node. If this happens, figure out why. If this is correct, modify get_path_length function to include last edge also")


def sanity_check_7(bstsptw_tours, bstsp_tours, route_num):
    if bstsp_tours is None and bstsptw_tours is None:
        raise ValueError(f"scalerization failed for route {route_num}\n")


def sanity_check_8(cycle_subpath):
    if not cycle_subpath:
        raise ValueError("No positive gain in step 1. Can this happen. If yes, document the case and let it pass")
        # Check it is. If the case if valid, document.
    return None


def sanity_check_9(p41, p63, p05, p27):
    if len(p41) == 0 or len(p63) == 0 or len(p05) == 0 or len(p27) == 0:
        raise ValueError("One of the paths is empty. This should not happen. Check here")
    return None


def sanity_check_10(terminal_set, new_path, operator_name):
    terminal_list_copy = list(terminal_set)
    for nodes in new_path:
        if nodes in terminal_list_copy:
            terminal_list_copy.remove(nodes)
    if len(terminal_list_copy) != 0:
        raise ValueError(f"reorder_and_fix_tour not working with nbd {operator_name}")
    return None


def sanity_check_11(terminal_set, new_path, operator_name):
    terminal_list_set = set(terminal_set)
    for terminal in terminal_list_set:
        if terminal not in new_path:
            raise ValueError(f"nbd {operator_name} is not working, {terminal} not present")
    return None


def sanity_check_12(unsatisfied_terminals):
    if not unsatisfied_terminals:
        raise ValueError("Explore set has a TW feasible soln. Exit code 3416")
    return None


def sanity_check_13(tour, terminal_set):
    # Test if the tour is valid
    for term in terminal_set:
        if term not in tour:
            raise ValueError(f"Error in tour. Terminal {term} not in tour")
    if tour[-1] != tour[0]:
        raise ValueError("Error in tour. Last node is not same as first node")
    return None


def sanity_check_14(v1_v4, v4_v5, v5_v2, v2_v3, v3_v6, v6_v1):
    if v1_v4[-1] != v4_v5[0] or v4_v5[-1] != v5_v2[0] or v5_v2[-1] != v2_v3[0] or v2_v3[-1] != v3_v6[0] or v3_v6[-1] != v6_v1[0]:
        raise ValueError("Cycle not broken properly")


def sanity_check_15(max_x, max_y):
    if max_x < 0 or max_y < 0:
        raise ValueError("Max value of HV is negative. This should not happen. Check here")
    return None


def sanity_check_16(terminal_set, set_z):
    for idx, (path, e, t) in enumerate(set_z):
        path_nodes = set(path)
        flag = terminal_set.issubset(path_nodes)
        if not flag: raise ValueError(f"Path {idx} in set_z failed: Terminals missing!")
    return None


def sanity_check_17(set_z, route_num):
    for idx, (path, e, t) in enumerate(set_z):
        if path[0] != path[-1]:
            raise ValueError(f"Path {idx} in set_z failed: Tour is not a cycle! in route {route_num}")
    return None


def sanity_check_18(terminal_set, newtours, depot):
    for tour in newtours:
        if terminal_set - set(tour):
            raise ValueError("Not all terminals are present in the tour")
        if tour[0] != tour[-1]:
            raise ValueError("First and last node are not same")
        if tour[0] != depot or tour[-1] != depot:
            raise ValueError("First and last node are not depot")

    return None


def sanity_check_20(used, curr_tour, route_num):
    if len(used) != 8:
        raise ValueError("Quad move failed: used list is not of length 8")
    if len(set(used)) != 8:
        raise ValueError("Quad move failed: used list has duplicates")
    if len(set([curr_tour[x] for x in used])) != 8:
        for x in used:
            print(curr_tour[x], x)
        print(curr_tour)
        print(curr_tour[0])
        print(route_num)
        raise ValueError("Quad move failed: used list has duplicates")
    return None


def sanity_check_21(weighted_paths_dict):
    for terminal_pair, path in weighted_paths_dict.items():
        if len(path) == 0:
            raise ValueError(f"Path for terminal pair {terminal_pair} is empty")
    return None


def sanity_check_22(time_window_dict, depot, route_num):
    if time_window_dict[depot] != (-float("inf"), float("inf")):
        raise ValueError(f"Depot time window is not (-inf, inf) for route {route_num}")
    return None


def sanity_check_23(angle):
    if - math.pi <= angle <= math.pi:
        return None
    else:
        raise ValueError("Angle not in range -pi to pi")
