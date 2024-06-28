import folium
import pandas as pd
import multiprocessing
from localSearch.lsInitial import mixintprog
from folium.vector_layers import CircleMarker
from shapely.geometry import LineString

from helpers.graphBuilder import *
from preprocessing import process_primal_graph
from helpers.functions import remove_consecutive_duplicates, get_path_length

ox.settings.use_cache = True


def addToMap(tour, graph, map_center, primal_energies):
    # tour, graph, map_center, primal_energies  = least_turns_tour_primal, G, map_center, primal_energies
    attr = (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
        'contributors, &copy; <a href="https://cartodb.com/attributions">CartoDB</a>'
    )
    tiles = "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png"
    mapObj = folium.Map(location=map_center, zoom_start=14, tiles=tiles, attr=attr)
    edges = [(tour[i], tour[i + 1]) for i in range(len(tour) - 1)]
    negative_edges = primal_energies.intersection(edges)
    for edge in edges:
        if edge in negative_edges:
            edge_color = "green"
            thickness = 5
        else:
            edge_color = "black"
            thickness = 2
        try:
            if "geometry" in graph[edge[0]][edge[1]][0]:
                geometry = graph[edge[0]][edge[1]][0]["geometry"]
                line_coords = [(lat, lon) for lon, lat in LineString(geometry).coords]
                folium.PolyLine(line_coords, color=edge_color, weight=thickness, opacity=1).add_to(mapObj)
            else:
                folium.PolyLine([(graph.nodes[edge[0]]['y'], graph.nodes[edge[0]]['x']), (graph.nodes[edge[1]]['y'], graph.nodes[edge[1]]['x'])], color=edge_color, weight=3, opacity=1).add_to(mapObj)
        except KeyError:
            pass
    return mapObj


def getConflicts(tour, primal_graph_processed, G):
    # tour = least_turns_tour_primal
    tour = remove_consecutive_duplicates(tour)
    conflicting_nodes = []
    for node_idx in range(1, len(tour)):
        prev_node = tour[node_idx - 1]
        node = tour[node_idx]
        if node_idx == len(tour) - 1:
            next_node = tour[0]
        else:
            next_node = tour[node_idx + 1]
        if node == next_node or node == prev_node:
            continue
        if primal_graph_processed[node].adj_nodes_turns[prev_node, node, next_node] == 1:
            conflicting_nodes.append(node)
    conflicting_nodes_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in conflicting_nodes]
    return conflicting_nodes_coords


def plot_routes(route_num):
    amazon_del_coordinates, original_bbox, time_windowlist, depot = read_dataset(route_num)
    (G, L, terminal_set, time_window_dict, time_cost_dict, energy_dual, direction_dual, mean_energy_dual, mean_turn_dual, tw_term_count,
     stdev_energy_dual, stdev_turn_dual, depot) = get_route_information(route_num)
    combined_dict = {i: {j: (energy_dual[i][j], direction_dual[i][j]) for j in L.neighbors(i)} for i in L.nodes()}
    primal_graph_processed = process_primal_graph(G, straight_angle_tolerance)
    primal_energies = pd.read_csv(f'./lsInputs/energy_primal_{route_num}.csv')
    primal_energies = primal_energies[primal_energies.energy < 0]
    primal_energies = set([tuple(x) for x in primal_energies[["v1", "v2"]].values.tolist()])
    allowed_time = 60 * 60 * 2
    allowed_init_time = allowed_time / 5
    allowed_ls_time = allowed_time - allowed_init_time
    nbd_flags = {"S3opt": True, "S3optTW": True, "GapRepair": True, "FixedPerm": True, "Quad": True, "RandomPermute": True, "SimpleCycle": True}
    ls_constants = {"allowed_time": allowed_time, "allowed_init_time": allowed_init_time, "allowed_ls_time": allowed_ls_time, "cores": 22,
                    "parallelize": True, "generate_log_file": False, "set_p_limit": 100, "ls_sp_timeout_limit": 30, "or_factor": 1000, "max_recursion_depth": 50,
                    "nbd_flags": nbd_flags, "max_nb_runtime": 60 * 5, "fp_std_dev_factor": 0.1, "set_y_limit": 100, "fp_terminal_count": 8, "fp_flip_count": 2, "RMpathCount": 1,
                    "simple_cycle_time": 10, "gr_max_tries": 10, "fp_max_tries": 5, "quad_max_tries": 15, "gr_terminal_count": 8, "gr_flip_count": 3, "rm_tries": 1}
    almost_zero = 0.001
    tw_coodinates = []
    non_tw_coodinates = []
    for x in range(len(time_windowlist)):
        if str(time_windowlist[x][0]) != '-inf' or str(time_windowlist[x][1]) != 'inf':
            tw_coodinates.append(amazon_del_coordinates[x])
        else:
            non_tw_coodinates.append(amazon_del_coordinates[x])
    with open(f'./lsInputs/route_{route_num}_G.pkl', 'rb') as f:
        G = pickle.load(f)
    tw_coodinates = tw_coodinates + non_tw_coodinates

    map_center = (original_bbox[0][0] + original_bbox[2][0]) / 2, (original_bbox[1][1] + original_bbox[3][1]) / 2

    # Plot least turns tour
    least_turns_tour, energy_2, turns_2, _ = mixintprog(1, almost_zero, energy_dual, direction_dual, terminal_set, L, route_num, ls_constants)
    duration = round(get_path_length(least_turns_tour, time_cost_dict))
    print(f"Least Turns: Energy: {round(energy_2, 1)} Turns: {turns_2} Duration: {round(duration / (60* 60), 1)}")
    least_turns_tour_primal = [i[0] for i in least_turns_tour]
    least_turns_tour_map = addToMap(least_turns_tour_primal, G, map_center, primal_energies)
    for i in range(len(tw_coodinates)):
        CircleMarker(tw_coodinates[i], radius=1, color="blue", fill=True, fill_color="blue").add_to(least_turns_tour_map)
    conflicting_nodes_coords = getConflicts(least_turns_tour_primal, primal_graph_processed, G)
    for i in range(len(conflicting_nodes_coords)):
        CircleMarker(conflicting_nodes_coords[i], radius=2, color="red", fill=True, fill_color="red").add_to(least_turns_tour_map)
    least_turns_tour_map.save(f'./routeVisualization/{route_num}_leastTurns.html')

    # Plot weighted tour
    weighted_tour, energy_3, turns_3, _ = mixintprog(0.5, 0.5, energy_dual, direction_dual, terminal_set, L, route_num, ls_constants)
    duration = round(get_path_length(weighted_tour, time_cost_dict))
    print(f"Weighted: Energy: {round(energy_3, 1)} Turns: {turns_3} Duration: {round(duration / (60* 60), 1)}")
    weighted_tour_primal = [i[0] for i in weighted_tour]
    weighted_tour_map = addToMap(weighted_tour_primal, G, map_center, primal_energies)
    for i in range(len(tw_coodinates)):
        CircleMarker(tw_coodinates[i], radius=2, color="blue", fill=True, fill_color="blue").add_to(weighted_tour_map)
    conflicting_nodes_coords = getConflicts(weighted_tour_primal, primal_graph_processed, G)
    for i in range(len(conflicting_nodes_coords)):
        CircleMarker(conflicting_nodes_coords[i], radius=2, color="red", fill=True, fill_color="red").add_to(weighted_tour_map)
    weighted_tour_map.save(f'./routeVisualization/{route_num}_weighted.html')

    # Plot least energy tour
    least_energy_tour, energy_5, turns_5, _ = mixintprog(almost_zero, 1, energy_dual, direction_dual, terminal_set, L, route_num, ls_constants)
    duration = round(get_path_length(least_energy_tour, time_cost_dict))
    print(f"Least Energy: Energy: {round(energy_5, 1)} Turns: {turns_5} Duration: {round(duration / (60* 60), 1)}")
    least_energy_tour_primal = [i[0] for i in least_energy_tour]
    least_energy_tour_map = addToMap(least_energy_tour_primal, G, map_center, primal_energies)
    for i in range(len(tw_coodinates)):
        CircleMarker(tw_coodinates[i], radius=2, color="blue", fill=True, fill_color="blue").add_to(least_energy_tour_map)
    conflicting_nodes_coords = getConflicts(least_energy_tour_primal, primal_graph_processed, G)
    for i in range(len(conflicting_nodes_coords)):
        CircleMarker(conflicting_nodes_coords[i], radius=2, color="red", fill=True, fill_color="red").add_to(least_energy_tour_map)
    least_energy_tour_map.save(f'./routeVisualization/{route_num}_leastEnergy.html')

    # Plot least time tour
    least_time_tour, energy_4, turns_4, _ = mixintprog(1, 0, energy_dual, time_cost_dict, terminal_set, L, route_num, ls_constants)
    energy_4 = get_path_length(least_time_tour, energy_dual)
    turns_4 = get_path_length(least_time_tour, direction_dual)
    duration = round(get_path_length(least_time_tour, time_cost_dict))
    print(f"Least Time: Energy: {round(energy_4, 1)} Turns: {turns_4} Duration: {round(duration / (60* 60), 1)}")
    least_time_tour_primal = [i[0] for i in least_time_tour]
    least_time_tour_map = addToMap(least_time_tour_primal, G, map_center, primal_energies)
    for i in range(len(tw_coodinates)):
        CircleMarker(tw_coodinates[i], radius=2, color="blue", fill=True, fill_color="blue").add_to(least_time_tour_map)
    conflicting_nodes_coords = getConflicts(least_time_tour_primal, primal_graph_processed, G)
    for i in range(len(conflicting_nodes_coords)):
        CircleMarker(conflicting_nodes_coords[i], radius=2, color="red", fill=True, fill_color="red").add_to(least_time_tour_map)
    least_time_tour_map.save(f'./routeVisualization/{route_num}_leastTime.html')

    return None


if __name__ == '__main__':
    cores = 10
    all_routes = pd.read_csv(f"lsInputs/working_amazon_routes.csv").Route_num.tolist()
    straight_angle_tolerance = math.pi / 4  # 45 degrees
    # with multiprocessing.Pool(processes=cores) as pool:
    #     outputs = pool.map(plot_routes, all_routes)
    all_routes = [4946]
    for route_num in all_routes:
        plot_routes(route_num)
