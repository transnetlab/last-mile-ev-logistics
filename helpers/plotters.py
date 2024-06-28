import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from helpers.functions import get_closest_point

# warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def plot_efficiency_frontier(result_combined: tuple, route_num: int, allowed_time: int) -> None:
    """
    Plot the efficiency frontier for the given route number

    Args:
        result_combined (tuple): Combined results from the LS
        route_num (int): Route number
        allowed_time (int): Allowed time for the LS

    Returns:
        None

    """
    folder_path = f'./logs/{route_num}/plots/'
    bstsptw_tours, bstsp_tours, set_z = result_combined
    if bstsptw_tours:
        bstsptw_turns, bstsptw_energy = zip(*[(t, e) for p, e, t in bstsptw_tours])
    else:
        bstsptw_turns, bstsptw_energy = [], []
    if bstsp_tours:
        bstsp_turns, bstsp_energy = zip(*[(t, e) for p, e, t, _ in bstsp_tours])
    else:
        bstsp_turns, bstsp_energy = [], []
    if set_z:
        ls_turns, ls_energy = zip(*[(t, e) for p, e, t in set_z])
    else:
        ls_turns, ls_energy = [], []

    plt.clf()
    # plt.figure(figsize=(5, 5))
    plt.plot(bstsp_turns, bstsp_energy, marker='o', linestyle='dotted', linewidth=3, color='orange', label=f"Initial BSTSP", zorder=1)
    plt.scatter(bstsptw_turns, bstsptw_energy, color='darkgreen', marker='o', facecolor='none', s=120, linewidth=1,
                label=f"Initial BSTSPTW", zorder=2)
    plt.plot(ls_turns, ls_energy, '*--', color='steelblue', label=f"LS", markersize=8, linewidth=3, zorder=3)
    plt.title(f"{route_num}", fontsize=14)
    plt.xlabel("Turn count", fontsize=14)
    plt.ylabel("Energy (kWh)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f"{folder_path}/LS_{route_num}_{allowed_time}_All.pdf")
    plt.legend(fontsize=14, loc='upper right')
    plt.grid(False)
    # plt.show()
    plt.clf()
    return None


def plot_clustering_img(result_combined: tuple, route_num: int, allowed_time: int) -> None:
    """
    Plot the tour clustering for the given route number

    Args:
        result_combined (tuple): Combined results from the LS
        route_num (int): Route number
        allowed_time (int): Allowed time for the LS

    Returns:
        None
    """
    folder_path = f'./logs/{route_num}/plots/'
    bstsptw_tours, bstsp_tours, set_z = result_combined
    if bstsptw_tours:
        bstsptw_turns, bstsptw_energy = zip(*[(t, e) for p, e, t in bstsptw_tours])
    else:
        bstsptw_turns, bstsptw_energy = [], []
    if bstsp_tours:
        bstsp_turns, bstsp_energy = zip(*[(t, e) for p, e, t, _ in bstsp_tours])
    else:
        bstsp_turns, bstsp_energy = [], []
    if set_z:
        ls_turns, ls_energy = zip(*[(t, e) for p, e, t in set_z])
    else:
        ls_turns, ls_energy = [], []
    # if len(ls_turns) == 0 or len(bstsptw_turns) == 0 or len(bstsp_turns) == 0 or len(bstsptw_energy) == 0 or len(bstsp_energy) == 0 or len(ls_energy) == 0:
    #     return None
    plt.clf()
    plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization
    plt.plot(bstsp_turns, bstsp_energy, marker='o', linestyle='dotted', linewidth=3, color='orange',
             label=f"BSTSP initial tours", zorder=1)
    plt.scatter(bstsptw_turns, bstsptw_energy, color='darkgreen', marker='o', facecolor='none', s=120, linewidth=1,
                label=f"BSTSPTW initial tours", zorder=2)
    plt.xlabel("Turn count", fontsize=22)
    plt.ylabel("Energy (kWh)", fontsize=22)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{folder_path}/Clustering_InitialSol_{route_num}_{allowed_time}.pdf")
    # plt.show()
    plt.clf()

    plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization
    plt.scatter(bstsptw_turns, bstsptw_energy, color='darkgreen', marker='o', alpha=0, facecolor='none', s=120, linewidth=1, zorder=2)
    plt.plot(ls_turns, ls_energy, '*--', color='steelblue', label=f"Local search tours", markersize=8, linewidth=3, zorder=3)
    # plt.title(f"{route_num}", fontsize=20)
    plt.xlabel("Turn count", fontsize=22)
    plt.ylabel("Energy (kWh)", fontsize=22)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f"{folder_path}/Clustering_LSSol_{route_num}_{allowed_time}.pdf")
    # plt.show()
    plt.clf()

    x_val = [[x, y] for x, y in zip(ls_turns, ls_energy)]
    optimal_num_clusters = min(6, len(ls_turns))
    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    kmeans.fit(x_val)
    labels = kmeans.labels_
    rep_x = []
    rep_y = []
    for i in range(optimal_num_clusters):
        target = (kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1])
        centre_x, centre_y = get_closest_point(x_val, target)
        rep_x.append(centre_x)
        rep_y.append(centre_y)
    data_x = [val[0] for val in x_val]
    data_y = [val[1] for val in x_val]
    data_x, data_y, labels = zip(*sorted(zip(data_x, data_y, labels)))

    plt.figure(figsize=(10, 7))  # Adjust figure size for better visualization
    plt.plot(data_x, data_y, c='steelblue', linestyle='dotted')
    plt.scatter(data_x, data_y, c=labels, cmap='rainbow', alpha=0.6)
    plt.scatter(rep_x, rep_y, s=200, c='black', alpha=0.9, label='Representative paths', marker='*')
    plt.xlabel("Turn count", fontsize=22)
    plt.ylabel("Energy (kWh)", fontsize=22)
    plt.title(f"Route Id {route_num} (After clustering)", fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(False)
    # plt.legend(fontsize=14, loc='upper right')
    plt.savefig(f"{folder_path}/Clustering_Clustered_{route_num}_{allowed_time}_All.pdf")
    # plt.show()
    plt.clf()
    return None

def plotSampleNetworkTours():
    # import matplotlib.pyplot as plt
    BSTSPTW_pareto = [(4, 14), (6,11), (8, 10)]
    BSTSP = [(4, 19), (5, 6)]
    BSTSPTW_nonpareto = [(7, 18)]
    plt.figure(figsize=(10, 8))  # Adjust figure size for better visualization
    BSTSPTW_pareto_x, BSTSPTW_pareto_y = zip(*BSTSPTW_pareto)
    plt.plot(BSTSPTW_pareto_x, BSTSPTW_pareto_y, color='black', linewidth=3)
    plt.scatter(BSTSPTW_pareto_x, BSTSPTW_pareto_y, color='pink', edgecolors='grey', marker='o', s=600, zorder=2)
    BSTSPTW_x, BSTSPTW_y = zip(*BSTSPTW_nonpareto)
    plt.scatter(BSTSPTW_x, BSTSPTW_y, color='pink', edgecolors='grey', marker='o', s=600, zorder=2)
    BSTP_x, BSTP_y = zip(*BSTSP)
    plt.scatter(BSTP_x, BSTP_y, color='darkgreen', marker='o', facecolor='none', s=700, linewidth=2)
    plt.scatter(BSTP_x, BSTP_y, color='black', marker='o', facecolor='black', s=80, linewidth=1)
    # plt.plot(BSTP_x, BSTP_y, color='orange', linestyle='dotted', linewidth=3)
    # plt.scatter(BSTSPTW_pareto_x, BSTSPTW_pareto_y, color='darkgreen', marker='o', facecolor='none', s=240, linewidth=2)
    # plt.scatter(BSTSPTW_pareto_x, BSTSPTW_pareto_y, color='black', marker='o', facecolor='black', s=40, linewidth=1)
    plt.xlabel("Turn count", fontsize=34)
    plt.ylabel("Energy (kWh)", fontsize=34)
    # manually set x ticks
    plt.xticks([3, 4, 5, 6, 7, 8], fontsize=24)
    plt.yticks([6, 8, 10, 12, 14, 16, 18, 20], fontsize=24)
    plt.grid(False)
    plt.yticks(fontsize=24)
    # put a box on all four sides around the plot of pink color
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['left'].set_color('black')
    # plt.show()
    plt.savefig(f"sample_network_tours.pdf")
    plt.clf()
    return None


"""
def plot_lineage_histogram(parents, result_combined, route_num):
    folder_path = f'./logs/{route_num}/plots/'
    bstsptw_tours, bstsp_tours, set_z = result_combined
    parents2 = dict(parents)
    root_nodes = [p for p, e, t in bstsptw_tours] + [p for p, e, t, _ in bstsp_tours] + [tuple()]
    ls_paths_final = [tuple(p) for p, _, _ in set_z]
    move_counter = {'FixedPerm': 0, 'S3opt': 0, 'S3optTW': 0, 'Quad': 0, 'GapRepair': 0, 'SimpleCycle': 0, 'RandomPermute': 0}
    for parent_tour in ls_paths_final:
        while parent_tour not in root_nodes:
            parent_tour2, move = parents2[parent_tour][0]
            move_counter[move] += 1
            parent_tour2 = tuple(parent_tour2)
            parent_tour = parent_tour2
    # PLot the bar plot
    plt.bar(["S3opt", "S3optTW", "GapRepair", "FixedPerm", "Quad", "SimpleCycle", "RandomPermute"],
            [move_counter["S3opt"], move_counter["S3optTW"], move_counter["GapRepair"], move_counter["FixedPerm"], move_counter["Quad"], move_counter["SimpleCycle"], move_counter["RandomPermute"]],
            color='skyblue', edgecolor='black')
    plt.xlabel('Move')
    plt.ylabel('Frequency')
    plt.title(f'Move Analysis for {route_num}')
    plt.savefig(f"{folder_path}/MoveAnalysis_{route_num}.pdf")
    # plt.show()
    plt.clf()
    return None

def generate_plots_IP(ip_test, efficient_turns_ip, efficient_energy_ip, benchmark_id, time_taken_ip, bstsp_turns, bstsp_energy, bstsptw_turns, bstsptw_energy,
                      terminal_percent, L, ls_turns, ls_energy, terminal_set, ls_paths, bstsptw_paths, last_imp_time):
    plt.clf()
    if ip_test:
        plt.plot(efficient_turns_ip, efficient_energy_ip,
                 label=f'IP (time = {round(time_taken_ip)}s, #{len(efficient_turns_ip)})', color='k')
        plt.scatter(efficient_turns_ip, efficient_energy_ip, color='k', marker='o', facecolors='none', s=320,
                    linewidths=2, zorder=1)

    plt.plot(bstsp_turns, bstsp_energy, color='orange', linestyle='dotted', linewidth=3, marker='o',
             label=f"Initial BSTSP tours #{len(ls_paths)}", zorder=2)

    plt.scatter(bstsptw_turns, bstsptw_energy, color='darkgreen', marker='o', facecolor='none', s=120,
                linewidth=1, label=f"BSTSPTW initial tours #{len(bstsptw_paths)}", zorder=3)

    if len(ls_turns):
        turns_ls, energy_ls = zip(*sorted(zip(ls_turns, ls_energy)))

    plt.plot(turns_ls, energy_ls, "*--", linewidth=2, markersize=8, color='steelblue',
             label=f"Final BSTSTW tours (time {round(last_imp_time, 0)}s, #{len(ls_energy)})", zorder=4)

    plt.xlabel("Turns count", fontsize=20)
    plt.ylabel("Energy (kWh)", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # plt.legend(fontsize=14, loc='upper right')
    plt.title(f"{benchmark_id}_{len(L.nodes)}_{len(terminal_set)}", fontsize=20)
    plt.tight_layout()
    plt.savefig(
        f"final_results/benchmarks/{terminal_percent}_{benchmark_id}_{len(L.nodes)}_{len(terminal_set)}_AllNbd.pdf")
    # plt.show()
    return energy_ls



def write_newline_IP(energy_ls, ls_turns, ls_energy, efficient_turns_ip, efficient_energy_ip, allowed_time, time_windows, L, fixed_pass_list, bstsp_paths,
                     bstsptw_turns_copy, bstsptw_energy_copy, benchmark_id, time_taken_ip, time_taken_ls, p):
    if len(energy_ls) > 0:
        A = np.array(list(zip(ls_turns, ls_energy)))
        Z = np.array(list(zip(efficient_turns_ip, efficient_energy_ip)))
        # Calculate Euclidean distances from each point in A to the nearest reference point in Z
        distances = np.min(np.sqrt(np.sum((A[:, np.newaxis] - Z) ** 2, axis=2)), axis=1)
        # Calculate the average GD
        average_GD = np.power(np.mean(np.power(distances, p)), 1 / p)
        print("Generalized Distance all(GD):", average_GD)

        distances = np.min(np.sqrt(np.sum((A[:, np.newaxis] - Z) ** 2, axis=2)), axis=1)
        distances = list(sorted(distances))[0: len(efficient_turns_ip)]
        # Calculate the average GD
        average_GD_c = np.power(np.mean(np.power(distances, p)), 1 / p)
        print("Generalized Distance common(GD):", average_GD_c)
        print("allowed time", allowed_time)

        time_range = {v: tw[1] - tw[0] for v, tw in time_windows.items() if tw[1] - tw[0] <= 1000}

        len_L_nodes = len(L.nodes)
        len_L_edges = len(L.edges)
        len_fixed_pass_list = len(fixed_pass_list)
        tw_term_count = len(time_range)
        len_efficient_turns = len(bstsp_paths)
        len_turns_tw = len(bstsptw_turns_copy)
        len_energy_ls = len(bstsptw_energy_copy)
        len_efficient_turns_ip = len(efficient_turns_ip)

        rounded_values = [
            benchmark_id, len_L_nodes, len_L_edges,
            len_fixed_pass_list, len_efficient_turns,
            len_turns_tw, len_energy_ls, len_efficient_turns_ip,
            round(allowed_time, 1), round(time_taken_ip, 1),
            round(time_taken_ls, 1), round(average_GD, 1), round(average_GD_c, 1)
        ]

        formatted_line = '\n'
        formatted_line += ' & '.join(map(str, rounded_values))
        with open(f"./logs/final_IPlog_file.txt", 'a') as file:
            file.write(formatted_line)
    return None

def generate_plots(route_num, result_combined, allowed_time, nbd_tw_feasible_tours, nbd_optimal_tours, nbd_flags, parents):
    plot_efficiency_frontier(result_combined, route_num, allowed_time)
    plot_clustering_img(result_combined, route_num, allowed_time)
    plot_nbd_compare(nbd_tw_feasible_tours, nbd_optimal_tours, result_combined, nbd_flags, route_num, allowed_time)
    plot_lineage_tree(parents, result_combined, route_num)
    plot_lineage_histogram(parents, result_combined, route_num)
    return None

def plot_nbd_compare(nbd_tw_feasible_tours, nbd_optimal_tours, result_combined, nbd_flags, route_num, allowed_time):
    folder_path = f'./logs/{route_num}/plots/'
    _, _, set_z = result_combined
    if set_z:
        ls_turns, ls_energy = zip(*[(t, e) for p, e, t in set_z])
    else:
        ls_turns, ls_energy = [], []

    color_dict = {'S3opt': "g", 'S3optTW': "m", 'GapRepair': "y", 'FixedPerm': "c", 'Quad': "k", "SimpleCycle": "r", "RandomPermute": "b"}
    marker_dict = {'S3opt': "v", 'S3optTW': "s", 'GapRepair': "D", 'FixedPerm': "o", 'Quad': "P", "SimpleCycle": "X", "RandomPermute": "x"}
    plt.figure(figsize=(7, 5))
    plt.plot(ls_turns, ls_energy, linestyle='dotted', linewidth=1, color='steelblue', alpha=0.7, zorder=1)
    plt.xlabel("Turn count", fontsize=18)
    plt.title(f"{route_num}", fontsize=18)
    plt.grid(False)
    plt.ylabel("Energy (kWh)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for nbd in nbd_flags.keys():
        if not nbd_flags[nbd]: continue
        tw_feasible_energies = [e for p, e, t in nbd_tw_feasible_tours[nbd]]
        tw_feasible_turns = [t for p, e, t in nbd_tw_feasible_tours[nbd]]
        optimal_energies = [e for p, e, t in nbd_optimal_tours[nbd]]
        optimal_turns = [t for p, e, t in nbd_optimal_tours[nbd]]

        plt.scatter(tw_feasible_turns, tw_feasible_energies, c=color_dict[nbd], marker=marker_dict[nbd], label=f"{nbd}: {len(tw_feasible_turns)}")
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()

    seq = [nbd_flags[x] for x in ["S3opt", "S3optTW", "GapRepair", "FixedPerm", "Quad", "SimpleCycle", "RandomPermute"]]
    seq = "-".join(map(str, seq))
    plt.savefig(f"{folder_path}/NBD_{seq}_{route_num}_{allowed_time}.pdf")
    # plt.show()
    plt.clf()
    return None

def plot_lineage_tree(parents, result_combined, route_num):
    folder_path = f'./logs/{route_num}/plots/'
    bstsptw_tours, bstsp_tours, set_z = result_combined
    parents2 = dict(parents)
    G = nx.DiGraph()
    root_nodes1 = [p for p, e, t in bstsptw_tours]
    root_nodes2 = [p for p, e, t, _ in bstsp_tours]
    root_nodes = root_nodes1 + root_nodes2 + [tuple()]
    ls_paths_final = [tuple(p) for p, _, _ in set_z]
    root_nodes = set(root_nodes)
    for parent_tour in ls_paths_final:
        while parent_tour not in root_nodes:
            parent_tour2, move = parents2[parent_tour][0]
            parent_tour2 = tuple(parent_tour2)
            G.add_edge(parent_tour2, parent_tour, label=move)
            parent_tour = parent_tour2
    mapping = {y: x for x, y in enumerate(G.nodes())}
    rev_map = {x: y for x, y in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    labels = nx.get_edge_attributes(G, 'label')
    pos = nx.spring_layout(G)

    good_nodes = [i for i, n in rev_map.items() if n in ls_paths_final]
    colors = ["salmon" if node in good_nodes else "lightgrey" for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=200, node_color=colors, font_size=5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red', font_size=5)
    plt.title(f'Lineage Tree for Route {route_num}.\nLS Paths: {len(ls_paths_final)}. Scal paths: {len(root_nodes)}')
    plt.savefig(f"{folder_path}/LineageTree.pdf")
    # plt.show()
    plt.clf()
    return None


"""
