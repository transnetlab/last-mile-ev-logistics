import math
import pickle
import shutil
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from shapely.geometry import Point, Polygon

from helpers.plotters import plot_efficiency_frontier, plot_clustering_img
from helpers.loggers import get_scalarization_results, get_results_ls, clear_collective_stats

def get_table_8():
    # Columns to extract from the CSV file
    columns = ["RouteNum", "G_nodes", "G_edges", "L_nodes", "L_edges", "Terminals_withTW", "Terminals", "BSTSTPTW_paths"]
    final_LSlog_file = pd.read_csv('./logs/final_LSlog_file.csv').sort_values(by="Terminals_withTW", ascending=False).iloc[:top_routes].reset_index(drop=True)

    # Generate the LaTeX table code
    latex_code = """
    \\begin{table}[H]
    \\centering
    \\caption{Local search performance of real-world instances. \\textit{TTW}: Number of terminals with non-trivial time-windows and values in parenthesis ($\\sizedualterminal$) represent the total number of terminals. \\textit{Initial $|\\localsearchndset|$}: Number of initial Pareto-optimal BSTSPTW tours, \\textit{Final $|\\localsearchndset|$}: Number of final Pareto-optimal BSTSPTW tours}
    \\label{tab:graphdet_heur}
    \\begin{tabular}{c|cc|cc|c|c|c}
    \\hline
    \\multirow{2}{*}{\\textbf{Route Id}} & \\multicolumn{2}{c|}{\\textbf{Original Graph}} & \\multicolumn{2}{c|}{\\textbf{Line Graph}} & \\multirow{2}{*}{\\textbf{TTW} \\boldsymbol{$(\\sizedualterminal)$}} & \\multirow{2}{*}{\\textbf{Initial \\boldsymbol{$|\\localsearchndset|$}}} & \\multirow{2}{*}{\\textbf{Final \\boldsymbol{$|\\localsearchndset|$}}} \\\\
     & \\boldsymbol{$|V|$} & \\boldsymbol{$|E|$} & \\boldsymbol{$|V^\\prime|$} & \\boldsymbol{$|E^\\prime|$} & & &\\\\ \\hline
    """

    # Fill the LaTeX table with data
    for _, row in final_LSlog_file.iterrows():
        row['G_nodes'] = str(row['G_nodes'])[:-3] + "\mys"+  str(row['G_nodes'])[-3:]
        row['G_edges'] = str(row['G_edges'])[:-3] + "\mys"+  str(row['G_edges'])[-3:]
        row['L_nodes'] = str(row['L_nodes'])[:-3] + "\mys"+  str(row['L_nodes'])[-3:]
        row['L_edges'] = str(row['L_edges'])[:-3] + "\mys"+  str(row['L_edges'])[-3:]
        latex_code += f"{row['RouteNum']} & {row['G_nodes']} & {row['G_edges']} & {row['L_nodes']} & {row['L_edges']} & {row['Terminals_withTW']} ({row['Terminals']}) & {row['BSTSTPTW_paths']} & {row['OnlyLS_paths']}\\\\\n"

    latex_code += """
    \\hline
    \\end{tabular}
    \\end{table}
    """
    print(latex_code)
    return None

def get_latex_table_9():
    df = pd.read_csv(r"./logs/summary/collective/NBD_paths_distribution.csv")
    latex_table = r"""
    \begin{table}[H]
        \centering
        \caption{Aggregate operators statistics (\textit{Total Tours}: Total tours found, \textit{TW\%}: Percentage of total tours that were BSTSPTW feasible, \textit{Count}: Number of times the operator was called, \textit{Mean Time}: Mean runtime per call in seconds)}
        \begin{tabular}{lrrrr}
        \hline
            \textbf{Operator} & \textbf{Total Tours} & \textbf{TW\%} & \textbf{Count} & \textbf{Mean Time} \\ \hline
    """

    # Iterate through the DataFrame and add each row to the LaTeX table
    for index, row in df.iterrows():
        if row['Method'] == "RM":
            row['Method'] = "RandomPermute"
        if row['Method'] == "GapRepair":
            row['Method'] = "RepairTW"
        if row['Method'] == "SimpleCycle":
            continue
        latex_table += f"        \\textsc{{{row['Method']}}} & {int(row['TotalPaths']):,} & {int(row['TW%'])} & {int(row['CallCount']):,} & {round(row['MeanRunTime'], 1)} \\\\\n"

    # End the LaTeX table
    latex_table += r"""
        \hline
        \end{tabular}
        \label{tab:nbdresults}
    \end{table}
    """
    print(latex_table)
    return None

sns.set_style("whitegrid")
warnings.filterwarnings("ignore", category=FutureWarning)

clear_collective_stats()
overall_s3opt_time_per, overall_s3opttw_time_per, overall_gr_time_per, overall_fp_time_per, overall_quad_time_per, overall_RM_per, overall_SimpleCycle_per = [], [], [], [], [], [], []
overall_udpateset_per = []
iteration_count = []
logs_folder = "./logs"
folder_path = logs_folder + "/summary"
top_routes = 9
all_routes = pd.read_csv(f"./lsInputs/working_amazon_routes.csv").Route_num.tolist()[:top_routes]
S3opt_details_all = pd.DataFrame()
S3optTW_details_all = pd.DataFrame()
GapRepair_details_all = pd.DataFrame()
FixedPerm_details_all = pd.DataFrame()
Quad_details_all = pd.DataFrame()
RM_details_all = pd.DataFrame()
SimpleCycle_details_all = pd.DataFrame()
updateSet_all = pd.DataFrame()
for rotue_num in all_routes:
    ls_final = pd.read_csv(f"{logs_folder}/{rotue_num}/ls_final.txt")
    if len(ls_final) == 0:
        overall_s3opt_time_per.append(0)
        overall_s3opttw_time_per.append(0)
        overall_gr_time_per.append(0)
        overall_fp_time_per.append(0)
        overall_quad_time_per.append(0)
        overall_SimpleCycle_per.append(0)
        overall_RM_per.append(0)
        iteration_count.append(0)
    else:
        ls_final["Total_Time"] = ls_final["Total_Time"] / ls_final["Total_Time"].sum() * 100
        overall_s3opt_time_per.append(ls_final[ls_final["Method"] == "S3opt"].Total_Time.iloc[0])
        overall_s3opttw_time_per.append(ls_final[ls_final["Method"] == "S3optTW"].Total_Time.iloc[0])
        overall_gr_time_per.append(ls_final[ls_final["Method"] == "GR"].Total_Time.iloc[0])
        overall_fp_time_per.append(ls_final[ls_final["Method"] == "FP"].Total_Time.iloc[0])
        overall_quad_time_per.append(ls_final[ls_final["Method"] == "Quad"].Total_Time.iloc[0])
        overall_RM_per.append(ls_final[ls_final["Method"] == "RP"].Total_Time.iloc[0])
        overall_SimpleCycle_per.append(ls_final[ls_final["Method"] == "CR"].Total_Time.iloc[0])
        iteration_count.append(ls_final[ls_final["Method"] == "Iteration"].TW_Count.iloc[0])

    S3opt_details = pd.read_csv(f"{logs_folder}/{rotue_num}/S3opt.txt")
    S3optTW_details = pd.read_csv(f"{logs_folder}/{rotue_num}/S3optTW.txt")
    GapRepair_details = pd.read_csv(f"{logs_folder}/{rotue_num}/GapRepair.txt")
    FixedPerm_details = pd.read_csv(f"{logs_folder}/{rotue_num}/FixedPerm.txt")
    Quad_details = pd.read_csv(f"{logs_folder}/{rotue_num}/Quad.txt")
    RandomPermute_details = pd.read_csv(f"{logs_folder}/{rotue_num}/RandomPermute.txt")
    SimpleCycle_details = pd.read_csv(f"{logs_folder}/{rotue_num}/SimpleCycle.txt")
    updateSet = pd.read_csv(f"{logs_folder}/{rotue_num}/updateSet.txt")

    S3opt_details_all = pd.concat([S3opt_details_all, S3opt_details], ignore_index=True)
    S3optTW_details_all = pd.concat([S3optTW_details_all, S3optTW_details], ignore_index=True)
    GapRepair_details_all = pd.concat([GapRepair_details_all, GapRepair_details], ignore_index=True)
    FixedPerm_details_all = pd.concat([FixedPerm_details_all, FixedPerm_details], ignore_index=True)
    Quad_details_all = pd.concat([Quad_details_all, Quad_details], ignore_index=True)
    RM_details_all = pd.concat([RM_details_all, RandomPermute_details], ignore_index=True)
    SimpleCycle_details_all = pd.concat([SimpleCycle_details_all, SimpleCycle_details], ignore_index=True)
    updateSet_all = pd.concat([updateSet_all, updateSet], ignore_index=True)
final_LSlog_file_all = pd.read_csv(f"{logs_folder}/final_LSlog_file.txt")
mixintprog = pd.read_csv(f"{logs_folder}/mixintprog.txt")
graph_stats = pd.read_csv(f"{logs_folder}/graph_stats.txt")
scalerization_end_condition = pd.read_csv(f"{logs_folder}/scalerizationEndConditions.txt")
#########################################################################################################
try:
    points = Counter(final_LSlog_file_all["OnlyLS_paths"].tolist()).items()
    plt.figure(figsize=(10, 5))
    plt.xticks(range(0, max([x[0] for x in points]) + 1, 10))
    # plt.(range(len(points)), [x[0] for x in points])
    plt.yticks(range(0, max([x[1] for x in points]) + 1, 2))
    plt.scatter([x[0] for x in points], [x[1] for x in points])
    plt.title("Number of only LS paths")
    plt.xlabel("Number of only LS paths")
    plt.ylabel("Number of routes")
    plt.grid(False)
    # plt.show()
    plt.savefig(f"{folder_path}/collective/OnlyLS_paths.pdf")
    plt.clf()
except:
    print("OnlyLS_paths failed")

try:
    stuck_rutes = set(all_routes) - set(final_LSlog_file_all["RouteNum"].tolist())
    stuck_moves = []
    for route in stuck_rutes:
        progressTrack = open(f"{logs_folder}/{route}/progressTrack.txt", "r")
        # Find last occurrence of "Entering" and track the line before it
        previous_line = ""
        for line in progressTrack:
            if "Entering" in line:
                move = previous_line
            previous_line = line
        move = move.strip("\n")
        stuck_moves.append(move)
    # Plot the bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(['S3opt', 'S3optTW', 'GapRepair', 'FixedPerm', 'Quad', "SimpleCycle", "RandomPermute"],
            [stuck_moves.count('S3opt'), stuck_moves.count('S3optTW'), stuck_moves.count('GapRepair'), stuck_moves.count('FixedPerm'), stuck_moves.count('Quad'), stuck_moves.count('SimpleCycle'),
             stuck_moves.count('RandomPermute')], color='skyblue', edgecolor='black')
    plt.title('Stuck Move Analysis')
    plt.xlabel('Move')
    plt.ylabel('Frequency')
    plt.savefig(f"{folder_path}/collective/StuckMoveAnalysis.pdf")
    # plt.show()
    plt.clf()
    if len(stuck_rutes) > 0:
        print("Stuck Routes: ", stuck_rutes)
    else:
        print("No Stuck Routes")
except Exception as e:
    print(e)
    print("StuckMoveAnalysis failed")
#####################################################g####################################################
try:
    rgrousp = scalerization_end_condition.groupby(["Route_num"])
    plt.figure(figsize=(10, 5))
    for _, r in rgrousp:
        plt.scatter(r["Route_num"].iloc[0], r["Condition"].iloc[0])
    plt.yticks(range(1, 7), ["None Argument", "Left=Right", "Timeout", "New None", "New=Right/Left", "Max Depth"])
    plt.ylabel("Condition")
    # plt.xticks(range(len(scalerization_end_condition)), scalerization_end_condition["Route_num"])
    plt.xlabel("Routes")
    plt.title("Scalrization End Conditions")
    plt.savefig(f"{folder_path}/collective/ScalerizationEndConditions.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("ScalerizationEndConditions failed")
    #########################################################################################################
    # try:
    #     num_rows = 22
    #     num_columns = 4
    #     fig, axs = plt.subplots(num_rows, num_columns, figsize=(90, 180))
    #     axs = axs.flatten()
    #     route_list = final_LSlog_file_all.sort_values(by=["OnlyLS_paths"]).RouteNum.tolist()
    #     for i in range(num_rows * num_columns):
    #         try:
    #             route_num = route_list[i]
    #         except IndexError:
    #             break
    #         bstsptw_tours, bstsp_tours = get_scalarization_results(route_num)
    #         (route_num, set_z, time_for_last_imp, move_counter, total_ls_time, set_p) = get_results_ls(route_num)
    #         if bstsptw_tours:
    #             bstsptw_turns, bstsptw_energy = zip(*[(t, e) for p, e, t in bstsptw_tours])
    #         else:
    #             bstsptw_turns, bstsptw_energy = [], []
    #         if bstsp_tours:
    #             bstsp_turns, bstsp_energy = zip(*[(t, e) for p, e, t in bstsp_tours])
    #         else:
    #             bstsp_turns, bstsp_energy = [], []
    #         if set_z:
    #             ls_turns, ls_energy = zip(*[(t, e) for p, e, t in set_z])
    #         else:
    #             ls_turns, ls_energy = [], []
    #         axs[i].plot(bstsp_turns, bstsp_energy, marker='o', linestyle='dotted', linewidth=3, color='orange', label=f"Initial BSTSP", zorder=1)
    #         axs[i].scatter(bstsptw_turns, bstsptw_energy, color='darkgreen', marker='o', facecolor='none', s=120, linewidth=1, label=f"Initial BSTSPTW", zorder=2)
    #         axs[i].plot(ls_turns, ls_energy, '*--', color='steelblue', label=f"LS", markersize=8, linewidth=3, zorder=3)
    #         axs[i].set_title(f"{route_num}", fontsize=14)
    #         axs[i].set_xlabel("Turn count", fontsize=14)
    #         axs[i].set_ylabel("Energy (kWh)", fontsize=14)
    #     plt.tight_layout()
    #     plt.savefig(f"{folder_path}/collective/LS_paths_withBSTSP.pdf")
    #     # plt.show()
    #     plt.clf()
    # except Exception as e:
    print(e)
#     print("LS_paths_withBSTSP failed")
try:
    num_rows = 22
    num_columns = 4
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(90, 180))
    axs = axs.flatten()
    route_list = final_LSlog_file_all.sort_values(by=["OnlyLS_paths"]).RouteNum.tolist()
    for i in range(num_rows * num_columns):
        try:
            route_num = route_list[i]
        except IndexError:
            break
        bstsptw_tours, bstsp_tours, _ = get_scalarization_results(route_num)
        (route_num, set_z, time_for_last_imp, move_counter, total_ls_time, set_p) = get_results_ls(route_num)
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
        # axs[i].plot(bstsp_turns, bstsp_energy, marker='o', linestyle='dotted', linewidth=3, color='orange', label=f"Initial BSTSP", zorder=1)
        axs[i].scatter(bstsptw_turns, bstsptw_energy, color='darkgreen', marker='o', facecolor='none', s=120, linewidth=1, label=f"Initial BSTSPTW", zorder=2)
        axs[i].plot(ls_turns, ls_energy, '*--', color='steelblue', label=f"LS", markersize=8, linewidth=3, zorder=3)
        axs[i].set_title(f"{route_num}", fontsize=14)
        axs[i].set_xlabel("Turn count", fontsize=14)
        axs[i].set_ylabel("Energy (kWh)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{folder_path}/collective/LS_paths.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("LS_paths failed")

try:
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(90, 180))
    axs = axs.flatten()
    color_dict = {'S3opt': "g", 'S3optTW': "m", 'GapRepair': "y", 'FixedPerm': "c", 'Quad': "k", "SimpleCycle": "r", "RandomPermute": "b"}
    marker_dict = {'S3opt': "v", 'S3optTW': "s", 'GapRepair': "D", 'FixedPerm': "o", 'Quad': "P", "SimpleCycle": "X", "RandomPermute": "x"}
    for i in range(num_rows * num_columns):
        try:
            route_num = route_list[i]
        except IndexError:
            break
        bstsptw_tours, bstsp_tours, _ = get_scalarization_results(route_num)
        (route_num, set_z, time_for_last_imp, move_counter, total_ls_time, set_p) = get_results_ls(route_num)
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

        # PLot the bar plot
        axs[i].bar(['S3opt', 'S3optTW', 'GapRepair', 'FixedPerm', 'Quad', "SimpleCycle", "RandomPermute"],
                   [move_counter['S3opt'], move_counter['S3optTW'], move_counter['GapRepair'], move_counter['FixedPerm'], move_counter['Quad'], move_counter['SimpleCycle'],
                    move_counter['RandomPermute']],
                   color='skyblue', edgecolor='black')
        axs[i].set_title(f"{route_num}", fontsize=14)
        axs[i].set_xlabel('Move')
        axs[i].set_ylabel('Frequency')
    plt.title('Move Analysis')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{folder_path}/collective/MoveAnalysis.pdf")
    plt.clf()
except Exception as e:
    print(e)
    print("MoveAnalysis failed")

#########################################################################################################
try:
    fig, ax = plt.subplots()
    scatter_points = pd.merge(mixintprog, graph_stats, left_on="RouteNum", right_on="Route_num")[["Dijkstra", "DualEdges"]].values
    plt.scatter(scatter_points[:, 0], scatter_points[:, 1])
    plt.xlabel("Dijkstra time (seconds)")
    plt.ylabel("DualEdges Count")
    plt.title("Dijkstra Runtime vs. DualEdges")
    plt.savefig(f"{folder_path}/collective/Dijkstra_vs_DualEdges.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("Dijkstra_vs_DualEdges failed")

######################################################################################################
try:
    temp = mixintprog.sort_values(by=["JhonsanTime"])
    fig, ax = plt.subplots()
    plt.plot(range(len(temp)), [final_LSlog_file_all.AllowedInitTime] * len(temp), color="black", alpha=0.5)
    plt.plot(range(len(temp)), temp.GoogleOR, color="green", alpha=0.5)
    plt.plot(range(len(temp)), temp.JhonsanTime, color="red", alpha=0.5)
    plt.plot(range(len(temp)), temp.Bellman, color="orange", alpha=0.5)
    plt.plot(range(len(temp)), temp.Dijkstra, color="blue", alpha=0.5)

    ax.set_xlabel('Mixint call')
    ax.set_ylabel('Time in seconds')
    ax.set_title('Time split for Mixint function')
    # plt.legend()
    legend_labels = ["Total time", "GoogleOR", "JhonsanTime", "Bellman", "Dijkstra"]
    legend_colors = ["black", "green", "red", "orange", "blue"]
    plt.legend(labels=legend_labels, handles=[
        plt.Line2D([0], [0], marker='o', color=color, label=label) for label, color in zip(legend_labels, legend_colors)
    ])
    plt.savefig(f"{folder_path}/collective/Scalerization_mixint_time_split.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("Scalerization_mixint_time_split failed")

######################################################################################################
try:
    temp = final_LSlog_file_all.sort_values(by=["RouteNum"])
    bar_labels = list(range(1, len(temp) + 1))
    stack_labels = ['BSTSP', 'BSTSPTW', 'LS']
    stack_colors = ['salmon', 'orangered', 'orange']
    fig, ax = plt.subplots()
    plt.bar(bar_labels, temp.BSTSTP_paths, color=stack_colors[0])
    plt.bar(bar_labels, temp.BSTSTPTW_paths, bottom=temp.BSTSTP_paths, color=stack_colors[1])
    plt.bar(bar_labels, temp.OnlyLS_paths, bottom=temp.BSTSTP_paths + temp.BSTSTPTW_paths, color=stack_colors[2])
    ax.set_xlabel('Route numbers')
    ax.set_ylabel('Number of routes')
    ax.set_title('Final routes (Scalarization vs. LS)')
    plt.legend(stack_labels)
    plt.savefig(f"{folder_path}/collective/FinalRoutes_Split.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("FinalRoutes_Split failed")
######################################################################################################
try:
    temp = final_LSlog_file_all.sort_values(by=["TimeForLastImp"]).reset_index(drop=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(temp)), temp.TimeForLastImp, color="blue", alpha=0.5)
    plt.plot(range(len(temp)), temp.TotalTime, color="red", alpha=0.5)
    plt.xlabel("Routes")
    plt.ylabel("Time taken (seconds)")
    plt.title(f"Last time when the new path was found")
    plt.grid(False)
    plt.legend(["TimeForLastImp", "TotalTime"])
    plt.savefig(f"{folder_path}/collective/LastImprovementTime.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("LastImprovementTime failed")
######################################################################################################
try:
    allpaths = S3opt_details_all["PathsFound"].sum()
    if allpaths == 0:
        S3optParetoPaths = 0
        S3optNonParetoPaths = 0
        S3optExplorePaths = 0
        S3optNonExplorePaths = 0
    else:
        S3optParetoPaths = S3opt_details_all["ParetoPaths"].sum() / allpaths * 100
        S3optNonParetoPaths = S3opt_details_all["NonParetoPaths"].sum() / allpaths * 100
        S3optExplorePaths = S3opt_details_all["ExplorePaths"].sum() / allpaths * 100
        S3optNonExplorePaths = S3opt_details_all["NonExplorePaths"].sum() / allpaths * 100

    allpaths = S3optTW_details_all["PathsFound"].sum()
    if allpaths == 0:
        S3optTWParetoPaths = 0
        S3optTWNonParetoPaths = 0
        S3optTWExplorePaths = 0
        S3optTWNonExplorePaths = 0
    else:
        S3optTWParetoPaths = S3optTW_details_all["ParetoPaths"].sum() / allpaths * 100
        S3optTWNonParetoPaths = S3optTW_details_all["NonParetoPaths"].sum() / allpaths * 100
        S3optTWExplorePaths = S3optTW_details_all["ExplorePaths"].sum() / allpaths * 100
        S3optTWNonExplorePaths = S3optTW_details_all["NonExplorePaths"].sum() / allpaths * 100

    allpaths = GapRepair_details_all["PathsFound"].sum()
    if allpaths == 0:
        GapRepairParetoPaths = 0
        GapRepairNonParetoPaths = 0
        GapRepairExplorePaths = 0
        GapRepairNonExplorePaths = 0
    else:
        GapRepairParetoPaths = GapRepair_details_all["ParetoPaths"].sum() / allpaths * 100
        GapRepairNonParetoPaths = GapRepair_details_all["NonParetoPaths"].sum() / allpaths * 100
        GapRepairExplorePaths = GapRepair_details_all["ExplorePaths"].sum() / allpaths * 100
        GapRepairNonExplorePaths = GapRepair_details_all["NonExplorePaths"].sum() / allpaths * 100

    allpaths = FixedPerm_details_all["PathsFound"].sum()
    if allpaths == 0:
        FixedPermParetoPaths = 0
        FixedPermNonParetoPaths = 0
        FixedPermExplorePaths = 0
        FixedPermNonExplorePaths = 0
    else:
        FixedPermParetoPaths = FixedPerm_details_all["ParetoPaths"].sum() / allpaths * 100
        FixedPermNonParetoPaths = FixedPerm_details_all["NonParetoPaths"].sum() / allpaths * 100
        FixedPermExplorePaths = FixedPerm_details_all["ExplorePaths"].sum() / allpaths * 100
        FixedPermNonExplorePaths = FixedPerm_details_all["NonExplorePaths"].sum() / allpaths * 100

    allpaths = Quad_details_all["PathsFound"].sum()
    if allpaths == 0:
        QuadParetoPaths = 0
        QuadNonParetoPaths = 0
        QuadExplorePaths = 0
        QuadNonExplorePaths = 0
    else:
        QuadParetoPaths = Quad_details_all["ParetoPaths"].sum() / allpaths * 100
        QuadNonParetoPaths = Quad_details_all["NonParetoPaths"].sum() / allpaths * 100
        QuadExplorePaths = Quad_details_all["ExplorePaths"].sum() / allpaths * 100
        QuadNonExplorePaths = Quad_details_all["NonExplorePaths"].sum() / allpaths * 100

    allpaths = RM_details_all["PathsFound"].sum()
    if allpaths == 0:
        RMParetoPaths = 0
        RMNonParetoPaths = 0
        RMExplorePaths = 0
        RMNonExplorePaths = 0
    else:
        RMParetoPaths = RM_details_all["ParetoPaths"].sum() / allpaths * 100
        RMNonParetoPaths = RM_details_all["NonParetoPaths"].sum() / allpaths * 100
        RMExplorePaths = RM_details_all["ExplorePaths"].sum() / allpaths * 100
        RMNonExplorePaths = RM_details_all["NonExplorePaths"].sum() / allpaths * 100

    allpaths = SimpleCycle_details_all["PathsFound"].sum()
    if allpaths == 0:
        SimpleCycleParetoPaths = 0
        SimpleCycleNonParetoPaths = 0
        SimpleCycleExplorePaths = 0
        SimpleCycleNonExplorePaths = 0
    else:
        SimpleCycleParetoPaths = SimpleCycle_details_all["ParetoPaths"].sum() / allpaths * 100
        SimpleCycleNonParetoPaths = SimpleCycle_details_all["NonParetoPaths"].sum() / allpaths * 100
        SimpleCycleExplorePaths = SimpleCycle_details_all["ExplorePaths"].sum() / allpaths * 100
        SimpleCycleNonExplorePaths = SimpleCycle_details_all["NonExplorePaths"].sum() / allpaths * 100

    ParetoPaths_stack = np.array([S3optParetoPaths, S3optTWParetoPaths, GapRepairParetoPaths, FixedPermParetoPaths, QuadParetoPaths, RMParetoPaths, SimpleCycleParetoPaths])
    NonParetoPaths_stack = np.array([S3optNonParetoPaths, S3optTWNonParetoPaths, GapRepairNonParetoPaths, FixedPermNonParetoPaths, QuadNonParetoPaths, RMNonParetoPaths, SimpleCycleNonParetoPaths])
    ExplorePaths_stack = np.array([S3optExplorePaths, S3optTWExplorePaths, GapRepairExplorePaths, FixedPermExplorePaths, QuadExplorePaths, RMExplorePaths, SimpleCycleExplorePaths])
    NonExplorePaths_stack = np.array(
        [S3optNonExplorePaths, S3optTWNonExplorePaths, GapRepairNonExplorePaths, FixedPermNonExplorePaths, QuadNonExplorePaths, RMNonExplorePaths, SimpleCycleNonExplorePaths])

    bar_labels = ['S3opt', 'S3optTW', 'GR', 'FP', 'Quad', "RM", "CR"]
    stack_colors = ['salmon', 'orangered', 'orange', 'wheat', 'lightgrey', "lightblue", "lightgreen"]
    plt.clf()
    fig, ax = plt.subplots()
    plt.bar(bar_labels, ParetoPaths_stack, color=stack_colors[0])
    plt.bar(bar_labels, NonParetoPaths_stack, bottom=ParetoPaths_stack, color=stack_colors[1])
    plt.bar(bar_labels, ExplorePaths_stack, bottom=ParetoPaths_stack + NonParetoPaths_stack, color=stack_colors[2])
    plt.bar(bar_labels, NonExplorePaths_stack, bottom=ParetoPaths_stack + NonParetoPaths_stack + ExplorePaths_stack, color=stack_colors[3])

    ax.set_xlabel('Neighbourhoods')
    ax.set_ylabel('Percentage of Paths')
    ax.set_title('Percentage of paths obtained by each neighbourhood')
    plt.legend(["ParetoPaths", "NonParetoPaths", "ExplorePaths", "NonExplorePaths"])
    plt.savefig(f"{folder_path}/collective/NBD_paths_distribution.pdf")
    # plt.show()
    plt.clf()

    quad_pareto = Quad_details_all["ParetoPaths"].sum()
    fpm_pareto = FixedPerm_details_all["ParetoPaths"].sum()
    gr_pareto = GapRepair_details_all["ParetoPaths"].sum()
    s3opttw_pareto = S3optTW_details_all["ParetoPaths"].sum()
    s3op_pareto = S3opt_details_all["ParetoPaths"].sum()
    rm_pareto = RM_details_all["ParetoPaths"].sum()
    cr_pareto = SimpleCycle_details_all["ParetoPaths"].sum()

    quad_nonpareto = Quad_details_all["NonParetoPaths"].sum()
    fpm_nonpareto = FixedPerm_details_all["NonParetoPaths"].sum()
    gr_nonpareto = GapRepair_details_all["NonParetoPaths"].sum()
    s3opttw_nonpareto = S3optTW_details_all["NonParetoPaths"].sum()
    s3op_nonpareto = S3opt_details_all["NonParetoPaths"].sum()
    rm_nonpareto = RM_details_all["NonParetoPaths"].sum()
    cr_nonpareto = SimpleCycle_details_all["NonParetoPaths"].sum()

    quad_explore = Quad_details_all["ExplorePaths"].sum()
    fpm_explore = FixedPerm_details_all["ExplorePaths"].sum()
    gr_explore = GapRepair_details_all["ExplorePaths"].sum()
    s3opttw_explore = S3optTW_details_all["ExplorePaths"].sum()
    s3op_explore = S3opt_details_all["ExplorePaths"].sum()
    rm_explore = RM_details_all["ExplorePaths"].sum()
    cr_explore = SimpleCycle_details_all["ExplorePaths"].sum()

    quad_nonexplore = Quad_details_all["NonExplorePaths"].sum()
    fpm_nonexplore = FixedPerm_details_all["NonExplorePaths"].sum()
    gr_nonexplore = GapRepair_details_all["NonExplorePaths"].sum()
    s3opttw_nonexplore = S3optTW_details_all["NonExplorePaths"].sum()
    s3op_nonexplore = S3opt_details_all["NonExplorePaths"].sum()
    rm_nonexplore = RM_details_all["NonExplorePaths"].sum()
    cr_nonexplore = SimpleCycle_details_all["NonExplorePaths"].sum()

    quad_total = quad_pareto + quad_nonpareto + quad_explore + quad_nonexplore
    fpm_total = fpm_pareto + fpm_nonpareto + fpm_explore + fpm_nonexplore
    gr_total = gr_pareto + gr_nonpareto + gr_explore + gr_nonexplore
    s3opttw_total = s3opttw_pareto + s3opttw_nonpareto + s3opttw_explore + s3opttw_nonexplore
    s3op_total = s3op_pareto + s3op_nonpareto + s3op_explore + s3op_nonexplore
    rm_total = rm_pareto + rm_nonpareto + rm_explore + rm_nonexplore
    cr_total = cr_pareto + cr_nonpareto + cr_explore + cr_nonexplore

    # S3opt_details_all_stats = S3opt_details_all.Time.describe().round()
    # S3optTW_details_all_stats = S3optTW_details_all.Time.describe().round()
    # GapRepair_details_all_stats = GapRepair_details_all.Time.describe().round()
    # FixedPerm_details_all_stats = FixedPerm_details_all.Time.describe().round()
    # Quad_details_all_stats = Quad_details_all.Time.describe().round()
    # RM_details_all_stats = RM_details_all.Time.describe().round()
    # SimpleCycle_details_all_stats = SimpleCycle_details_all.Time.describe().round()
    S3opt_details_all_stats = S3opt_details_all.Time.describe()
    S3optTW_details_all_stats = S3optTW_details_all.Time.describe()
    GapRepair_details_all_stats = GapRepair_details_all.Time.describe()
    FixedPerm_details_all_stats = FixedPerm_details_all.Time.describe()
    Quad_details_all_stats = Quad_details_all.Time.describe()
    RM_details_all_stats = RM_details_all.Time.describe()
    SimpleCycle_details_all_stats = SimpleCycle_details_all.Time.describe()

    temp = (pd.DataFrame([
        ["S3opt", s3op_pareto, s3op_nonpareto, s3op_explore, s3op_nonexplore, s3op_total, len(S3opt_details_all), S3opt_details_all_stats["mean"], S3opt_details_all_stats["std"],
         S3opt_details_all_stats["max"]],
        ["S3optTW", s3opttw_pareto, s3opttw_nonpareto, s3opttw_explore, s3opttw_nonexplore, s3opttw_total, len(S3optTW_details_all), S3optTW_details_all_stats["mean"],
         S3optTW_details_all_stats["std"], S3optTW_details_all_stats["max"]],
        ["GapRepair", gr_pareto, gr_nonpareto, gr_explore, gr_nonexplore, gr_total, len(GapRepair_details_all), GapRepair_details_all_stats["mean"], GapRepair_details_all_stats["std"],
         GapRepair_details_all_stats["max"]],
        ["FixedPerm", fpm_pareto, fpm_nonpareto, fpm_explore, fpm_nonexplore, fpm_total, len(FixedPerm_details_all), FixedPerm_details_all_stats["mean"], FixedPerm_details_all_stats["std"],
         FixedPerm_details_all_stats["max"]],
        ["SimpleCycle", cr_pareto, cr_nonpareto, cr_explore, cr_nonexplore, cr_total, len(SimpleCycle_details_all), SimpleCycle_details_all_stats["mean"], SimpleCycle_details_all_stats["std"],
         SimpleCycle_details_all_stats["max"]],
        ["Quad", quad_pareto, quad_nonpareto, quad_explore, quad_nonexplore, quad_total, len(Quad_details_all), Quad_details_all_stats["mean"], Quad_details_all_stats["std"],
         Quad_details_all_stats["max"]],
        ["RM", rm_pareto, rm_nonpareto, rm_explore, rm_nonexplore, rm_total, len(RM_details_all), RM_details_all_stats["mean"], RM_details_all_stats["std"],
         RM_details_all_stats["max"]],
    ], columns=["Method", "ParetoPaths", "NonParetoPaths", "ExplorePaths", "NonExplorePaths", "TotalPaths", "CallCount", "MeanRunTime", "StdRunTime", "MaxRuntime"]))
    temp["TW%"] = ((temp["ParetoPaths"] + temp["NonParetoPaths"]) / temp["TotalPaths"] * 100).round(1)
    temp = temp.fillna(0)
    temp.to_csv(f"{folder_path}/collective/NBD_paths_distribution.csv", index=False)
except Exception as e:
    print(e)
    print("NBD_paths_distribution failed")
file = pd.read_csv(fr"{folder_path}/collective/NBD_paths_distribution.csv").reindex([0, 1, 2, 3, 6, 4, 5]).reset_index(drop=True)
file.to_csv(fr"{folder_path}/collective/NBD_paths_distribution.csv", index=False)
try:
    plt.figure(figsize=(10, 5))
    for x in overall_s3opt_time_per:
        plt.scatter(1, x, color="red")
    for x in overall_s3opttw_time_per:
        plt.scatter(2, x, color="green")
    for x in overall_gr_time_per:
        plt.scatter(3, x, color="blue")
    for x in overall_fp_time_per:
        plt.scatter(4, x, color="yellow")
    for x in overall_quad_time_per:
        plt.scatter(5, x, color="black")
    for x in overall_RM_per:
        plt.scatter(6, x, color="pink")
    for x in overall_SimpleCycle_per:
        plt.scatter(7, x, color="purple")
    plt.xticks([1, 2, 3, 4, 5, 6, 7], ["S3opt", "S3optTW", "GapRepair", "FixedPerm", "Quad", "RandomPermute", "SimpleCycle"])
    plt.ylabel("Time Percentage")
    plt.xlabel("Method")
    plt.grid(False)
    plt.title("Percentage of time used by of each method w.r.t total time")
    plt.savefig(f"{folder_path}/collective/NBD_time_percentage.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("NBD_time_percentage failed")

try:
    # Generate a histogram
    plt.figure(figsize=(10, 5))
    a, b, c = plt.hist(iteration_count, color="red", alpha=0.5, bins=range(1, max(iteration_count) + 5, 2))
    plt.xlabel("Number of Iterations")
    plt.ylabel("Number of Routes")
    # plt.yticks(range(0, int(max(a) + 10), 5))
    plt.title("Number of LS Iterations")
    plt.xticks(range(1, max(iteration_count) + 1, 2))
    plt.savefig(f"{folder_path}/collective/LS_iteration_count.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("LS_iteration_count failed")

#####################################################################################
# S3opt plots
try:
    # plot a histogram plot of s3opt_step1
    plt.figure(figsize=(10, 5))
    plt.hist(S3opt_details_all["Step1Time"], color="red", alpha=0.5, bins=(range(0, 101, 10)))
    plt.xlabel("% of time taken by Step 1")
    plt.xticks(range(0, 101, 10))
    plt.ylabel("Number of Calls")
    plt.title(f"Percentage of time taken by Step 1 w.r.t total time for S3opt. Total Calls: {len(S3opt_details_all)}")
    plt.savefig(f"{folder_path}/S3opt/S3opt_step1.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("S3opt_step1 failed")

try:
    plt.figure(figsize=(10, 5))
    plt.hist(S3opt_details_all["Step2Time"], color="red", alpha=0.5, bins=range(0, 101, 10))
    plt.xlabel("% of time taken by Step 2")
    plt.xticks(range(0, 101, 10))
    plt.ylabel("Number of Calls")
    plt.title(f"Percentage of time taken by Step 2 w.r.t total time for S3opt. Total Calls: {len(S3opt_details_all)}")
    plt.savefig(f"{folder_path}/S3opt/S3opt_step2.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("S3opt_step2 failed")

try:
    # Plot a histogram of reduction
    plt.figure(figsize=(10, 5))
    reduction_factors = [x for x in S3opt_details_all.Reduction.tolist() if type(x) == int]
    plt.hist(reduction_factors, color="red", alpha=0.5)
    plt.xlabel("Percentage reduction")
    plt.ylabel("Number of Calls")
    plt.title(f"Percentage Reduction of Candidate set in S3opt. Total Calls: {len(S3opt_details_all)}")
    plt.savefig(f"{folder_path}/S3opt/S3opt_reduction.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("S3opt_reduction failed")
try:
    # Plot a histogram of time for S3opt
    plt.figure(figsize=(10, 5))
    plt.hist(S3opt_details_all.Time, color="red", alpha=0.5, bins=range(0, int(S3opt_details_all.Time.max()) + 10, 100))
    plt.xlabel("Time taken (seconds)")
    plt.ylabel("Number of Calls")
    plt.title(f"Time taken by S3opt in seconds. Total Calls: {len(S3opt_details_all)}")
    plt.savefig(f"{folder_path}/S3opt/S3opt_time.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("S3opt_time failed")
try:
    # Plot a histogram of time for S3opt
    plt.figure(figsize=(10, 5))
    plt.hist(S3opt_details_all.Alpha, color="red", alpha=0.5)
    plt.xlabel("Alpha")
    plt.ylabel("Number of Calls")
    plt.title(f"Histogram for values of Alpha in S3opt. Total Calls: {len(S3opt_details_all)}")
    plt.savefig(f"{folder_path}/S3opt/S3opt_alpha.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("S3opt_alpha failed")
try:
    plt.figure(figsize=(10, 5))
    plt.hist(S3opt_details_all.PathsFound, color="red", alpha=0.5)
    plt.xlabel("Number of paths found")
    plt.ylabel("Number of Calls")
    plt.title(f"Number of paths found in S3opt Move. Total Calls: {len(S3opt_details_all)}")
    plt.savefig(f"{folder_path}/S3opt/S3opt_pathsFound.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("S3opt_pathsFound failed")

#####################################################################################
# S3opttw plots
try:
    plt.figure(figsize=(10, 5))
    plt.hist(S3optTW_details_all["Step1Time"].tolist(), color="red", alpha=0.5, bins=range(0, 101, 10))
    plt.xlabel("% of time taken by Step 1 in S3optTW")
    plt.xticks(range(0, 101, 10))
    plt.ylabel("Number of Calls")
    plt.title(f"Percentage of time taken by Step 1 w.r.t total time for S3optTW. Total Calls: {len(S3optTW_details_all)}")
    plt.savefig(f"{folder_path}/S3optTW/S3optTW_step1.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("S3optTW_step1 failed")
try:
    plt.figure(figsize=(10, 5))
    plt.hist(S3optTW_details_all["Step2Time"].tolist(), color="red", alpha=0.5, bins=range(0, 101, 10))
    plt.xlabel("% of time taken by Step 2 in S3optTW")
    plt.xticks(range(0, 101, 10))
    plt.ylabel("Number of Calls")
    plt.title(f"Percentage of time taken by Step 2 w.r.t total time for S3optTW. Total Calls: {len(S3optTW_details_all)}")
    plt.savefig(f"{folder_path}/S3optTW/S3optTW_step2.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("S3optTW_step2 failed")
try:
    plt.figure(figsize=(10, 5))
    plt.hist(S3optTW_details_all.Time, color="red", alpha=0.5)
    plt.xlabel("Time taken (seconds) in S3optTW")
    plt.ylabel("Number of Calls")
    plt.title(f"Time taken by S3optTW in seconds. Total Calls: {len(S3optTW_details_all)}")
    plt.savefig(f"{folder_path}/S3optTW/S3optTW_time.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("S3optTW_time failed")
try:
    plt.figure(figsize=(10, 5))
    plt.hist(S3optTW_details_all.Alpha, color="red", alpha=0.5)
    plt.xlabel("Alpha")
    plt.ylabel("Number of Calls")
    plt.title(f"Histogram for values of Alpha in S3optTW. Total Calls: {len(S3optTW_details_all)}")
    plt.savefig(f"{folder_path}/S3optTW/S3optTW_alpha.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("S3optTW_alpha failed")

try:
    plt.figure(figsize=(10, 5))
    plt.hist(S3optTW_details_all.PathsFound, color="red", alpha=0.5, bins=2)
    plt.xlabel("Number of paths found")
    plt.ylabel("Number of Calls")
    plt.title(f"Number of paths found in S3optTW Move. Total Calls: {len(S3optTW_details_all)}")
    plt.savefig(f"{folder_path}/S3optTW/S3optTW_pathsFound.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("S3optTW_pathsFound failed")

#####################################################################################
# GapRepair plots
try:
    a = Counter(GapRepair_details_all.ReturnFlag)
    labels = ["Cycle Q was BSTSPTW", "Repair Path was BSTSP", "Success"]
    sizes = [a[1] / sum(a.values()), a[2] / sum(a.values()), a[3] / sum(a.values())]
    colors = ['skyblue', 'salmon', 'green']
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Gap Repair Return Flag Distribution')
    plt.savefig(f"{folder_path}/GapRepair/GR_return_flag_distribution.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("GR_return_flag_distribution failed")
try:
    plt.figure(figsize=(10, 5))
    plt.hist(GapRepair_details_all.Alpha, color="red", alpha=0.5)
    plt.xlabel("Alpha")
    plt.ylabel("Number of Calls")
    plt.title(f"Histogram for values of Alpha in GR. Total Calls: {len(GapRepair_details_all)}")
    plt.savefig(f"{folder_path}/GapRepair/GR_alpha.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("GR_alpha failed")
try:
    GapRepair_details_success = GapRepair_details_all[GapRepair_details_all.ReturnFlag == 3]
    plt.figure(figsize=(10, 5))
    plt.hist(GapRepair_details_success.Time, color="red", alpha=0.5)
    plt.xlabel("Time taken (seconds)")
    plt.ylabel("Number of Calls")
    plt.title(f"Time taken by Successful GR (seconds). Total Calls: {len(GapRepair_details_success)}")
    plt.savefig(f"{folder_path}/GapRepair/GR_time.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("GR_time failed")
try:
    plt.figure(figsize=(10, 5))
    plt.hist(GapRepair_details_success.MaxWidth, color="red", alpha=0.5)
    plt.xlabel("Max Width")
    plt.ylabel("Number of Calls")
    plt.title(f"Histogram for values of Max Width in Successful GR. Total Calls: {len(GapRepair_details_success)}")
    plt.savefig(f"{folder_path}/GapRepair/GR_maxWidth.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("Max Width plot failed")
try:
    plt.figure(figsize=(10, 5))
    plt.hist(GapRepair_details_success.DestroyTimePer, color="red", alpha=0.5)
    plt.xlabel("Percentage of time required by destroy part")
    plt.ylabel("Number of Calls")
    plt.title(f"Histogram for values of % of destroy time in Successful GR. Total Calls: {len(GapRepair_details_success)}")
    plt.savefig(f"{folder_path}/GapRepair/destroy_percent_hist.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("Destroy time plot failed")
try:
    QuickestTWFlag = Counter(GapRepair_details_success.QuickestTWFlag.astype(bool).tolist())
    WeightTWFlag = Counter(GapRepair_details_success.WeightTWFlag.astype(bool).tolist())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sizes = [QuickestTWFlag[True] / sum(QuickestTWFlag.values()), QuickestTWFlag[False] / sum(QuickestTWFlag.values())]
    colors = ['salmon', 'mistyrose']
    ax1.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=140)
    ax1.axis('equal')
    ax1.set_title('Quickest Path Flag Distribution')
    ax1.legend(["TW Feasible %", "Non TW Feasible %"])
    sizes = [WeightTWFlag[True] / sum(WeightTWFlag.values()), WeightTWFlag[False] / sum(WeightTWFlag.values())]
    ax2.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=140)
    ax2.axis('equal')
    ax2.set_title('Weight Path Flag Distribution')
    ax2.legend(["TW Feasible %", "Non TW Feasible %"])
    plt.savefig(f"{folder_path}/GapRepair/GR_flag_distribution.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("GapRepair Flag plot failed")

#####################################################################################
# FixedPerm plots
try:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    # Plotting T_flipped histogram
    ax1.hist(FixedPerm_details_all['T_flipped'], bins=10, color='blue', alpha=0.7)
    ax1.set_title('T_flipped')
    ax1.set_xlabel('% of T_flipped terminal pairs')
    ax1.set_xticks(np.arange(0, 101, 10))
    ax1.set_ylabel('Number of FPM calls')
    # Plotting T_fetched histogram
    ax2.hist(FixedPerm_details_all['T_fetched'], bins=10, color='green', alpha=0.7)
    ax2.set_title('T_fetched')
    ax2.set_xlabel('% of T_fetched terminal pairs')
    ax2.set_xticks(np.arange(0, 101, 10))
    ax2.set_ylabel('Number of FPM calls')
    # Plotting T_timeout histogram
    ax3.hist(FixedPerm_details_all['T_timeout'], bins=10, color='red', alpha=0.7)
    ax3.set_title('T_timeout')
    ax3.set_xlabel('% of T_timeout terminal pairs')
    ax3.set_xticks(np.arange(0, 101, 10))
    ax3.set_ylabel('Number of FPM calls')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(f"{folder_path}/FixedPerm/FP_terminal_type.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("FixedPerm Terminal type plot failed")
try:
    plt.figure(figsize=(10, 5))
    a, b, c = plt.hist(FixedPerm_details_all.PathsFound, color="red", alpha=0.5, bins=50)
    plt.xlabel("Number of paths found")
    plt.ylabel("Number of Calls")
    plt.title(f"Number of paths found in FPM Move. Total Calls: {len(FixedPerm_details_all)}")
    plt.savefig(f"{folder_path}/FixedPerm/FP_pathsFound.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("FixedPerm PathsFound plot failed")
try:
    a = Counter(FixedPerm_details_all.Obj)
    labels = 'Turns', 'Energy'
    sizes = [a["turns"] / sum(a.values()), a["energy"] / sum(a.values())]
    colors = ['mistyrose', 'salmon']
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('FPM Objective Distribution')
    plt.savefig(f"{folder_path}/FixedPerm/FP_obj_distribution.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("FixedPerm Obj plot failed")

try:
    plt.figure(figsize=(10, 5))
    plt.hist(FixedPerm_details_all.Time, color="red", alpha=0.5)
    plt.xlabel("Time taken (seconds)")
    plt.ylabel("Number of Calls")
    plt.title(f"Time taken by FixedPerm in seconds. Total Calls: {len(FixedPerm_details_all)}")
    plt.grid(False)
    plt.savefig(f"{folder_path}/FixedPerm/FP_time.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("FixedPerm Time plot failed")

#####################################################################################
# Quad plots
try:
    flags = Quad_details_all["Flag27"].tolist() + Quad_details_all["Flag05"].tolist() + Quad_details_all["Flag63"].tolist() + Quad_details_all["Flag41"].tolist()
    a = Counter(flags)
    labels = ['Biobjective', 'Precomputed']
    sizes = [a["Biobjective"] / sum(a.values()), a["Precomputed"] / sum(a.values())]
    colors = ['mistyrose', 'salmon']
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=100)
    plt.axis('equal')
    plt.title('Quad Flag Distribution')
    plt.savefig(f"{folder_path}/Quad/Quad_flag_distribution.pdf")
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("Quad Flag plot failed")

try:
    plt.figure(figsize=(10, 5))
    plt.hist(Quad_details_all.Time, color="red", alpha=0.5, bins=range(0, int(Quad_details_all.Time.max()) + 10, 10))
    plt.xlabel("Time taken (seconds)")
    plt.xticks(range(0, int(Quad_details_all.Time.max()) + 10, 10))
    plt.ylabel("Number of Calls")
    plt.title(f"Time taken by Quad in seconds. Total Calls: {len(Quad_details_all)}")
    plt.savefig(f"{folder_path}/Quad/Quad_time.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("Quad Time plot failed")

try:
    plt.figure(figsize=(10, 5))
    plt.hist(Quad_details_all.PathsFound, color="red", alpha=0.5, bins=range(0, int(Quad_details_all.PathsFound.max()) + 1, 10))
    plt.xlabel("Number of paths found")
    plt.xticks(range(0, int(Quad_details_all.PathsFound.max()) + 10, 10))
    plt.ylabel("Number of Calls")
    plt.title(f"Number of paths found in Quad Move. Total Calls: {len(Quad_details_all)}")
    plt.savefig(f"{folder_path}/Quad/Quad_pathsFound.pdf")
    plt.grid(False)
    # plt.show()
    plt.clf()
except Exception as e:
    print(e)
    print("Quad PathsFound plot failed")

#####################################################################################
# try:
#     epsilon = 5
#     (route_num, set_z, time_for_last_imp, move_counter, total_ls_time, set_p) = get_results_ls(route_num)
#     ls_turns, ls_energy = zip(*[(t, e) for p, e, t in set_z])
#     nbd_flags = {"S3opt": True, "S3optTW": True, "GapRepair": True, "FixedPerm": True, "Quad": True, "SimpleCycle": True, "RandomPermute": True}
#     tw_feasible_energies  = []
#     tw_feasible_turns = []
#     for nbd in nbd_flags.keys():
#         if not nbd_flags[nbd]: continue
#         tw_feasible_energies = [e for p, e, t in nbd_tw_feasible_tours[nbd]]
#         tw_feasible_turns = [t for p, e, t in nbd_tw_feasible_tours[nbd]]
#
#     max_allowed_turns = (1+epsilon/100)*max(ls_turns)
#     max_allowed_energy = (1+epsilon/100)*max(ls_energy)
#
#     filtered_tw_feasible_energy = [e for e in tw_feasible_energies if e <= max_allowed_energy]
#     filtered_tw_deasible_turns = [t for t in tw_feasible_turns if t <= max_allowed_turns]
#
#     plt.plot(ls_turns, ls_energy, 'o', color='blue')
#     plt.scatter(filtered_tw_deasible_turns, filtered_tw_feasible_energy, color='red', alpha=0.5)
#     plt.xlabel("Turns")
#     plt.ylabel("Energy")
#     plt.title(f"5% Pareto Front for route {route_num}")
#     plt.savefig(f"{folder_path}/collective/pf/{route_num}_epsilon_{epsilon}_pareto_fronts.pdf")
# except Exception as e:
#     print(e)
#     print("5% Pareto Front plot failed")

try:
    final_routes = final_LSlog_file_all.sort_values(by="Terminals_withTW", ascending=False).RouteNum.tolist()[:top_routes]

    # def point_inside_polygon(x, y):
    #     point = Point(x, y)
    #     return polygon.contains(point)

    with open(f'./logs/configuration_params.pkl', 'rb') as f:
        configuration_params = pickle.load(f)
    allowed_time = configuration_params["allowed_time"]
    allowed_init_time = configuration_params["allowed_init_time"]
    nbd_flags = configuration_params["nbd_flags"]

    color_dict = {'S3opt': "g", 'S3optTW': "m", 'GapRepair': "y", 'FixedPerm': "c", 'Quad': "k", "SimpleCycle": "r", "RandomPermute": "b"}
    marker_dict = {'S3opt': "v", 'S3optTW': "s", 'GapRepair': "D", 'FixedPerm': "o", 'Quad': "P", "SimpleCycle": "X", "RandomPermute": "x"}
    legend_labels = ['S3opt', 'S3optTW', 'GapRepair', 'FixedPerm', 'Quad', "SimpleCycle", "RandomPermute"]
    node_revisit_dict = {}
    edge_revisit_dict = {}
    avg_terminal_revisit = {}
    max_terminal_revisit = {}
    plot_energies_histogram = False
    from collections import defaultdict

    terminal_revist_dict = defaultdict(lambda: 0)  # Keys: max revisit, Value: number of paths

    for route_num in tqdm(all_routes):
        bstsptw_tours, bstsp_tours, _ = get_scalarization_results(route_num)
        (route_num, set_z, time_for_last_imp, move_counter, total_ls_time, set_p) = get_results_ls(route_num)
        result_combined = (bstsptw_tours, bstsp_tours, set_z)
        plot_efficiency_frontier(result_combined, route_num, allowed_time)
        plot_clustering_img(result_combined, route_num, allowed_time)
        # plot_lineage_tree(parents, result_combined, route_num)
        # plot_lineage_histogram(parents, result_combined, route_num)

        # count_dict = {'S3opt': 0, 'S3optTW': 0, 'GapRepair': 0, 'FixedPerm': 0, 'Quad': 0, "SimpleCycle": 0, "RandomPermute": 0}
        node_revisit_dict[route_num], edge_revisit_dict[route_num] = [], []
        # (G, L, terminal_set, time_window_dict, time_cost_dict, energy_dual, direction_dual, mean_energy_dual, mean_turn_dual, tw_term_count,
        #  stdev_energy_dual, stdev_turn_dual) = (get_route_information(route_num))
        for path, _, _ in set_z:
            node_count = Counter(path)
            # ter_revisit = [y for x, y in node_count.items() if x in terminal_set]
            # max_terminal_revisit[route_num] = max(ter_revisit)
            # avg_terminal_revisit[route_num] = np.mean(ter_revisit)
            # terminal_revist_dict[max(ter_revisit)] += 1
            node_revisit_dict[route_num].append(max([y for x, y in node_count.items() if y > 1], default=0))
            edges_count = Counter([(path[i], path[i + 1]) for i in range(len(path) - 1)])
            edge_revisit_dict[route_num].append(max([y for x, y in edges_count.items() if y > 1], default=0))
        if len(set_z) <= 1:
            if route_num in final_routes:
                print(f"Route {route_num} (one of the top TW route) has only one path!")
            continue
        plt.figure(figsize=(10, 7))  # Adjust figure size for better visualization
        ls_turns, ls_energy = zip(*[(t, e) for p, e, t in set_z])
        # plt.plot(ls_turns, ls_energy, 'ko-', label='LS', color="magenta", linewidth=4, markerfacecolor='none', markersize=20, marker='o', alpha=0.5, markeredgewidth=4)
        plt.plot(ls_turns, ls_energy, color='black', linewidth=2)
        # plt.scatter(ls_turns, ls_energy, color='magenta', marker='o', s=70, label='LS Paths', zorder=2)
        plt.scatter(ls_turns, ls_energy, color='pink', edgecolors='grey', marker='o', s=100, zorder=2)
        plt.xlabel("Turn count", fontsize=22)
        plt.ylabel("Energy (kWh)", fontsize=22)
        plt.title(f"Route Id {route_num}", fontsize=26)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.grid(False)
        # plt.legend(loc='upper right', fontsize=12, shadow=True)
        # plt.legend(loc='upper right', fontsize=12, frameon=True, edgecolor='black', shadow=True)
        # plt.show()

        # factor = 1.1
        # x_values_shifted = [x * factor for x in x_values]
        # y_values_shifted = [y * factor for y in y_values]
        # polygon_points = list(zip(x_values, y_values)) + sorted(list(zip(x_values_shifted, y_values_shifted)), key=lambda x: x[0])
        # polygon = Polygon(polygon_points)
        # plt.fill(*polygon.exterior.xy, alpha=0.5, color='lavender')
        # plt.plot(ls_turns, ls_energy, color='black', zorder=1)
        # plt.plot(x_values, y_values, label='EF', color='blue')
        # plt.scatter(ls_turns, ls_energy, color='magenta', marker='.', s=50, zorder=2, label = "LS Paths")
        # plt.legend(loc='upper right', fontsize=15)
        # points_in_Area = []
        # for nbd, nbd_paths in nbd_tw_feasible_tours.items():
        #     point_list = [(path[-1], path[-2]) for path in nbd_paths]
        #     for point in point_list:
        #         if point_inside_polygon(*point):
        # count_dict[nbd] += 1
        # points_in_Area.append(point)
        # plt.scatter(point[0], point[1], c=color_dict[nbd], marker=marker_dict[nbd], s=7)
        # points_in_Area = custom_shuffle(points_in_Area)[:10]
        # for point in points_in_Area:
        #     plt.scatter(point[0], point[1], c=color_dict[nbd], marker=marker_dict[nbd], s=2)
        # scatter_handles = [plt.Line2D([0], [0], marker=marker_dict[label], color='w', label=f"{label}:{0}", markerfacecolor=color_dict[label], markersize=7) for label in legend_labels]
        # plt.legend(handles=scatter_handles, loc='upper right', fontsize=8)
        # plt.tight_layout()
        # plt.show()
        plt.savefig(f"./logs/{route_num}/plots/AreaEFplot_{route_num}.pdf")
        if route_num in final_routes:
            source_path = f"./logs/{route_num}/plots/AreaEFplot_{route_num}.pdf"
            destination_path = f"{folder_path}/final/AreaEFplot_{route_num}.pdf"
            shutil.copy(source_path, destination_path)
            source_path = f"./logs/{route_num}/plots/Clustering_Clustered_{route_num}_{allowed_time}_All.pdf"
            destination_path = f"{folder_path}/final/Clustering_Clustered_{route_num}_{allowed_time}_All.pdf"
            shutil.copy(source_path, destination_path)
        plt.clf()
    print(f" Average Number of Maximum node revisits in optimal tours: {round(np.mean([count for max_revisit_count_list in node_revisit_dict.values() for count in max_revisit_count_list]), 1)}")
    print(f" Average Number of Maximum edge revisits in optimal tours: {round(np.mean([count for max_revisit_count_list in edge_revisit_dict.values() for count in max_revisit_count_list]), 1)}")
except Exception as e:
    print(e)
    print("AreaEFplot plot failed")

overalllog_file = pd.read_csv(r"./logs/final_LSlog_file.csv")
print("S3opt:", round(len(overalllog_file[overalllog_file["S3opt"] != 0]) / len(overalllog_file) * 100))
print("FixedPerm:", round(len(overalllog_file[overalllog_file["FixedPerm"] != 0]) / len(overalllog_file) * 100))
print("Quad:", round(len(overalllog_file[overalllog_file["Quad"] != 0]) / len(overalllog_file) * 100))
print("RandomPermute:", round(len(overalllog_file[overalllog_file["RandomPermute"] != 0]) / len(overalllog_file) * 100))
print("S3optTW:", round(len(overalllog_file[overalllog_file["S3optTW"] != 0]) / len(overalllog_file) * 100))
print("GapRepair:", round(len(overalllog_file[overalllog_file["GapRepair"] != 0]) / len(overalllog_file) * 100))
print("SimpleCycle:", round(len(overalllog_file[overalllog_file["SimpleCycle"] != 0]) / len(overalllog_file) * 100))

######################################################################################
get_table_8()
get_latex_table_9()
top_routes = 9
final_LSlog_file = pd.read_csv('./logs/final_LSlog_file.csv').sort_values(by="Terminals_withTW", ascending=False).iloc[:top_routes].reset_index(drop=True)
print(f"Average number of tours: {round(final_LSlog_file['OnlyLS_paths'].mean())}")

######################################################################################
all_routes = pd.read_csv(f"./lsInputs/working_amazon_routes.csv").Route_num.tolist()
save_path = f'./logs/oneRoute/'
all_operators = ["S3opt", "S3optTW", "GapRepair", "FixedPerm", "Quad", "RandomPermute", "SimpleCycle", "all"]
operator_color = {"S3opt": "g", "S3optTW": "purple", "GapRepair": "y", "FixedPerm": "c", "Quad": "k", "RandomPermute": "b", "SimpleCycle": "r"}
operator_marker = {"S3opt": "v", "S3optTW": "s", "GapRepair": "D", "FixedPerm": "+", "Quad": "P", "RandomPermute": "x", "SimpleCycle": "X"}
route_num = 3793
for route_num in all_routes:
    try:
        path = f'./logs/oneRoute/{route_num}'
        set_z_points = {}
        for operator in all_operators:
            if operator=="SimpleCycle": continue
            folder_path = f'./{path}/{operator}'
            with open(f'{folder_path}/ls_output_{route_num}.pkl', 'rb') as f:
                (route_num, set_z, time_for_last_imp, move_counter, total_ls_time, set_p) = pickle.load(f)
            set_z_points[operator] = [(t, e) for _, e, t in set_z]

        plt.figure(figsize=(10, 8))  # Adjust figure size for better visualization
        for operator, points in set_z_points.items():
            if operator == "all" or operator=="SimpleCycle": continue
            try:
                ls_turns, ls_energy = zip(*points)
            except ValueError:
                ls_turns, ls_energy = [], []
            operator_name = operator
            if operator == "all":
                operator_name = "All"
            if operator == "GapRepair":
                operator_name = "RepairTW"
            plt.scatter(ls_turns, ls_energy, color=operator_color[operator], marker=operator_marker[operator], s=50, zorder=2, label=f"{operator_name}: {len(points)}")
        try:
            ls_turns, ls_energy = zip(*set_z_points["all"])
        except ValueError:
            # pass
            continue
        plt.scatter(ls_turns, ls_energy, color='pink', edgecolors='grey', marker='o', s=50, zorder=3, label=f'All: {len(ls_turns)}')
        plt.subplots_adjust(bottom=0.32)


        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.19), ncol=3, fontsize=18, shadow=True, columnspacing=0.3, handletextpad=0.1, labelspacing=0.1)
        # Customizing legend to have three columns: first two columns with operators, third column with "All"
        handles, labels = plt.gca().get_legend_handles_labels()
        custom_handles = handles[::-1]
        custom_labels = labels[::-1]
        # Plot the custom legend with increased point size
        plt.legend(custom_handles, custom_labels, loc='upper center', bbox_to_anchor=(0.5, -0.19), ncol=3, fontsize=18, shadow=True,
                   columnspacing=0.3, handletextpad=0.1, labelspacing=0.1, scatterpoints=1, markerscale=1.5)

        plt.plot(ls_turns, ls_energy, color='black', linewidth=2, alpha=0.1)
        plt.xlabel("Turn count", fontsize=22)
        plt.ylabel("Energy (kWh)", fontsize=22)
        plt.title(f"Route Id {route_num}", fontsize=26)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.grid(False)
        # plt.show()
        plt.savefig(f"{path}_OneRoute.pdf")
        plt.clf()
    except FileNotFoundError:
        print(f"Route {route_num} not complete")
        continue

