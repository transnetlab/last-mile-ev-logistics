import folium
import pandas as pd
import multiprocessing
from haversine import Unit
from folium.vector_layers import CircleMarker

from helpers.graphBuilder import *

ox.settings.use_cache = True


def plot_routes(route_num):
    amazon_del_coordinates, original_bbox, time_windowlist, depot = read_dataset(route_num)
    tw_coodinates = []
    tw_window = []
    non_tw_coodinates = []
    for x in range(len(time_windowlist)):
        if str(time_windowlist[x][0]) != '-inf' or str(time_windowlist[x][1]) != 'inf':
            tw_coodinates.append(amazon_del_coordinates[x])
            tw_window.append(time_windowlist[x])
        else:
            non_tw_coodinates.append(amazon_del_coordinates[x])
    # Find distance between each delivery coordinate
    dist_mat = haversine_vector(tw_coodinates, tw_coodinates, unit=Unit.METERS, comb=True)
    max_dist = np.max(dist_mat)
    window_list = [x[1] - x[0] for x in tw_window]
    mean_dist = np.mean(dist_mat)
    std_dist = np.std(dist_mat)
    mean_tw = np.mean(window_list)
    if window_list:
        description = pd.Series(window_list).describe().tolist()
        if str(description[2]) == 'nan':
            description[2] = 0
    else:
        description = [0, 0, 0, 0, 0, 0, 0, 0]

    with open(f'./lsInputs/route_{route_num}_G.pkl', 'rb') as f:
        G = pickle.load(f)
    # Plot a folium map and add markers for the delivery coordinates
    # m = ox.plot_graph_folium(G, popup_attribute='name', edge_width=2)
    map_center = (original_bbox[0][0] + original_bbox[2][0]) / 2, (original_bbox[1][1] + original_bbox[3][1]) / 2
    m = folium.Map(location=map_center, zoom_start=14)
    for i in range(len(tw_coodinates)):
        CircleMarker(tw_coodinates[i], radius=5, color="red", fill=True, fill_color="red").add_to(m)
    # Plot blue circles
    for i in range(len(non_tw_coodinates)):
        CircleMarker(non_tw_coodinates[i], radius=5, color="blue", fill=True, fill_color="blue").add_to(m)
    m.save(f'./routeVisualization/{route_num}.html')
    return (route_num, max_dist, mean_dist, std_dist, mean_tw, description)


if __name__ == '__main__':
    cores = 30
    all_routes = pd.read_csv(f"./lsInputs/working_amazon_routes.csv").Route_num.tolist()

    with multiprocessing.Pool(processes=cores) as pool:
        outputs = pool.map(plot_routes, all_routes)
    dist_dict = {x[0]: x[1:] for x in outputs}
    with open('./routeVisualization/distance_dict.txt', 'w') as f:
        f.write(str(dist_dict))
    with open('./routeVisualization/distance_dict.txt', 'r') as f:
        dist_dict = eval(f.read())
    temp = pd.read_csv(r"./logs/final_LSlog_file.csv")
    window_df = pd.DataFrame(columns=['RouteNum', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
    for x in temp.iterrows():
        route_num = x[1]['RouteNum']
        max_dist, mean_dist, std_dist, mean_tw, description = dist_dict[route_num]
        # temp.loc[x[0], 'Max_dist'] = max_dist
        # temp.loc[x[0], 'Mean_dist'] = mean_dist
        # temp.loc[x[0], 'Std_dist'] = std_dist
        # temp.loc[x[0], 'Mean_tw'] = mean_tw
        window_df.loc[len(window_df)] = [route_num] + description
    #     temp.loc[x[0], 'Terminals'] = len(x[1]['Path'].split(','))
    #     if x[1]["OnlyLS_paths"] == 0:
    #         temp.loc[x[0], 'Bool'] = 0
    #     else:
    #         temp.loc[x[0], 'Bool'] = 1
    # temp.to_csv(r"./logs/final_LSlog_file_e.csv", index=False)
    window_df.to_csv(r"./window_description.csv", index=False)
    # # Plot max_dist for Bool == 1 and Bool == 0
    # temp = pd.read_csv(r"./logs/final_LSlog_file_e.csv")
    #
    # plt.figure(figsize=(10, 5))
    # plt.scatter(temp['Bool'],temp['Max_dist'] , s=5)
    # plt.title("Analysing factors affecting the number of only LS paths")
    # plt.xlabel("LS Found paths or not")
    # plt.ylabel("Maximum distance between delivery terminals")
    # # plt.show()
    # plt.savefig(r"./logs/summary/collective/Max_dist.png")
    #
    # plt.clf()
    # plt.figure(figsize=(10, 5))
    # plt.scatter(temp['Bool'],temp['Mean_dist'] , s=5)
    # plt.title("Analysing factors affecting the number of only LS paths")
    # plt.xlabel("LS Found paths or not")
    # plt.ylabel("Mean distance between delivery terminals")
    # # plt.show()
    # plt.savefig(r"./logs/summary/collective/Mean_dist.png")
    #
    # plt.clf()
    # plt.figure(figsize=(10, 5))
    # plt.scatter(temp['Bool'],temp['Std_dist'] , s=5)
    # plt.title("Analysing factors affecting the number of only LS paths")
    # plt.xlabel("LS Found paths or not")
    # plt.ylabel("Std_dist of distance between delivery terminals")
    # # plt.show()
    # plt.savefig(r"./logs/summary/collective/Std_dist.png")
    #
    # plt.clf()
    # plt.figure(figsize=(10, 5))
    # plt.scatter(temp['Bool'],temp['Mean_tw'] , s=5)
    # plt.title("Analysing factors affecting the number of only LS paths")
    # plt.xlabel("LS Found paths or not")
    # plt.ylabel("Mean tw length between delivery terminals")
    # # plt.show()
    # plt.savefig(r"./logs/summary/collective/Mean_tw.png")
    #
    # plt.clf()
    # plt.figure(figsize=(10, 5))
    # plt.scatter(temp['Bool'],temp['Terminals'] , s=5)
    # plt.title("Analysing factors affecting the number of only LS paths")
    # plt.xlabel("LS Found paths or not")
    # plt.ylabel("Total number of terminals")
    # # plt.show()
    # plt.savefig(r"./logs/summary/collective/Terminals.png")
    #
    # plt.clf()
    # plt.figure(figsize=(10, 5))
    # plt.scatter(temp['Bool'],temp['Terminals_withTW'] , s=5)
    # plt.title("Analysing factors affecting the number of only LS paths")
    # plt.xlabel("LS Found paths or not")
    # plt.ylabel("Terminals with TW")
    # # plt.show()
    # plt.savefig(r"./logs/summary/collective/Terminals_withTW.png")
    #
    # plt.clf()
    # plt.figure(figsize=(10, 5))
    # plt.scatter(temp['Bool'],temp['G_nodes'] , s=5)
    # plt.title("Analysing factors affecting the number of only LS paths")
    # plt.xlabel("LS Found paths or not")
    # plt.ylabel("Total number of nodes in graph")
    # # plt.show()
    # plt.savefig(r"./logs/summary/collective/G_nodes.png")
