import itertools
import copy
import math

from core import *
from .input_to_data import *
from cluster import Cluster
from core.utils.forecast import Forecast
import random


from server import Server


def get_server_position(servers, server):
    for i in range(len(servers)):
        if server.id == servers[i].id:
            return i
    return -1


def get_server_position_by_id(servers, server_id):
    for i in range(len(servers)):
        if server_id == servers[i].id:
            return i
    return -1


def get_underloaded_servers(underloaded_clusters):
    servers = []
    for c in underloaded_clusters:
        for server in c.servers:
            if server.status == 1:
                servers.append(server)
    return servers


# NOTE Verified
def get_closest_server(cluster, underloaded_servers):
    centroid_servers_network_delay = cluster.centroid_servers_network_delay
    delay = centroid_servers_network_delay[underloaded_servers[0].id]
    closest_server = underloaded_servers[0]
    i = 0
    pos = 0
    for server in underloaded_servers:
        if delay > centroid_servers_network_delay[server.id]:
            delay = centroid_servers_network_delay[server.id]
            closest_server = server
            pos = i
        i = i+1
    return closest_server, pos


# return the cluster in current clusters which the server belongs to (not shared with)
def get_initial_cluster(initial_clusters, server):
    cluster = Cluster(0)
    # first get the correspondent c.id from initial cluster and use it to search the cluster on current cluster
    for c in copy.deepcopy(initial_clusters):
        pos = get_server_position(c.servers, server)
        if pos >= 0:
            return c
    return cluster


def get_initial_cluster_by_id(initial_clusters, server_id):
    cluster = Cluster(0)
    # first get the correspondent c.id from initial cluster and use it to search the cluster on current cluster
    for c in copy.deepcopy(initial_clusters):
        pos = get_server_position_by_id(c.servers, server_id)
        if pos >= 0:
            return c
    return cluster


# Get the prediction of external workload (entry point function) in a single cluster and function NOTE: DONE
def get_cluster_function_external_workload_prediction(cluster_workload, function_id, num_points_sample,
                                                      num_forecast_points, slot_length, freq):
    forecast = Forecast()
    data = cluster_workload[function_id]
    external_predicted_workload = forecast.list_forecasted_data_poits(data, num_points_sample, num_forecast_points,
                                                                      slot_length=slot_length, freq=freq)
    return external_predicted_workload


def generate_network_topology(k, a, b, c, d, min_cores=1, max_cores=16, min_memory=4, max_memory=64):
    """
    Generate a random network topology with k servers, each having random (x, y) coordinates, cores, and memory.

    Parameters:
    k (int): Number of servers
    a, b (int): Range for x-coordinates
    c, d (int): Range for y-coordinates
    min_cores, max_cores (int): Range for random number of cores for each server
    min_memory, max_memory (int): Range for random memory (in GB) for each server

    Returns:
    list of dicts: A list where each dict represents a server with its attributes
    """
    servers = []
    for i in range(k):
        # Generate random x and y coordinates
        x = round(random.uniform(a, b),2)
        y = round(random.uniform(c, d),2)

        # Randomly assign cores and memory to the server
        cores = random.randint(min_cores, max_cores)
        memory = random.randint(min_memory, max_memory)
        server= Server(i)
        # Create a server representation
        server.location = (x, y)
        server.cores = cores
        server.memory = memory
        server.cores_available = cores
        server.memory_available = memory
        server.initialization()
        servers.append(server)
    return servers


def get_severs_list(total_servers=1, cores=2000, memory=4000, location=(0, 0)):
    servers = []
    for i in range(total_servers):
        server = Server(i)
        server.location = (location[i])
        server.cores = cores
        server.cores_available = cores
        server.memory_available = memory
        server.memory = memory
        server.initialization()
        servers.append(server)
    return servers


def get_distance(location1, location2):
    euclidian_distance = round(math.sqrt((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2),2)
    return euclidian_distance


def total_distance(graph, path):
    distance = 0
    for i in range(len(path) - 1):
        distance += graph[path[i]][path[i+1]]
    return distance


def adjust_clusters(data_list, dynamic_clustering, function_ids, lambs, servers_ids, topology):
    external_predicted_topology_workload = [[0 for _ in range(len(dynamic_clustering.functions))]
                                            for _ in range(len(topology.initial_clusters))]
    i = 0
    for server_id in servers_ids:
        c_id = get_initial_cluster_by_id(topology.initial_clusters, server_id).id
        external_predicted_topology_workload[c_id][function_ids[i]] = lambs[i]
        i = i + 1
    dynamic_clustering.external_predicted_topology_workload = external_predicted_topology_workload
    # total_workload_predict = dynamic_clustering.get_topology_total_workload_prediction(data)
    print(f'=============================BEFORE CLUSTERS CHANGES=========================================')
    print_all(dynamic_clustering, topology, data_list)

    dynamic_clustering.change_clusters(data_list)

    print(f'=============================AFTER CLUSTERS CHANGES=========================================')
    print_all(dynamic_clustering, topology, data_list)


def exists(clusters_ids, c_id):
    for id in clusters_ids:
        if id == c_id:
            return 1
    return 0


def get_position(servers, server_id):
    pos = 0
    for s in servers:
        if s.id == server_id:
            return pos
        pos = pos+1
    return -1


def update_data(json,  data_list, workloads, applications_list):
    i = 0
    for data in data_list:
        json_workload = json.dumps(workloads[i])
        app = applications_list[i]
        app.input["workload_on_source_matrix"] = json_workload
        setup_community_data(data, app.input)
        setup_runtime_data(data, app.input)
        i=i+1


def workload_init(functions, servers):
    workload = np.zeros((len(functions), len(servers))).tolist()
    # print(f':::::::: SERVERS={len(servers)}')
    # print(f'workload={workload}')
    return workload


def set_topology(cluster, app_input):
    # Update cpu_allocation dictionary after changing position
    cpu_allocation = {}
    for function in app_input["function_names"]:
        cpu_allocation[function] = {}
        for server in cluster.servers:
            cpu_allocation[function][server.name] = True

    app_input["actual_cpu_allocations"] = cpu_allocation


def print_all(dynamic_clustering, topology, data_list):
    # dynamic_clustering.external_predicted_topology_workload = external_predicted_topology_workload
    total_workload_predict = dynamic_clustering.total_predicted_topology_workload
    print(f'Total W={total_workload_predict}')

    print(f'Historical-Cores_needed={dynamic_clustering.historical_total_requested_cores_topology}')
    print(f'CLUSTERS={[[server.id for server in c.servers] for c in topology.current_clusters]}')
    print(f'CAPACITY-CLUSTERS={[c.capacity_cores for c in topology.current_clusters]}')
    for c in topology.current_clusters:
        print(f'RECEIVED[C{c.id}]={[server.id for server in c.received_servers]}')
        print(
            f'RECEIVED-SERVERS-ALLOCATED-CORES[C{c.id}]={[(server.id, c.received_servers_allocated_cores[server.id]) for server in c.received_servers]}')

        print(
            f'SERVERS-AVAILABLE-CORES[C{c.id}]={[(server.id, c.get_server_available_resources(server)[0]) for server in c.servers]}')
    print(f'clusters-status={[c.status for c in topology.current_clusters]}')
    over = []
    under = []
    requested_cores = dynamic_clustering.cluster_cores_requested(data_list)
    overloaded, underloaded = dynamic_clustering.get_underloaded_overloaded_clusters(requested_cores)
    for c_o in overloaded:
        over.append(c_o.id)
    for c_u in underloaded:
        under.append(c_u.id)
    print(f'OVERLOADED={over}, UNDERLOADED={under}')


def get_data(data_list, cluster):
    data_empty = Data()
    for data in data_list:
        if data.cluster.id == cluster.id:
            return data
    return data_empty


def held_karp_with_exclusions(graph, start, destination_servers, excluded_servers):
    n = len(graph)
    servers = [server for server in destination_servers if server not in excluded_servers]
    shortest_distance = float('inf')
    shortest_path = []

    for perm in itertools.permutations(servers):
        perm = (start,) + perm  # Ensure the start server is always at the beginning
        distance = total_distance(graph, perm)
        if distance < shortest_distance:
            shortest_distance = distance
            shortest_path = perm

    return shortest_path


def set_workload(dynamic_clustering, topology, data_list, servers_ids, function_ids, lambs, applications_list):
    adjust_clusters(data_list, dynamic_clustering, function_ids, lambs, servers_ids, topology)
    for cluster in topology.current_clusters:
        data=get_data(data_list, cluster)
        data.cluster = cluster
        data.nodes = [server for server in cluster.servers]
    clusters = []
    workloads = []
    for c in topology.current_clusters:
        single_workload = workload_init(c.functions, c.servers)
        workloads.append(single_workload)

    i = 0
    for server_id in servers_ids:
        c_id = get_initial_cluster_by_id(topology.initial_clusters, server_id).id
        current_cluster = topology.current_clusters[c_id]
        servers = current_cluster.servers
        pos_server = get_position(servers, server_id)
        # the workloads is a list of matrixes server x function for each cluster
        # while for functions we just use the id as index because all the functions are suppose
        # to be replicated on the clusters, for server we take the position on its cluster
        # print(f'workloads[{c_id}][{function_ids[i]}][{pos_server}] = lambs[{i}]')
        workloads[c_id][function_ids[i]][pos_server] = lambs[i]
        i = i+1

        if exists(clusters, c_id):
            continue
        clusters.append(c_id)

    for c in topology.current_clusters:
        c.compute_network_delays(topology.network_delays)
    i = 0
    # print(f'@@@@@@@@@@@@ Delays={[c.network_delays for c in topology.current_clusters]}')
    for c_id in clusters:
        cluster = topology.current_clusters[c_id]
        set_topology(cluster, applications_list[i].input)
        i = i+1

    return workloads