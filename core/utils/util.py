import itertools
import os
from math import radians, sin, cos, sqrt, atan2
# from dynamic_clustering import DynamicClustering
# from network_topology import Topology
from .input_to_data import *
from cluster import Cluster
from core.utils.forecast import Forecast
import random
import copy


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


def get_severs_list(old_servers=None, total_servers=1, cores_range=(0, 1), step=1, cores=None, memory=None, location=(0, 0)):
    servers = []
    qty_old_servers = 0
    if old_servers is not None:
        servers = old_servers
        qty_old_servers = len(old_servers)

    for i in range(qty_old_servers, total_servers):
        server = Server(i)
        server.location = (location[i])
        if cores is None:
            # e.g.: cores range (1000,8000), step =1000 the range will be [1000, 2000, 3000, ..., 8000]
            # and random.choice will choose one randomly
            cores = random.choice(range(cores_range[0], cores_range[1] + 1, step))
        memory = 2 * cores
        server.cores = cores
        server.cores_available = cores
        server.memory_available = memory
        server.memory = memory
        server.name = f's{i}'
        server.initialization()
        servers.append(server)
    return servers


def get_static_severs_list(cores_list, memory_list, location_list):
    servers = []
    for i in range(len(cores_list)):
        server = Server(i)
        server.location = (location_list[i])
        server.cores = cores_list[i]
        server.memory_available = memory_list[i]
        server.memory = memory_list[i]
        server.name = f's{i}'
        server.initialization()
        servers.append(server)
    return servers


def update_static_severs_list(cores_list, memory_list, servers):
    for i in range(len(cores_list)):
        servers[i].cores = cores_list[i]
        servers[i].memory_available = memory_list[i]
        servers[i].memory = memory_list[i]
        servers[i].name = f's{i}'
        servers[i].initialization()


def get_distance(coord1, coord2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.
    Args:
        coord1 (tuple): (latitude, longitude) of the first point.
        coord2 (tuple): (latitude, longitude) of the second point.
    Returns:
        float: Distance in kilometers.
    """
    R = 6371  # Earth's radius in kilometers
    unit = 1000  # for milliseconds

    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    haversine_distance = R * c*unit
    return haversine_distance


def total_distance(graph, path):
    distance = 0
    for i in range(len(path) - 1):
        distance += graph[path[i]][path[i+1]]
    return distance


def adjust_clusters(data_list, dynamic_clustering, function_ids, lambs, servers_ids):
    topology = dynamic_clustering.topology
    external_predicted_topology_workload = [[0 for _ in range(len(dynamic_clustering.functions))]
                                            for _ in range(len(topology.initial_clusters))]
    i = 0
    for server_id in servers_ids:
        c_id = get_initial_cluster_by_id(topology.initial_clusters, server_id).id
        # print(f'external_predicted_topology_workload[{c_id}][{function_ids[i]}] = {lambs[i]}')
        external_predicted_topology_workload[c_id][function_ids[i]] = lambs[i]
        i = i + 1
    dynamic_clustering.external_predicted_topology_workload = external_predicted_topology_workload
    # total_workload_predict = dynamic_clustering.get_topology_total_workload_prediction(data)
    # print(f'=============================BEFORE CLUSTERS CHANGES=========================================')
    # print_all(dynamic_clustering, topology, data_list)

    dynamic_clustering.change_clusters(data_list)

    # print(f'=============================AFTER CLUSTERS CHANGES=========================================')
    # print_all(dynamic_clustering, topology, data_list)


def exists(clusters_ids, c_id):
    for id in clusters_ids:
        if id == c_id:
            return 1
    return 0


def get_position(servers, server_id):
    pos = 0
    # print(f'servers={[s.name for s in servers]}')
    for s in servers:
        if s.id == server_id:
            return pos
        pos = pos+1
    # print(f'server_id={server_id}')
    # print(f'servers= {[s.id for s in servers]}')
    return -1


"""
compute the max cores to be allocated for each function on each server. The idea is to use the weighted cores according 
to the internal workload, cores required by each function to execute a single request and the capacity of the server. 
"""


def get_max_cores_f_on_server(server, w, dag):
    i = server.id
    f_max_cores = np.zeros((len(w)))
    sum_used_cores = 0
    for f in range(len(w)):
        function = dag.getFunction_byId(f)
        sum_used_cores = sum_used_cores+w[f][i]*function.uf
    sum_used_cores = max(sum_used_cores, 0.0001) # avoid division by zero
    contation = 1 if sum_used_cores > server.cores else 0

    for f in range(len(w)):
        function = dag.getFunction_byId(f)
        if contation:
            f_max_cores[f] = w[f][i]*function.uf*server.cores/sum_used_cores
        else:
            f_max_cores[f] = w[f][i] * function.uf
    return f_max_cores


def update_data(json,  data_list, workloads, input, is_heu=False):
    i = 0
    for data in data_list:
        workload = workloads[i]
        if is_heu:
            workloads_converted = [[[int(value) for value in row] for row in layer] for layer in workload]
        else:
            workloads_converted = [[int(value) for value in row] for row in workload]
        json_workload = json.dumps(workloads_converted)
        input["workload_on_source_matrix"] = json_workload
        setup_community_data(data, input)
        setup_runtime_data(data, input)
        i=i+1


def workload_init(functions, servers, lamb=None):
    if lamb is None:
        workload = np.zeros((len(functions), len(servers))).tolist()
    else:
        # workload = ([[0 for _ in range(len(servers))] for _ in range(len(functions))] for _ in range(lamb))
        workload = np.zeros((lamb,len(functions),len(servers))).tolist()
    return workload


def set_topology(cluster, app_input):
    # Update cpu_allocation dictionary after changing position
    cpu_allocation = {}
    for function in app_input["function_names"]:
        cpu_allocation[function] = {}
        for server in cluster.servers:
            cpu_allocation[function][server.name] = True
    # print(f'%%%%%%%%%%%%%%%%%%%%%%%%%% cpu_allocation[C{cluster.id}]={cpu_allocation}')
    cluster.cpu_allocation = cpu_allocation
    # app_input["actual_cpu_allocations"]=cpu_allocation


def print_all(dynamic_clustering, topology, data_list):
    # dynamic_clustering.external_predicted_topology_workload = external_predicted_topology_workload
    total_workload_predict = dynamic_clustering.total_predicted_topology_workload

    # print(f'Historical-Cores_needed={dynamic_clustering.historical_total_requested_cores_topology}')
    # print(f'CLUSTERS={[[server.id for server in c.servers] for c in topology.current_clusters]}')
    # print(f'CAPACITY-CLUSTERS={[c.capacity_cores for c in topology.current_clusters]}')
    # # for c in topology.current_clusters:
    #     print(f'RECEIVED[C{c.id}]={[server.id for server in c.received_servers]}')
    #     print(
    #         f'RECEIVED-SERVERS-ALLOCATED-CORES[C{c.id}]={[(server.id, c.received_servers_allocated_cores[server.id]) for server in c.received_servers]}')
    #
    #     print(
    #         f'SERVERS-AVAILABLE-CORES[C{c.id}]={[(server.id, c.get_server_available_resources(server)[0]) for server in c.servers]}')
    # print(f'clusters-status={[c.status for c in topology.current_clusters]}')
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


def get_requests(dynamic_clustering, data_list, function_ids, lambs, servers_ids):
    cluster_reqs = np.zeros(len(data_list), dtype=int)
    topology = dynamic_clustering.topology
    for i in range(len(servers_ids)):
        c_id = get_initial_cluster_by_id(topology.initial_clusters, servers_ids[i]).id
        cluster_reqs[c_id] = cluster_reqs[c_id] + lambs[i]
    return cluster_reqs


def set_workload(dynamic_clustering, data_list, servers_ids, function_ids, lambs, input, is_heu=False):
    adjust_clusters(data_list, dynamic_clustering, function_ids, lambs, servers_ids)

    # print(f'$$$$$ lambs={lambs}')
    topology = dynamic_clustering.topology
    for cluster in topology.current_clusters:
        data = get_data(data_list, cluster)
        data.cluster = cluster
        data.nodes = [server for server in cluster.servers]
    clusters = []
    workloads = []
    if is_heu:
        """
        given two clusters c1(s0,s1,s2) and c2(s3,s4), 3 functions and 4 servers
        e.g.: given lambs = [2,4] for functions_ids [0,1] on servers_ids [0, 1]
        the lamb[0] goes to function_ids[0] on server_ids[0]
        then we have sum_lamb= sum(lambs)=2+4=6 and the initial workloads will be:
        [c1[lamb1|f0[0,0,0],f1[0,0,0],f2[0,0,0]], lamb2|f0[0,0,0],f1[0,0,0],f2[0,0,0]], ..., 
        lamb6|f0[0,0,0],f1[0,0,0],f2[0,0,0]],
        c2[lamb1|f0[0,0],f1[0,0],f2[0,0]], lamb2|f0[0,0],f1[0,0],f2[0,0]], ..., lamb6|f0[0,0],f1[0,0],f2[0,0]]]
        """
        cluster_reqs = np.zeros(len(data_list), dtype=int)  # initialize number of requests per cluster
        cluster_total_reqs = get_requests(dynamic_clustering, data_list, function_ids, lambs, servers_ids)
        for c in topology.current_clusters:
            single_workload = workload_init(c.functions, c.servers, lamb=cluster_total_reqs[c.id])
            workloads.append(single_workload)

        """
       insert one request on each subset of workload on each cluster. set to correspondent function and server
       according to the previous info lambs = [2,4] for functions_ids [0,1] on servers_ids [0, 1] 
       [c1[lamb1|f0[1,0,0],f1[1,0,0],f2[0,0,0]], lamb2|f0[1,0,0],f1[1,0,0],f2[0,0,0]], ..., 
       lamb6|f0[0,0,0],f1[0,0,0],f2[0,0,0]],
       c2[lamb1|f0[0,0],f1[0,0],f2[0,0]], lamb2|f0[0,0],f1[0,0],f2[0,0]], ..., lamb6|f0[0,0],f1[0,0],f2[0,0]]]
        """
        i = 0
        for server_id in servers_ids:
            c_id = get_initial_cluster_by_id(topology.initial_clusters, server_id).id
            current_cluster = topology.current_clusters[c_id]
            servers = current_cluster.servers
            pos_server = get_position(servers, server_id)
            pos_init = cluster_reqs[c_id]
            pos_last = cluster_reqs[c_id] + lambs[i]
            for pos_request in range(pos_init, pos_last):
                workloads[c_id][pos_request][function_ids[i]][pos_server] = 1.0
                # print(f'workloads[{c_id}][{pos_request}][{function_ids[i]}][{pos_server}]')
            cluster_reqs[c_id] = pos_last  # the next set of requests(lamb)
            # incoming to the same cluster must be set from position pos_last
            i = i+1

            if exists(clusters, c_id):
                continue
            clusters.append(c_id)
    else:
        for c in topology.current_clusters:
            single_workload = workload_init(c.functions, c.servers)
            workloads.append(single_workload)

        i = 0
        for server_id in servers_ids:
            c_id = get_initial_cluster_by_id(topology.initial_clusters, server_id).id
            # print(f'c_id ={c_id }')
            current_cluster = topology.current_clusters[c_id]
            servers = current_cluster.servers
            pos_server = get_position(servers, server_id)
            # print(f'pos_server[{server_id}] ={pos_server}')
            # the workloads is a list of matrixes server x function for each cluster
            # while for functions we just use the id as index because all the functions are
            # suppose to be replicated on the clusters, for server we take the position on its cluster
            workloads[c_id][function_ids[i]][pos_server] = lambs[i]
            i = i + 1

            if exists(clusters, c_id):
                continue
            clusters.append(c_id)

    for c in topology.current_clusters:
        c.compute_network_delays(topology.network_delays)
    i = 0
    for c_id in clusters:
        cluster = topology.current_clusters[c_id]
        set_topology(cluster, input)
        i = i+1

    # print(f'workloads={workloads}')
    return workloads


def get_data_list(clustering, input):
    data_list = []
    clusters = clustering.topology.current_clusters
    for c in clusters:
        data = Data()
        data.cluster = c
        data.nodes = [server for server in c.servers]
        setup_community_data(data, input)
        setup_runtime_data(data, input)
        data_list.append(data)
    return data_list


def hide_functions_without_workload(data, func_ids):
    functions = [f.name for f in data.cluster.functions]
    memories = data.function_memories
    cores = data.core_per_req_matrix
    # print(f'******CORES ACTUAL ALLOCATION={data.actual_cpu_allocations}')
    # print(f'******CORES PER REQUEST={data.core_per_req_matrix}')

    entrypoint_f = []
    entrypoint_zero_workload = []
    for f in functions:
        predecessors_f = data.dag.get_predecessors(f)
        if len(predecessors_f) == 0:
            entrypoint_f.append(f)
    func_names = [functions[i] for i in func_ids]

    for f in entrypoint_f:
        if f not in func_names:
            entrypoint_zero_workload.append(f)
    f_to_remove = []
    for f in entrypoint_zero_workload:
        f_to_remove.append(int(f[1:]))  # f[-1] takes the last char. e.g.: "f0"[-1]= 0

        descendants_f = data.dag.get_descendents_ids(f)
        # print(f'@@@ descendants_f[{f}]={descendants_f}')
        for f_id in descendants_f:
            f_to_remove.append(f_id)

    for f_id in f_to_remove:
        memories[f_id] = 0
        for j in range(len(cores[f_id])):
            cores[f_id][j] = 0
    # print(f'@@@@ data.function_memories= {data.function_memories}')
    # print(f'##### AFTER data.core_per_req_matrix={data.core_per_req_matrix}')


def get_first_decision(decision_delay_approach, placement_delay):
    diff=0
    if len(decision_delay_approach) > 0:
        min_delay = min(decision_delay_approach[0][1],placement_delay)
        diff= decision_delay_approach[0][1] - min_delay
        first_decision=min_delay
    else:
        first_decision = placement_delay

    return first_decision, diff


def fill_output(first_sum_lamb, sumlamb, coldstart, coldstart_delay_approach, placement_delay, avg_placement_delay,  decision_delay_approach, avg_decision_delay_approach, delays_approach, i,
                memory, memory_approach, network_delay, avg_network_delay, network_delay_approach, avg_network_delay_approach, nodes_approach,
                total_nodes, universal_param, vary_topologies, has_coldstart=True, has_decision_delay=True, real_world=False):
    diff = 0
    if i > 0:
        if decision_delay_approach[i - 1, 1] > placement_delay*2:
            placement_delay = decision_delay_approach[i - 1, 1]*0.75  # correct the error of the hardware
        if real_world:
            if decision_delay_approach[i - 1, 1]*2 < placement_delay:
                placement_delay = decision_delay_approach[i - 1, 1] * 1.5  # correct the error of the hardware
        else:
            if memory_approach[i - 1, 1] > memory:
                memory = memory_approach[i - 1, 1]
        avg_placement_delay = placement_delay/sumlamb

    # if vary_topologies:  # if varying topology:
    #     if i > 0:
    #         if network_delay_approach[i - 1, 1] < network_delay:
    #             network_delay = network_delay_approach[i - 1, 1]
    #             avg_network_delay = network_delay/sumlamb
    #
    #         if coldstart_delay_approach[i - 1, 1] < coldstart:
    #             coldstart = coldstart_delay_approach[i - 1, 1]
    #
    #         if memory_approach[i - 1, 1] < memory:
    #             memory = memory_approach[i - 1, 1]

    # else:  # vary the workload or the quantity of applications
    #     if not real_world:
    #         if i > 0:
    #             if network_delay_approach[i - 1, 1] > network_delay:
    #                 network_delay = network_delay_approach[i - 1, 1]
    #                 avg_network_delay = network_delay / sumlamb
    #
    #             if coldstart_delay_approach[i - 1, 1] > coldstart:
    #                 coldstart = coldstart_delay_approach[i - 1, 1]
    #
    #             if memory_approach[i - 1, 1] > memory:
    #                 memory = memory_approach[i - 1, 1]
    #     first_decision, diff = get_first_decision(decision_delay_approach, placement_delay)
    #     decision_delay_approach[0][1] = first_decision
    #     avg_decision_delay_approach[0][1] = first_decision/first_sum_lamb
    #     delays_approach[0][1] = delays_approach[0][1]-has_decision_delay*diff

    total_delay = network_delay + has_decision_delay*placement_delay + has_coldstart*coldstart
    delays_approach[i] = universal_param, total_delay
    memory_approach[i] = universal_param, memory
    nodes_approach[i] = universal_param, total_nodes
    coldstart_delay_approach[i] = universal_param, coldstart
    network_delay_approach[i] = universal_param, network_delay
    decision_delay_approach[i] = universal_param, placement_delay
    avg_decision_delay_approach[i] = universal_param, avg_placement_delay
    avg_network_delay_approach[i] = universal_param, avg_network_delay
    return total_delay


def get_functions(data):
    function = []
    for func in data.functions:
        function.append(func.split("/")[1])
    return function


def get_cleaned_functions(functions):
    function = []
    for func in functions:
        function.append(func.split("/")[1])
    return function

def get_dep_results(data, x_approach, cfj):
    qty_functions, function, nodes = basic_data(data)
    x = np.zeros(x_approach.shape)
    y = np.zeros((qty_functions, nodes, qty_functions, nodes))
    z = np.zeros((qty_functions, nodes, qty_functions, nodes))
    dag = data.dag
    workload = data.workload_matrix
    # fill x[f,i,j]
    for f in range(qty_functions):
        for i in range(nodes):
            if workload[f, i] > 0.0000000001:
                x[f, i] = x_approach[f, i]
    # fill y[f,i,g,j]
    for f in range(qty_functions):
        for i in range(nodes):
            seq_successor = dag.get_sequential_successors_indexes(function[f])
            for fs in seq_successor:
                if cfj[f, i]:
                    j = getNode(data, i, fs, cfj)
                    y[f, i, fs, j] = 1
            # fill z[f,i,g,j]
            parallel_successors_groups = dag.get_parallel_successors_indexes(function[f])
            for par_group in parallel_successors_groups:
                for fp in par_group:
                    if cfj[f, i]:
                        j = getNode(data, i, fp, cfj)
                        z[f, i, fp, j] = 1
    return x, y, z


def get_internal_workload(data, x_approach, cfj):
    funcs, function, nodes = basic_data(data)
    w = np.zeros((funcs, nodes))
    x, y, z = get_dep_results(data, x_approach, cfj)
    lamb = data.workload_matrix
    dag = data.dag
    m = data.m

    # partial workload x*lamb for direct requests
    for f in range(funcs):
        for i in range(nodes):
            for j in range(nodes):
                w[f, j] = w[f, j] + x[f, i, j] * lamb[f, i]

    # partial internal and external workload for sequential invocations
    for f in range(funcs):
        for i in range(nodes):
            for j in range(nodes):
                seq_successor = dag.get_sequential_successors_indexes(function[f])
                for fs in seq_successor:
                    w[fs, j] = w[fs, j] + y[f, i, fs, j] * m[f][fs] * w[f, i]

    # partial internal and external workload for parallel invocations
    for f in range(funcs):
        for i in range(nodes):
            for j in range(nodes):
                parallel_successors_groups = dag.get_parallel_successors_indexes(function[f])
                for par_group in parallel_successors_groups:
                    for fp in par_group:
                        w[fp, j] = w[fp, j] + z[f, i, fp, j] * m[f][fp] * w[f, i]
    return w


def get_w(data,  x, y, z):
    funcs, function, nodes = basic_data(data)
    w = np.zeros((funcs, nodes))

    lamb = data.workload_on_source_matrix
    dag = data.dag
    m = data.m
    # partial workload x*lamb for direct requests
    for f in range(funcs):
        for i in range(nodes):
            for j in range(nodes):
                w[f, j] = w[f, j] + x[f, i, j] * lamb[f, i]

    # partial internal and external workload for sequential invocations
    for f in range(funcs):
        for i in range(nodes):
            for j in range(nodes):
                seq_successor = dag.get_sequential_successors_indexes(function[f])
                for fs in seq_successor:
                    w[fs, j] = w[fs, j] + y[f, i, fs, j] * m[f][fs] * w[f, i]

    # partial internal and external workload for parallel invocations
    for f in range(funcs):
        for i in range(nodes):
            for j in range(nodes):
                parallel_successors_groups = dag.get_parallel_successors_indexes(function[f])
                for par_group in parallel_successors_groups:
                    for fp in par_group:
                        w[fp, j] = w[fp, j] + z[f, i, fp, j] * m[f][fp] * w[f, i]
    return w


def get_net_delay(data, x_approach, cfj):
    w = get_internal_workload(data, x_approach, cfj)
    # print(f'W={w}')
    funcs, function, nodes = basic_data(data)
    dag = data.dag
    m = data.m
    network_delay = data.cluster.network_delays

    lamb = data.workload_matrix
    x, y, z = get_dep_results(data, x_approach, cfj)

    sum_f = 0
    for f in range(funcs):
        sum_i = 0
        for i in range(nodes):
            if lamb[f, i] > 0.1:
                for j in range(nodes):
                    sum_f = sum_f + network_delay[i][j] * x[f, i, j] * lamb[f, i]

            sum_sequential = 0
            for j in range(nodes):
                sum_fs = 0
                seq_successor = dag.get_sequential_successors_indexes(function[f])
                for fs in seq_successor:
                    # delay_y = y[f, i, fs, j]*m[f][fs]*w[f, i]*nrt[fs]
                    delay_y = y[f, i, fs, j] * m[f][fs] * w[f, i]

                    sum_fs = sum_fs + delay_y
                sum_sequential = sum_sequential + sum_fs * network_delay[i][j]
            parallel_successors_groups = dag.get_parallel_successors_indexes(function[f])
            sum_parallel = 0
            for par_group in parallel_successors_groups:
                max_delay_z = float('-inf')
                for fp in par_group:
                    for j in range(nodes):
                        # delay_z = z[f, i, fp, j] * m[f][fp] * w[f, i] * nrt[fp] * network_delay[i][j]
                        delay_z = z[f, i, fp, j] * m[f][fp] * w[f, i] * network_delay[i][j]
                        if delay_z > max_delay_z:
                            max_delay_z = delay_z

                if max_delay_z > float('-inf'):
                    sum_parallel = sum_parallel + max_delay_z
            sum_i = sum_i + sum_sequential + sum_parallel
        sum_f = sum_f + sum_i
    return sum_f


def basic_data(data):
    function = get_functions(data)
    nodes = len(data.cluster.servers)
    funcs = len(data.functions)
    return funcs, function, nodes


def getNode(data, node_i, f2, cfj, is_closest=True):
    selected_delay = float('inf')
    node_j = node_i
    delayij = data.cluster.network_delays
    nodes = delayij.shape[0]
    if is_closest:
        for j in range(nodes):
            if cfj[f2, j] > 0:
                if selected_delay > delayij[node_i, j]:
                    selected_delay = delayij[node_i, j]
                    node_j = j
    else:
        selected_delay = 0
        for j in range(nodes):
            if cfj[f2, j] > 0:
                if selected_delay < delayij[node_i, j]:
                    selected_delay = delayij[node_i, j]
                    node_j = j
    return node_j


def generate_functions(function_names):
    functions=[]
    for i in range(len(function_names)):
        functions.append(Function(i, name=function_names[i]))
    return functions


def generate_random_server_ids(total_apps, total_servers, fix_server_id=-1):
    servers_ids = []
    for num_servers in range(1, total_apps+1):
        if fix_server_id >= 0:
            servers_ids.append([0 for _ in range(num_servers)])
        else:
            servers_ids.append([random.randint(0, total_servers-1) for _ in range(num_servers)])
    return servers_ids


def generate_random_server_ids_complex(total_apps, initial_servers=10, step=10):
    servers_ids = []
    total_servers = initial_servers
    for num_servers in range(1, total_apps + 1):
        servers_ids.append([random.randint(0, total_servers - 1) for _ in range(num_servers)])
        total_servers = total_servers + step
    return servers_ids


def save_list_of_lists(log_directory, list_of_lists, file_name):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
        # Save the table as a PDF in the specified directory
    file_name = os.path.join(log_directory, file_name)
    with open(file_name, 'w') as file:
        for sublist in list_of_lists:
            file.write(f"{sublist}\n")


def save_dynamic_clusters(log_directory, clusterings, s_ids_file_name, cores_file_name, memories_file_name,
                          server_locations_file_name, f_ids_file_name, f_names_file_name):
    # if not os.path.exists(log_directory):
    #     os.makedirs(log_directory)
    # # Save the table as a PDF in the specified directory
    # s_ids_file_name = os.path.join(log_directory, s_ids_file_name)
    # cores_file_name = os.path.join(log_directory, cores_file_name)
    # memories_file_name = os.path.join(log_directory, memories_file_name)
    # server_locations_file_name = os.path.join(log_directory, server_locations_file_name)
    # f_ids_file_name = os.path.join(log_directory, f_ids_file_name)
    # f_names_file_name = os.path.join(log_directory, f_names_file_name)

    server_ids = []
    server_cores = []
    server_memories = []
    server_locations = []
    function_ids = []
    function_names = []
    for dcl in clusterings:
        s_ids = []
        s_cores = []
        s_memories = []
        s_locations = []
        f_ids = []
        f_names = []
        topology = dcl.topology
        servers = topology.servers

        for s in servers:
            s_ids.append(s.id)
            s_cores.append(s.cores)
            s_memories.append(s.memory)
            s_locations.append(s.location)
        functions = dcl.functions
        for f in functions:
            f_ids.append(f.id)
            f_names.append(f.name)

        server_ids.append(s_ids)
        server_cores.append(s_cores)
        server_memories.append(s_memories)
        server_locations.append(s_locations)
        function_ids.append(f_ids)
        function_names.append(f_names)
    save_list_of_lists(log_directory, server_ids, s_ids_file_name)
    save_list_of_lists(log_directory, server_cores, cores_file_name)
    save_list_of_lists(log_directory, server_memories, memories_file_name)
    save_list_of_lists(log_directory, server_locations, server_locations_file_name)
    save_list_of_lists(log_directory, function_ids, f_ids_file_name)
    save_list_of_lists(log_directory, function_names, f_names_file_name)


"""
save matrixes on txt files. Example of txt file f_network_delay_approach for matrix network_delay_approach
"""


def save_matrixes(log_directory, chart_param, network_delay_approach, f_network_delay_approach, avg_network_delay_approach,
                  f_avg_network_delay_approach, memory_approach, f_memory_approach, coldstart_delay_approach,
                  f_coldstart_delay_approach, decision_delay_approach, f_decision_delay_approach,
                  avg_decision_delay_approach, f_avg_decision_delay_approach, delays_approach, f_delays_approach,
                  nodes_approach, f_nodes_approach):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
        # Save the table as a PDF in the specified directory

    f_network_delay_approach=form_file_name(f_network_delay_approach, chart_param)
    f_network_delay_approach = os.path.join(log_directory, f_network_delay_approach)

    f_avg_network_delay_approach = form_file_name(f_avg_network_delay_approach, chart_param)
    f_avg_network_delay_approach = os.path.join(log_directory, f_avg_network_delay_approach)

    f_memory_approach = form_file_name(f_memory_approach, chart_param)
    f_memory_approach = os.path.join(log_directory, f_memory_approach)

    f_coldstart_delay_approach = form_file_name(f_coldstart_delay_approach, chart_param)
    f_coldstart_delay_approach = os.path.join(log_directory, f_coldstart_delay_approach)

    f_decision_delay_approach = form_file_name(f_decision_delay_approach, chart_param)
    f_decision_delay_approach = os.path.join(log_directory, f_decision_delay_approach)

    f_avg_decision_delay_approach = form_file_name(f_avg_decision_delay_approach, chart_param)
    f_avg_decision_delay_approach = os.path.join(log_directory, f_avg_decision_delay_approach)

    f_delays_approach = form_file_name(f_delays_approach, chart_param)
    f_delays_approach = os.path.join(log_directory, f_delays_approach)

    f_nodes_approach = form_file_name(f_nodes_approach, chart_param)
    f_nodes_approach = os.path.join(log_directory, f_nodes_approach)

    np.savetxt(f_network_delay_approach, network_delay_approach, fmt='%f')
    np.savetxt(f_avg_network_delay_approach, avg_network_delay_approach, fmt='%f')
    np.savetxt(f_memory_approach, memory_approach, fmt='%f')
    np.savetxt(f_coldstart_delay_approach, coldstart_delay_approach, fmt='%f')
    np.savetxt(f_decision_delay_approach, decision_delay_approach, fmt='%f')
    np.savetxt(f_avg_decision_delay_approach, avg_decision_delay_approach, fmt='%f')
    np.savetxt(f_delays_approach, delays_approach, fmt='%f')
    np.savetxt(f_nodes_approach, nodes_approach, fmt='%f')


def form_file_name(original_filename, chart_param):
    base_name, extension = original_filename.rsplit('.', 1)
    new_filename = f'{base_name}{chart_param}.{extension}'
    return new_filename


def get_matrixes(app, log_directory, log_subdirectory, chart_param, f_network_delay_approach, f_avg_network_delay_approach, f_memory_approach,
                  f_coldstart_delay_approach, f_decision_delay_approach, f_avg_decision_delay_approach,
                  f_delays_approach, f_nodes_approach):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
        # log_directory = f'{log_directory}/{log_subdirectory}'
        # print("Hi=========")
    else:
        # print(f'log_subdirectory={log_subdirectory}')
        log_directory = f'{log_directory}/{log_subdirectory}'
        # print(f'+++++++++++++++++ {log_directory}')
    # Save the table as a PDF in the specified directory
    f_network_delay_approach = form_file_name(f_network_delay_approach, chart_param)
    # print(f'============= {f_network_delay_approach}')
    f_network_delay_approach = os.path.join(log_directory, f_network_delay_approach)
    # print(f'********************** {f_network_delay_approach}')


    f_avg_network_delay_approach = form_file_name(f_avg_network_delay_approach, chart_param)
    f_avg_network_delay_approach = os.path.join(log_directory, f_avg_network_delay_approach)

    f_memory_approach = form_file_name(f_memory_approach, chart_param)
    f_memory_approach = os.path.join(log_directory, f_memory_approach)

    f_coldstart_delay_approach = form_file_name(f_coldstart_delay_approach, chart_param)
    f_coldstart_delay_approach = os.path.join(log_directory, f_coldstart_delay_approach)

    f_decision_delay_approach = form_file_name(f_decision_delay_approach, chart_param)
    f_decision_delay_approach = os.path.join(log_directory, f_decision_delay_approach)

    f_avg_decision_delay_approach = form_file_name(f_avg_decision_delay_approach, chart_param)
    f_avg_decision_delay_approach = os.path.join(log_directory, f_avg_decision_delay_approach)

    f_delays_approach = form_file_name(f_delays_approach, chart_param)
    f_delays_approach = os.path.join(log_directory, f_delays_approach)

    f_nodes_approach = form_file_name(f_nodes_approach, chart_param)
    f_nodes_approach = os.path.join(log_directory, f_nodes_approach)

    network_delay = np.loadtxt(f_network_delay_approach)
    avg_network_delay = np.loadtxt(f_avg_network_delay_approach)
    memory = np.loadtxt(f_memory_approach)
    coldstart_delay = np.loadtxt(f_coldstart_delay_approach)
    decision_delay = np.loadtxt(f_decision_delay_approach)
    avg_decision_delay = np.loadtxt(f_avg_decision_delay_approach)
    total_delays = np.loadtxt(f_delays_approach)
    nodes = np.loadtxt(f_nodes_approach)

    response = app.response_class(
        response=json.dumps({
            "cpu_routing_rules": {},
            "cpu_allocations": {},
            "gpu_routing_rules": {},
            "gpu_allocations": {},
        }),
        status=200,
        mimetype='application/json'
    )

    return response, network_delay, avg_network_delay, memory, coldstart_delay, decision_delay, avg_decision_delay, \
        total_delays, nodes


def get_list_of_lists(log_directory, file_name):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
        # Save the table as a PDF in the specified directory
    path_name = os.path.join(log_directory, file_name)
    workload = np.loadtxt(path_name)
    return workload

def get_ordered_locations(locations, scale=0):
    """
    order the locations (coordinates) according to the sum distance among them.
    We start from the coordinate in index 0 them the following by the closest to the all
    selected ones, i.e, a coordinate loc_j is selected if the sum of the distances from loc_j to the all
    the already selected coordinates is lesser than all the sums from other coordinates not yet selected to
    the selected ones.

    Ex: 0  2  1  10 3
        2  0  9  2  50
        1  30 0  5  10
        10 2  5  0  2
        3  50 10 2  0
    first step: selected_pos=[0]
    followed by position 2 because the sum of distances from this position to the selected is 1
    then we have selected_pos=[0,2]
    Next: from pos 1 we have: sum=2+30=32; pos 3: sum= 10+5=15; pos 4: sum=3+10=13 result: selected_pos=[0,2,4]
    etc.
    """
    remain_indexes = [i + 1 for i in range(len(locations) - 1)]
    delays = get_net_delay_matrix(locations)

    result_indexes = [0]
    while len(remain_indexes) > 0:
        min_sum_delays = float('inf')
        min_index = -1
        for j in remain_indexes:
            delays_j = 0
            for i in result_indexes:
                delays_j = delays_j + delays[i][j]
            if min_sum_delays > delays_j:
                min_sum_delays = delays_j
                min_index = j
        result_indexes.append(min_index)
        remain_indexes.remove(min_index)
    if scale > 0:
        step = round(scale / 2)
        for i in range(scale-1, len(result_indexes), scale):
            # Swap the element at position i and i+step
            if i < len(result_indexes) - step:
                result_indexes[i], result_indexes[i + step] = result_indexes[i + step], result_indexes[i]
                result_indexes[i - 1], result_indexes[i + step-1] = result_indexes[i + step-1], result_indexes[i - 1]
    new_locations = []
    for i in result_indexes:
        new_locations.append(locations[i])
    delays = get_net_delay_matrix(new_locations)
    return new_locations, delays


def get_net_delay_matrix(locations):
    delays = [[0] * len(locations) for _ in range(len(locations))]
    for i in range(len(locations)):
        for j in range(len(locations)):
            d = get_distance(locations[i], locations[j])
            delays[i][j] = d
    return delays


def get_ordered_locations(locations, scale=0):
    """
    order the locations (coordinates) according to the sum distance among them.
    We start from the coordinate in index 0 them the following by the closest to the all
    selected ones, i.e, a coordinate loc_j is selected if the sum of the distances from loc_j to the all
    the already selected coordinates is lesser than all the sums from other coordinates not yet selected to
    the selected ones.

    Ex: 0  2  1  10 3
        2  0  9  2  50
        1  30 0  5  10
        10 2  5  0  2
        3  50 10 2  0
    first step: selected_pos=[0]
    followed by position 2 because the sum of distances from this position to the selected is 1
    then we have selected_pos=[0,2]
    Next: from pos 1 we have: sum=2+30=32; pos 3: sum= 10+5=15; pos 4: sum=3+10=13 result: selected_pos=[0,2,4]
    etc.
    """
    remain_indexes = [i + 1 for i in range(len(locations) - 1)]
    delays = get_net_delay_matrix(locations)

    result_indexes = [0]
    while len(remain_indexes) > 0:
        min_sum_delays = float('inf')
        min_index = -1
        for j in remain_indexes:
            delays_j = 0
            for i in result_indexes:
                delays_j = delays_j + delays[i][j]
            if min_sum_delays > delays_j:
                min_sum_delays = delays_j
                min_index = j
        result_indexes.append(min_index)
        remain_indexes.remove(min_index)
    if scale > 0:
        step = round(scale / 2)
        for i in range(scale-1, len(result_indexes), scale):
            # Swap the element at position i and i+step
            if i < len(result_indexes) - step:
                result_indexes[i], result_indexes[i + step] = result_indexes[i + step], result_indexes[i]
                result_indexes[i - 1], result_indexes[i + step-1] = result_indexes[i + step-1], result_indexes[i - 1]
    new_locations = []
    for i in result_indexes:
        new_locations.append(locations[i])
    delays = get_net_delay_matrix(new_locations)
    return new_locations, delays


def get_net_delay_matrix(locations):
    delays = [[0] * len(locations) for _ in range(len(locations))]
    for i in range(len(locations)):
        for j in range(len(locations)):
            d = get_distance(locations[i], locations[j])
            delays[i][j] = d
    return delays


def get_processed_gateways_and_servers(server_locations, cores, memory, gateways_qty=10):
    servers = get_static_severs_list(cores, memory, server_locations)
    gateways_servers = []
    gateway_locations = []
    for i in range(gateways_qty):
        gateways_servers.append(servers[i])
        gateway_locations.append(server_locations[i])
    return servers, gateways_servers, gateway_locations


def compute_total_delays(networks, coldstarts, placement):
    delays = np.zeros((len(networks), 2))
    for i in range(len(networks)):
        delays[i] = networks[i, 0], networks[i, 1]+coldstarts[i, 1] + placement[i, 1]
    return delays


def convert_dict_to_cfj(cfj_dict, str_functions, str_servers):
    cfj = np.zeros((len(str_functions), len(str_servers)))
    print(f'cfj_dict++++++={cfj_dict}')
    print(f'str_functions++++++={str_functions}')
    print(f'str_servers++++++={str_servers}')
    # Fill the matrix
    for func_key, server_dict in cfj_dict.items():
        if func_key in str_functions:
            func_index = str_functions.index(func_key)
            for server in server_dict.keys():
                if server in str_servers:
                    server_index = str_servers.index(server)
                    cfj[func_index, server_index] = 1
    print(f'cfj===={cfj}')
    return cfj
