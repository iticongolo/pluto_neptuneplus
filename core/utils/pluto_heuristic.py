import copy
import time
import json

import numpy as np
from core import *
from core.utils.util import *
from dynamic_clustering import DynamicClustering
from function import Function
from network_topology import Topology
from simulations.json_file_test import JsonfileTest as app
# from simulations.json_file_sockshop import JsonfileSockshop as app
# from simulations.json_file_complex import JsonfileComplex as app
# from simulations.json_file_test_complex import JsonfileTestComplex as app
#
# input = app.input

node_cpu_usage = None
node_gpu_usage = None
node_memory_usage = None


class PLUTO:

    def __init__(self, data):
        self.cfj = np.zeros((len(data.functions), len(data.nodes)))
        self.node_cpu_available = copy.deepcopy(data.node_cores)
        self.node_memory_available = copy.deepcopy(data.node_memories)
        self.data = data
        self.w = []
        self.x = []
        self.y = []
        self.z = []
        self.coldstart = 0
        self.total_delay = 0
        self.network_delay = 0

    # note SELECTED
    def get_functions(self):
        function = []
        for func in self.data.functions:
            function.append(func.split("/")[1])
        return function

    def compute_resources_contention(self, total_requests_per_f, total_cores_needed_per_f):
        # e.g.: total_cores_needed_per_f= [3000, 4000,1000]
        total_cores_needed = sum([cores_needed for cores_needed in total_cores_needed_per_f])
        cores_available = self.data.cluster.cores_available
        if total_cores_needed > cores_available:
            cores_weights_per_f = [cores_needed/total_cores_needed for cores_needed in total_cores_needed_per_f]
            total_cores_allocated_per_f =[cores_available*weight for weight in cores_weights_per_f]
            max_cores_per_request_f = [total_cores_allocated_per_f[i]/total_requests_per_f[i] for i in range(len(total_requests_per_f))]
            self.data.cores_cluster = np.array([[max_cores_per_request_f[i] for _ in self.data.cluster.servers] for i in range(len(self.data.functions))])

    def object_function_heuristic(self, w, x, y, z, lambd=np.array([]), instances=np.array([])):
        function = self.get_functions()
        qty_f = len(self.data.functions)
        cluster = self.data.cluster
        servers = cluster.servers

        qty_servers = len(cluster.servers)
        # print(f'network_delay ={qty_servers}')
        dag = self.data.dag
        m = self.data.m
        network_delay = self.data.cluster.network_delays

        lamb = self.data.workload_on_source_matrix
        # print(f'network_delay ={lamb}')

        if len(lambd) > 0:
            lamb = lambd

        # print(f'lamb ={lamb }')
        cfj_temp = copy.deepcopy(self.cfj)
        if len(instances) > 0:
            cfj_temp = copy.deepcopy(instances)
        networkd=0
        for f in range(qty_f):
            temp_network_delay = 0
            temp_total_delay = 0
            for s in servers:
                i = get_position(servers, s.id)
                if lamb[f, i] > 0:
                    for s1 in servers:
                        j = get_position(servers, s1.id)
                        # print(f'servers={len(servers)}')
                        # print(f'network_delay[{i},{j}]={network_delay}')
                        temp_network_delay = temp_network_delay + network_delay[i][j] * x[f, i, j] * lamb[f, i]

                if cfj_temp[f, i]:
                    cfj_temp[f, i] = 0
                    temp_total_delay = temp_total_delay + temp_network_delay

                for s1 in servers:
                    j = get_position(servers, s1.id)
                    sum_fs = 0
                    seq_successor = dag.get_sequential_successors_indexes(function[f])
                    for fs in seq_successor:
                        delay_y = y[f, i, fs, j] * m[f][fs] * w[f, i]
                        if cfj_temp[fs, j]:
                            cfj_temp[fs, j] = 0
                        sum_fs = sum_fs + delay_y
                    # print(f'network_delay[C{cluster.id},{[i,j]}]={network_delay}')
                    if sum_fs * network_delay[i][j] > 0:
                        temp_network_delay = temp_network_delay + sum_fs * network_delay[i][j]

                parallel_successors_groups = dag.get_parallel_successors_indexes(function[f])

                for par_group in parallel_successors_groups:
                    max_network_delay_z = float('-inf')
                    for fp in par_group:
                        delay_z = 0
                        for s1 in servers:
                            j = get_position(servers, s1.id)
                            delay_z = delay_z + z[f, i, fp, j] * m[f][fp] * w[f, i] * network_delay[i][j]
                        if delay_z > max_network_delay_z:
                            max_network_delay_z = delay_z

                    if max_network_delay_z > float('-inf'):
                        temp_network_delay = temp_network_delay + max_network_delay_z
                    for fp in par_group:
                        for s1 in servers:
                            j = get_position(servers, s1.id)
                            cfj_temp[fp, j] = 0
            print(f'ABB={networkd} + {temp_network_delay}')
            self.network_delay = self.network_delay + temp_network_delay

            self.total_delay = self.coldstart + self.network_delay
        return self.total_delay, self.coldstart, self.network_delay

    def get_closest_available_server(self, f, mf, i):
        # NOTE: the delay matrix is not composed by sequential servers ids. e.g.: Cluster[2,6,1,9]
        #  delay matrix [[0,2,1,5], [2,0,3,4], [1,3,0,7], [5,4,7,0]] therefore,
        #  matrix[0,2]=1 is de delay between servers 2 (position 0 on the cluster) and 1 (position 2 on the cluster),
        #  so, we always take the server on the correspondent position of the cluster

        selected_server_position = -1
        cpu = 0.0
        memory = 0.0
        min_delay = float('inf')
        candidate_servers = []
        # less_used_servers = []
        cluster = self.data.cluster
        servers = cluster.servers

        max_delay_f = self.data.max_delay_matrix
        # print(f'max_delay_f={max_delay_f}')
        # node_delay = self.data.node_delay_matrix
        node_delay = self.data.cluster.network_delays
        f_placed = False
        # prepare a list of nodes with available cores and memory
        # for server in servers:
        #     cores_available, memory_available = cluster.get_server_available_resources(server)
        #     print(f'Server[{server.id}]={cores_available, memory_available}')
        for server in servers:
            cores_available, memory_available = cluster.get_server_available_resources(server)
            # print(f'mf[{sf}]={mf}')
            if cores_available > 0 and memory_available >= mf[f]:
                candidate_servers.append(server)

        previous_nodes = self.get_servers_with_instance_of_f(f)
        if len(previous_nodes) > 0:
            for p_node in previous_nodes:
                cores_available, _ = cluster.get_server_available_resources(p_node)
                if cores_available > 0:
                    f_placed = True
                    break

        # if no greater differences we select the present node_i if available or anyone with min network delay
        pos = get_position(candidate_servers, servers[i].id)

        if pos >= 0:  # the current server (in position i) exists in the list of candidates
            selected_server_position = i
        else:
            for s in candidate_servers:
                j = get_position(servers, s.id)
                # print(f'node_delay={node_delay}')
                if min_delay > node_delay[i][j]:
                    min_delay = node_delay[i][j]
                    selected_server_position = j
            if selected_server_position >= 0:
                # select all closest nodes
                candidates = [] # collect all the servers with the minimal network delay from the server i receiving workload
                for s in candidate_servers:
                    j = get_position(servers, s.id)
                    if node_delay[i][selected_server_position] == node_delay[i][j]:
                        candidates.append(s)
                most_resources = 0
                used_server_position = -1
                node_found = False

                # select the node where the instance is already placed to only increase the workload and save energy
                for s in candidates:
                    pos = get_position(servers, s.id)
                    if self.cfj[f, pos]:  # from the candidates select a server that has an instance of f
                        selected_server_position = pos
                        node_found = True
                        break
                    if self.has_instance(s):  # if no instance of f was placed, select a server that has at least one instance
                        used_server_position = pos

                if not node_found:
                    if used_server_position > -1:
                        selected_server_position = used_server_position
                    # select a server with the highest amount of resources
                    else:
                        for s in candidates:
                            pos = get_position(servers, s.id)
                            cores_available, _ = cluster.get_server_available_resources(s)
                            if cores_available > most_resources:
                                most_resources = cores_available
                                selected_server_position = pos
        #
        # # If the list of less used nodes is not empty then we select the least used node
        # self.v=self.v+1
        # if self.v==20: return
        if selected_server_position >= 0:
            cpu, memory = cluster.get_server_available_resources(servers[selected_server_position])
        return selected_server_position, cpu, memory, f_placed

    # note SELECTED
    def fill_x(self):
        cluster = self.data.cluster
        # print(f'len(self.data.functions)={len(self.data.functions)}, len(cluster.servers)={len(cluster.servers)}')
        x = np.zeros((len(self.data.functions), len(cluster.servers), len(cluster.servers)))
        w = np.zeros((len(self.data.functions), len(cluster.servers)))
        workload = self.data.workload_on_source_matrix
        perc_used_cpu = np.zeros(len(cluster.servers))
        _, functions, _, mf, servers, ufj, node_cpu = self.basic_fill_data()
        print(f'ufj={ufj}')
        for f in range(len(functions)):
            for s in servers:
                i = get_position(servers, s.id)
                lamb = workload[f][i]
                if lamb > 0:
                    cpu_requested = lamb*ufj[f, i]  # here xfij is 1
                    cpu_requested1 = copy.deepcopy(cpu_requested)
                    memory_requested = mf[f]  # here xfij is 1
                    memory_requested1 = copy.deepcopy(memory_requested)
                    allocation_finished = False
                    # print(f'cpu_requested1={cpu_requested1}')
                    while cpu_requested1 > 0:  # we can use cpu_requested or memory_requested
                        # print(f'cpu_requested1={cpu_requested1}')
                        if allocation_finished:
                            break
                        closest_server_position, cores_available, memory_available, f_placed = \
                            self.get_closest_available_server(f, mf, i)
                        # print(f'[closest_server_position, cores_available, memory_available, f_placed]={[closest_server_position, cores_available, memory_available, f_placed]}')
                        # print(f'closest_server_position(X)={closest_server_position}')
                        if closest_server_position < 0:
                            raise Exception("The nodes are overloaded, no more resources to be allocated!")
                        # print(f'closest_server[{closest_server_position}]={cores_available, memory_available}')

                        cores_capacity, _ = cluster.get_server_allocated_resources(servers[closest_server_position])

                        diff_cpu = cores_available - cpu_requested1
                        diff_memory = memory_available
                        if not f_placed:
                            diff_memory = memory_available - memory_requested1  # get_closest_available_server
                        # returns node with enough memory
                        if diff_cpu >= 0:
                            if cpu_requested1 == cpu_requested:
                                x[f, i, closest_server_position] = 1.0
                            else:
                                x[f, i, closest_server_position] = cpu_requested1/cpu_requested  # x[f, i, closest_node] +
                            new_cores = -cores_available+diff_cpu
                            cluster.update_server_available_resources(servers[closest_server_position], new_cores=new_cores) # available cores=diff_cpu
                            w[f, closest_server_position] = w[f, closest_server_position] + x[f, i, closest_server_position] * lamb
                            perc_used_cpu[closest_server_position] = perc_used_cpu[closest_server_position] + cpu_requested1 / cores_capacity
                            allocation_finished = True
                        else:
                            perc_cpu = cores_available / cpu_requested
                            cluster.update_server_available_resources(servers[closest_server_position], new_cores=-cores_available) # remove all the available cores
                            x[f, i, closest_server_position] = perc_cpu

                            w[f, closest_server_position] = w[f, closest_server_position] + x[f, i, closest_server_position] * lamb
                            perc_used_cpu[closest_server_position] = perc_used_cpu[closest_server_position] + (cpu_requested1*perc_cpu) / cores_capacity
                            cpu_requested1 = cpu_requested1 - cores_available
                        if not self.cfj[f, closest_server_position]:
                            new_memory = -memory_available + diff_memory
                            cluster.update_server_available_resources(servers[closest_server_position], new_memory=new_memory)
                            self.cfj[f, closest_server_position] = True
        # print(f'x={x}, w={w},  perc_used_cpu={perc_used_cpu}')
        # print(f'||||||| perc_used_cpu-X={perc_used_cpu}|| node_cpu={node_cpu}')
        # print(f'[s,cores,memory]={[[s.id, cluster.get_server_available_resources(s)] for s in cluster.servers]}')
        return x, w,  perc_used_cpu

    def fill_y(self, f, y, w, perc_used_cpu):
        dag, functions, m, mf, nodes, ufj, node_cpu = self.basic_fill_data()
        seq_successor = dag.get_sequential_successors_indexes(functions[f])
        cluster = self.data.cluster
        servers= cluster.servers
        if len(seq_successor) > 0:
            for i in range(len(nodes)):
                omega = w[f, i]
                if omega > 0:
                    for fs in seq_successor:
                        omega1 = copy.deepcopy(omega)
                        for j in range(len(nodes)):
                            if omega1 > 0:
                                cpu_requested = omega1 * m[f][fs] * ufj[fs, j]  # here y[f,i,fs,j] is 1
                                memory_requested = mf[fs]  ## here y[f,i,fs,j] is 1
                                cpu_requested1 = copy.deepcopy(cpu_requested)
                                allocation_finished = False
                                while cpu_requested1 > 0:  # we can use cpu_requested or memory_requested
                                    if allocation_finished:
                                        break
                                    closest_server_position, cores_available, memory_available, f_placed = self.get_closest_available_server(f,  mf, i)
                                    if closest_server_position < 0:
                                        raise Exception("The nodes are overloaded, no more resources to be allocated!")
                                    print(f'node_cpu_available[{closest_server_position}]={self.node_cpu_available}')

                                    cores_capacity, _ = cluster.get_server_allocated_resources(
                                        servers[closest_server_position])

                                    diff_cpu = cores_available - cpu_requested1
                                    # get_closest_available_server returns node with enough memory
                                    diff_memory = memory_available
                                    if not f_placed:
                                        diff_memory = memory_available - memory_requested  # get_closest_available_server
                                    if diff_cpu >= 0:
                                        if cpu_requested1 == cpu_requested:
                                            y[f, i, fs, closest_server_position] = 1.0
                                        else:
                                            y[f, i, fs, closest_server_position] = cpu_requested1 / cpu_requested # y[f, i, fs, closest_node] +

                                        new_cores = -cores_available + diff_cpu
                                        cluster.update_server_available_resources(servers[closest_server_position], new_cores=new_cores)  # available cores=diff_cpu
                                        w[fs, closest_server_position] = w[fs, closest_server_position] + y[f, i, fs, closest_server_position] * omega1 * m[f][fs]
                                        perc_used_cpu[closest_server_position] = perc_used_cpu[closest_server_position] + cpu_requested1 / cores_capacity
                                        allocation_finished = True
                                    else:
                                        perc_cpu = cores_available / cpu_requested
                                        cluster.update_server_available_resources(servers[closest_server_position],
                                                                                  new_cores=-cores_available)  # remove all the available cores
                                        y[f, i, fs, closest_server_position] = perc_cpu
                                        cpu_requested1 = cpu_requested * (1 - perc_cpu)
                                        w[fs, closest_server_position] = w[fs, closest_server_position] + y[f, i, fs, closest_server_position] * omega1 * m[f][fs]
                                        perc_used_cpu[closest_server_position] = perc_used_cpu[closest_server_position] + (cpu_requested1*perc_cpu)/ cores_capacity
                                        cpu_requested1 = cpu_requested1 - cores_available
                                    if not self.cfj[fs, closest_server_position]:
                                        new_memory = -memory_available + diff_memory
                                        cluster.update_server_available_resources(servers[closest_server_position],
                                                                                  new_memory=new_memory)
                                        self.cfj[fs, closest_server_position] = True
                            omega1 = 0
        return y, w, perc_used_cpu

    def fill_z(self, f, z, w, perc_used_cpu):
        dag, functions, m, mf, servers, ufj, node_cpu = self.basic_fill_data()
        parallel_successors_groups = dag.get_parallel_successors_indexes(functions[f])
        cluster = self.data.cluster

        if len(parallel_successors_groups) > 0:
            for i in range(len(servers)):
                omega = w[f, i]
                if omega > 0:
                    for par_group in parallel_successors_groups:
                        # print(f'par_group={par_group}, parallel_successors_group={parallel_successors_groups}')
                        for fp in par_group:
                            omega1 = copy.deepcopy(omega)
                            for j in range(len(servers)):
                                if omega1 > 0:
                                    cpu_requested = omega1 * m[f][fp] * ufj[fp, j]  # here z[f,i,fp,j] is 1
                                    memory_requested = mf[fp]  ## here z[f,i,fp,j] is 1
                                    cpu_requested1 = copy.deepcopy(cpu_requested)
                                    memory_requested1 = copy.deepcopy(memory_requested)
                                    allocation_finished = False
                                    while cpu_requested1 > 0:  # we can use cpu_requested or memory_requested
                                        if allocation_finished:
                                            break
                                        closest_server_position, cores_available, memory_available, f_placed = self.get_closest_available_server(f,  mf, i)
                                        if closest_server_position < 0:
                                            raise Exception(f'The nodes are overloaded, no more '
                                                            f'resources to be allocated! You are trying to allocate '
                                                            f'{cpu_requested1} cores in [{self.node_cpu_available}] and {memory_requested1}MB in [{self.node_memory_available}]')

                                        cores_capacity, _ = cluster.get_server_allocated_resources(
                                            servers[closest_server_position])

                                        diff_cpu = cores_available - cpu_requested1
                                        # get_closest_available_server returns node with enough memory
                                        diff_memory = cores_available
                                        if not f_placed:
                                            diff_memory = cores_available - memory_requested1  # get_closest_available_server
                                        if diff_cpu >= 0:
                                            if cpu_requested1 == cpu_requested:
                                                z[f, i, fp, closest_server_position] = 1.0
                                            else:
                                                z[f, i, fp, closest_server_position] = z[f, i, fp, closest_server_position] + cpu_requested1 / cpu_requested

                                            new_cores = -cores_available + diff_cpu
                                            cluster.update_server_available_resources(servers[closest_server_position],
                                                                                      new_cores=new_cores)  # available cores=diff_cpu
                                            w[fp, closest_server_position] = w[fp, closest_server_position] + z[f, i, fp, closest_server_position] * omega1 * m[f][fp]
                                            perc_used_cpu[closest_server_position] = perc_used_cpu[closest_server_position] + cpu_requested1 / cores_capacity
                                            allocation_finished = True
                                        else:
                                            perc_cpu = cores_available / cpu_requested
                                            cluster.update_server_available_resources(servers[closest_server_position],
                                                                                      new_cores=-cores_available)  # remove all the available cores
                                            z[f, i, fp, closest_server_position] = perc_cpu
                                            w[fp, closest_server_position] = w[fp, closest_server_position] + z[f, i, fp, closest_server_position] * omega1 * m[f][fp]
                                            perc_used_cpu[closest_server_position] = perc_used_cpu[closest_server_position] + (cpu_requested1*perc_cpu)/cores_capacity
                                            cpu_requested1 = cpu_requested1 - cores_available
                                        if not self.cfj[fp, closest_server_position]:
                                            new_memory = -memory_available + diff_memory
                                            cluster.update_server_available_resources(servers[closest_server_position],
                                                                                      new_memory=new_memory)
                                            self.cfj[fp, closest_server_position] = True
                                    omega1 = 0
        return z, w, perc_used_cpu

    def heuristic_placement(self, perc_node_resources_x=0.2, perc_node_resources_y=0.2, perc_node_resources_z=0.2):
        cluster = self.data.cluster
        start_time = time.time()
        _, functions, _, _, _, _, _ = self.basic_fill_data()
        print('FFFFFF0000')
        x, w,  perc_used_cpu = self.fill_x()
        print('FFFFFF')
        y = np.zeros((len(self.data.functions), len(cluster.servers), len(self.data.functions), len(cluster.servers)))
        z = np.zeros((len(self.data.functions), len(cluster.servers), len(self.data.functions), len(cluster.servers)))
        for f in range(len(functions)):
            y, w,  perc_used_cpu = self.fill_y(f, y, w, perc_used_cpu)
            z, w,  perc_used_cpu = self.fill_z(f, z, w, perc_used_cpu)
        end_time = time.time()
        self.set_coldstart()
        # Calculate the elapsed time
        decision_time = (end_time - start_time)*1000  # milliseconds
        return x, y, z, w, self.node_cpu_available, self.node_memory_available, self.cfj, decision_time

    def resource_usage(self):
        total_nodes = 0
        memory = 0
        cpus = 0
        memories = []
        cores = []
        nodes = len(self.cfj[0])
        functions = len(self.cfj)
        for i in range(nodes):
            memories.append(self.data.node_memories[i]-self.node_memory_available[i])
            cores.append(self.data.node_cores[i] - self.node_cpu_available[i])
            for f in range(functions):
                if self.cfj[f, i]:
                    total_nodes = total_nodes+1
                    break
        for m in memories:
            memory = memory+m
        for cpu in cores:
            cpus = cpus + cpu
        return total_nodes, memory, cpus

    def basic_fill_data(self):
        functions = self.get_functions()
        servers = self.data.cluster.servers
        dag = self.data.dag
        ufj = self.data.cores_cluster
        mf = self.data.function_memories
        m = self.data.m
        node_cpu = self.data.node_cores
        return dag, functions, m, mf, servers, ufj, node_cpu

    def has_instance(self, server):
        _, memory_capacity = self.data.cluster.get_server_allocated_resources(server)
        _, memory_available = self.data.cluster.get_server_available_resources(server)
        return memory_available-memory_capacity < 0

    def get_servers_with_instance_of_f(self, f):
        servers = []
        pos = 0
        for server in self.data.cluster.servers:
            if self.cfj[f, pos]:
                servers.append(server)
            pos = pos+1
        return servers

    def set_coldstart(self):
        cold_starts = self.data.function_cold_starts
        max_coldstart = 0
        for f in range(len(cold_starts)):
            if cold_starts[f] > max_coldstart:
                max_coldstart = cold_starts[f]
        self.coldstart = max_coldstart


# class FunctionData:
#     def __init__(self, id, data):
#         self.id = id
#         self.data = data

# Generate the network topology
servers_location = [(3.75, 14.5), (4.5, 11.25), (8.5, 13.5), (13.75, 6.25), (14.5, 3.0), (19.5, 2.0), (21.5, 4.5),
                    (17.5, 5.5), (15.75, 15.0), (19.75, 14.5), (22.5, 17.5), (24.75, 15.0), (22.5, 12.5)]
servers = get_severs_list(total_servers=13, cores=2000, memory=4000, location=servers_location)

topology = Topology(servers=servers)
topology.generate_network_delays()

# for node in topology.nodes:
#     print(f'Node{node.id}-Location={node.location}')
clusters = topology.generate_balanced_clusters(delay_threshold=5)

for c in topology.current_clusters:
    c.compute_network_delays(topology.network_delays)

functions_names = ['f0','f1','f2','f3','f4']
# functions_names = ['f0', 'f1']
functions=[]
for i in range(len(functions_names)):
    functions.append(Function(i, name=functions_names[i]))

for c in clusters:
    c.functions = functions

topology.initial_clusters = copy.deepcopy(clusters)
topology.current_clusters = clusters

dynamic_clustering = DynamicClustering(topology, functions)

# [c0[id_f|data_frame|...],... ck[id_f|data_frame|...]]
list_historical_workload = [[] for _ in range(len(clusters))]
list_predicted_workload = [[] for _ in range(len(clusters))]

# NOTE New ------------------------------------
applications_list = []
data_list = []
servers_ids = [0, 0, 8, 8]
function_ids = [0, 1, 0, 1]
lambs = [400, 300, 300, 100]

# servers_ids = [0, 0, 8, 8]
# function_ids = [0, 1, 0, 1]
# lambs = [5000, 0, 0, 0]

for c in clusters:
    data = Data()
    applications_list.append(app)
    input = app.input
    data.cluster = c
    data.nodes = [server for server in c.servers]
    # print(f'{c.functions}')
    setup_community_data(data, input)
    setup_runtime_data(data, input)
    data_list.append(data)



# NOTE New ------------------------------------

# workloads[c_id][pos_server][function_ids[i]]
# e.g.: workloads= [[c0|[s3|[f0|0, f1|0, ... f5|0], s0[f0|400, f1|100, ... f5|0]], ..., s8[f0|200, f1|50, ... f5|0]]]

workloads = set_workload(dynamic_clustering, topology, data_list, servers_ids, function_ids,
                                         lambs, applications_list)


# delays = np.zeros((7,2))
# qty_values = 1
# topology_position = 0   # select the topology
# # app.set_topology(app, topology_position)
# input = app.input
# param.set_topology(param, topology_position, input)
# f = len(input["function_names"])
# nod = len(input["node_names"])
# workload = param.workload_init(param, f, nod)

# Vary the workload from 50 to 300
# for i in range(7):
#     if i == 0:
#         lamb = 10
# else:
#     lamb = 50 * i
# lamb = 200
# workload[0][0] = lamb
# print(f'workload={workload}')
#
# json_workload = json.dumps(workload)
# input["workload_on_source_matrix"] = json_workload
# print(f'workload1={json_workload}')
# data = Data()
# setup_community_data(data, input)
# setup_runtime_data(data, input)
# print(f'Nodes={data.nodes}')

update_data(json, data_list, workloads, applications_list)
perc_workload_balance = 1.0

i = 0
delays=[]

for c in clusters:
    asc = PLUTO(data_list[i])
    asc.compute_resources_contention(c.total_workload_per_f, c.cores_needed_per_f)
    # for server in c.servers:
    #     cores_available, memory_available = c.get_server_available_resources(server)
    #     print(f'ServerAAA[{server.id}]={cores_available, memory_available}')

    x, y, z, w, node_cpu_available, node_memory_available, instance_fj, decision_time = \
        asc.heuristic_placement(perc_workload_balance, perc_workload_balance, perc_workload_balance)
    total_delay, cold_start, network_delay = asc.object_function_heuristic(w, x, y, z)
    delays.append(f'C[{c.id}], {total_delay}')
    # print(f'C[{c.id}], {cold_start}')
    i = i+1
np.set_printoptions(threshold=np.inf)

print(f'DELAYS={delays}')
# print(f'x={x}')
# print('++++++++++++++++++++++++++++++++++++++++++++++++')
#
# # print(f'y={y}')
#
# print('++++++++++++++++++++++++++++++++++++++++++++++++')
#
# # print(f'z={z}')
# print('+++++++++++++++++++++Y+++++++++++++++++++++++++++')
# print(f'X={x}')
#
# print('+++++++++++++++++++++Y+++++++++++++++++++++++++++')
# for f in range(len(data.functions)):
#     stop=False
#     for fs in range(len(data.functions)):
#         for i in range(len(data.nodes)):
#             for j in range(len(data.nodes)):
#                 if y[f,i,fs,j]>0:
#                     print(f'y[f{f}, Node{i}, fs{fs}]')
#                     print(y[f,i,fs])
#                     stop=True
#                     break
#
# print('+++++++++++++++++++++Z+++++++++++++++++++++++++++')
# for f in range(len(data.functions)):
#     stop=False
#     for fs in range(len(data.functions)):
#         for i in range(len(data.nodes)):
#             for j in range(len(data.nodes)):
#                 if z[f,i,fs,j]>0:
#                     print(f'z[f{f}, Node{i}, fs{fs}]')
#                     print(z[f,i,fs])
#                     stop=True
#                     break
#
#
# print('+++++++++++++++++++++W+++++++++++++++++++++++++++')
# print(f'w={w}')
#
# print('++++++++++++++++++++++++++++++++++++++++++++++++')
# print(f' instance_fj')
# print(f'{instance_fj}')
#
# print(f'Total Delay/ColdStart/Network delay/decisionTime={total_delay}/{cold_start}/{network_delay}/{decision_time}')
