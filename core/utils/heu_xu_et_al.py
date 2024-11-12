# class HeuXu
import copy

import numpy as np
import time
from core.utils.pluto_heuristic import PLUTO

node_cpu_usage = None
node_gpu_usage = None
node_memory_usage = None

class HeuXu:
    def __init__(self, data):
        self.cfj = np.zeros((len(data.functions), len(data.nodes)))
        self.node_cpu_available = copy.deepcopy(data.node_cores)
        self.node_memory_available = copy.deepcopy(data.node_memories)
        self.data = data
        self.list_r_cfj = []
        self.w = []
        self.x = []
        self.y = []
        self.z = []
        self.coldstart = 0
        self.total_delay = 0
        self.network_delay = 0
        self.last_used_node = -1

    def fill_w(self, request):  # will fill internal workload for each single request
        dag, functions, m, mf, nodes, ufj, node_cpu = self.basic_fill_data()
        lamb = copy.deepcopy(self.data.workload_on_source_matrix[request])
        for f in range(len(functions)):
            for i in range(len(nodes)):
                if lamb[f, i] > 0.001:
                    seq_successor = dag.get_sequential_successors_indexes(functions[f])
                    if len(seq_successor) > 0:
                        for fs in seq_successor:
                            lamb[fs, i] = lamb[fs, i] + lamb[f, i] * m[f][fs]
                    group_par_successor = dag.get_parallel_successors_indexes(functions[f])
                    if len(group_par_successor) > 0:
                        for group in group_par_successor:
                            for fp in group:
                                lamb[fp, i] = lamb[fp, i] + lamb[f, i] * m[f][fp]
        self.w.append(lamb)

    def basic_fill_data(self):
        functions = self.get_functions()
        nodes = self.data.nodes
        dag = self.data.dag
        ufj = self.data.core_per_req_matrix
        mf = self.data.function_memories
        m = self.data.m
        node_cpu = self.data.node_cores
        return dag, functions, m, mf, nodes, ufj, node_cpu

    # first step
    def place_app(self, request, i,  parallel_scheduler):
        r_cfj = np.zeros((len(self.data.functions), len(self.data.nodes)))
        new_instances = []
        first_layer = parallel_scheduler[0]

        # start_node = self.get_start_node(i, request, first_layer)
        for layer in parallel_scheduler:
            memory_required = self.get_total_memory(layer)  # only used in case of exception
            cpu_required = self.get_cpu_demand(layer, request, i)
            j, remain_cpu, remain_memory = self.get_closest_available_node(self.node_cpu_available,
                                                                             self.node_memory_available, layer, cpu_required,i)
            if j < 0:
                raise Exception(f'The nodes are overloaded, no more '
                                f'resources to be allocated! You are trying to allocate '
                                f'{cpu_required} millicores in [{self.node_cpu_available}] and {memory_required}MB in '
                                f'[{self.node_memory_available}]')
            for f in layer:
                r_cfj[f, j] = 1
                if not self.cfj[f, j]:
                    self.cfj[f, j] = 1
                    new_instances.append(f)
            self.node_cpu_available[j] = remain_cpu
            self.node_memory_available[j] = remain_memory
        self.list_r_cfj.append(r_cfj)
        max_cold_start = self.get_max_coldstart(new_instances)
        self.coldstart = self.coldstart + max_cold_start

    # note SELECTED
    def get_functions(self):
        function = []
        for func in self.data.functions:
            function.append(func.split("/")[1])
        return function

    def object_function_heu(self, request):
        dag, functions, m, mf, nodes, ufj, node_cpu = self.basic_fill_data()
        lamb = copy.deepcopy(self.data.workload_on_source_matrix[request])
        function = self.get_functions()
        qty_f = len(self.data.functions)
        qty_nodes = len(self.data.nodes)
        dag = self.data.dag
        network_delay = self.data.node_delay_matrix
        x = self.x[request]
        y = self.y[request]
        z = self.z[request]
        w = self.w[request]
        # print(f'self.w[{request}]={self.w[request]}')

        sum_f = 0
        for f in range(qty_f):
            sum_i = 0
            for i in range(qty_nodes):
                if lamb[f, i] > 0.001:
                    for j in range(qty_nodes):
                        sum_f = sum_f + network_delay[i][j] * x[f, i, j] * lamb[f, i]

                sum_sequential = 0
                for j in range(qty_nodes):
                    sum_fs = 0
                    seq_successor = dag.get_sequential_successors_indexes(function[f])
                    for fs in seq_successor:
                        # delay_y = y[f, i, fs, j]*m[f][fs]*w[f, i]*nrt[fs]
                        delay_y = y[f, i, fs, j] * m[f][fs] * w[f, i]

                        sum_fs = sum_fs + delay_y
                    if sum_fs * network_delay[i][j] > 0:
                        sum_sequential = sum_sequential + sum_fs * network_delay[i][j]
                parallel_successors_groups = dag.get_parallel_successors_indexes(function[f])
                sum_parallel = 0
                for par_group in parallel_successors_groups:
                    max_delay_z = float('-inf')
                    for fp in par_group:
                        for j in range(qty_nodes):
                            # delay_z = z[f, i, fp, j] * m[f][fp] * w[f, i] * nrt[fp] * network_delay[i][j]
                            delay_z = z[f, i, fp, j] * m[f][fp] * w[f, i] * network_delay[i][j]
                            if delay_z > max_delay_z:
                                max_delay_z = delay_z

                    if max_delay_z > float('-inf'):
                        sum_parallel = sum_parallel + max_delay_z
                sum_i = sum_i + sum_sequential + sum_parallel
            sum_f = sum_f + sum_i
            return sum_f

    def object_function_global_results(self):
        # self.data.function_cold_starts
        aseco = PLUTO(self.data)
        x, y, z = self.compose_xyz()
        w = self.compose_w(x, y, z)


        lamb = np.zeros((len(self.data.functions), len(self.data.nodes)))
        lamb_aux = self.data.workload_on_source_matrix
        for lam in lamb_aux:
            lamb = lamb + np.array(lam)
        _, _, network_delay = aseco.object_function_heuristic(w, x, y, z, lambd=lamb, instances=self.cfj)

        return self.coldstart+network_delay, self.coldstart, network_delay
    def compose_xyz(self):

        x = np.zeros((len(self.data.functions), len(self.data.nodes), len(self.data.nodes)))
        y = np.zeros((len(self.data.functions), len(self.data.nodes), len(self.data.functions),len(self.data.nodes)))
        z = np.zeros((len(self.data.functions), len(self.data.nodes), len(self.data.functions), len(self.data.nodes)))
        w = np.zeros((len(self.data.functions), len(self.data.nodes)))
        for x_aux in self.x:
            x = x+np.array(x_aux)
        x_aux = copy.deepcopy(x)

        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                total_xfi= np.sum(x_aux[f,i])
                if total_xfi > 0:
                    for j in range(len(self.data.nodes)):
                        x[f,i,j]=x[f,i,j]/total_xfi

        for y_aux in self.y:
            y = y + np.array(y_aux)

        y_aux = copy.deepcopy(y)
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                for fs in range(len(self.data.functions)):
                    total_yfifs = np.sum(y_aux[f, i, fs])
                    if total_yfifs > 0:
                        for j in range(len(self.data.nodes)):
                            y[f, i, fs, j] = y[f, i, fs, j]/total_yfifs
        for z_aux in self.z:
            z = z + np.array(z_aux)

        z_aux = copy.deepcopy(z)
        for f in range(len(self.data.functions)):
            for i in range(len(self.data.nodes)):
                for fp in range(len(self.data.functions)):
                    total_zfifp = np.sum(z_aux[f, i, fp])
                    if total_zfifp > 0:
                        for j in range(len(self.data.nodes)):
                            z[f, i, fp, j] = z[f, i, fp, j] / total_zfifp
        for w_aux in self.w:
            w = w + np.array(w_aux)

        # x = x/lamb
        # y = y/lamb
        # z = z/lamb
        # print(f'x={x}')
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++')
        # print(f'y={y}')
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++')
        # print(f'z={z}')
        # print(f'primer w={w}')
        return x, y, z
    def compose_w(self, x, y, z):
        # print(f'x={x}')
        # print(f'y={y}')
        # print(f'z={z}')
        w = np.zeros((len(self.data.functions), len(self.data.nodes)))
        dag, functions, m, mf, nodes, ufj, node_cpu = self.basic_fill_data()
        workload = np.zeros((len(self.data.functions), len(self.data.nodes)))
        for lamb in self.data.workload_on_source_matrix:
            workload = workload + np.array(lamb)

        for f in range(len(functions)):
            for i in range(len(nodes)):
                lamb = workload[f][i]
                if lamb > 0:
                   for j in range(len(nodes)):
                            w[f, j] = w[f, j] + x[f, i, j] * lamb
        for f in range(len(functions)):
            seq_successor = dag.get_sequential_successors_indexes(functions[f])
            # print(seq_successor)
            for i in range(len(nodes)):
                if w[f, i] > 0:
                    for fs in seq_successor:
                        for j in range(len(nodes)):
                            w[fs, j] = w[fs, j] + y[f, i, fs, j] * w[f, i] * m[f][fs]

        # for f in range(len(functions)):
            parallel_successors_groups = dag.get_parallel_successors_indexes(functions[f])
            print(f'P[{functions[f]}]={parallel_successors_groups}')
            for i in range(len(nodes)):
                if w[f, i] > 0:
                    for par_group in parallel_successors_groups:
                        for fp in par_group:
                            for j in range(len(nodes)):
                                w[fp, j] = w[fp, j] + z[f, i, fp, j] * w[f, i] * m[f][fp]
        return w

    # note: SELECTED
    # we assume node as a cloudlet to be aligned with the assumption of Xu et. al (2023)
    # We consider available only the nodes with enough capacity to place all the
    # functions in a given layer-list_f ( independent functions)
    # GREEDY SEARCH - is any approximation to get the shortest path. held_karp algorithm
    # gives the optimal but has higher computing cost
    def get_closest_available_node(self, node_cpu_available, node_memory_available,
                                   list_f, cpu_demand, i):
        selected_node = -1
        cpu = 0.0
        memory = 0.0
        min_delay = float('inf')
        nodes = self.data.nodes
        candidate_nodes = []
        node_delay = self.data.node_delay_matrix
        # prepare a list of nodes with available cores and memory
        for j in range(len(nodes)):
            if node_cpu_available[j] >= cpu_demand:
                if node_memory_available[j] >= self.get_memory(list_f, j):
                    candidate_nodes.append(j)
        if i in candidate_nodes:
            selected_node = i
        else:
            for j in candidate_nodes:
                if min_delay > node_delay[i][j]:
                    min_delay = node_delay[i][j]
                    selected_node = j

        if selected_node >= 0:
            cpu = node_cpu_available[selected_node]-cpu_demand
            memory = node_memory_available[selected_node]
            if len(list_f) > 0:
                f = list_f[0]
                if not self.cfj[f, selected_node]:
                    memory = memory - self.get_memory(list_f, selected_node)
        return selected_node, cpu, memory

    def fill_x(self,  cfj_r, lamb_r):
        functions = self.get_functions()
        nodes = self.data.nodes
        x_r = np.zeros((len(functions), len(nodes), len(nodes)))
        for f in range(len(functions)):
            for i in range(len(nodes)):
                if lamb_r[f, i] > 0.001:
                    for j in range(len(nodes)):
                        if cfj_r[f, j]:
                            x_r[f, i, j] = 1
        self.x.append(x_r)

    def fill_y(self, cfj_r):
        functions = self.get_functions()
        nodes = self.data.nodes
        dag = self.data.dag
        y_r = np.zeros((len(functions), len(nodes), len(functions), len(nodes)))
        for f in range(len(functions)):
            seq_successor = dag.get_sequential_successors_indexes(functions[f])
            if len(seq_successor) > 0:
                for i in range(len(nodes)):
                    if cfj_r[f, i]:
                        for fs in seq_successor:
                            for j in range(len(nodes)):
                                if cfj_r[fs, j]:
                                    y_r[f, i, fs, j] = 1
        self.y.append(y_r)

    def fill_z(self, cfj_r):
        functions = self.get_functions()
        nodes = self.data.nodes
        dag = self.data.dag
        z_r = np.zeros((len(functions), len(nodes), len(functions), len(nodes)))
        for f in range(len(functions)):
            par_successor = dag.get_parallel_successors_indexes(functions[f])
            if len(par_successor) > 0:
                for i in range(len(nodes)):
                    if cfj_r[f, i]:
                        for group in par_successor:
                            for fp in group:
                                for j in range(len(nodes)):
                                    if cfj_r[fp, j]:
                                        z_r[f, i, fp, j] = 1
        self.z.append(z_r)

    def fill_xyz(self, request):
        dag, functions, _, _, nodes, _, _ = self.basic_fill_data()
        cfj_r = self.list_r_cfj[request]
        lamb_r = copy.deepcopy(self.data.workload_on_source_matrix[request])
        # fill x
        self.fill_x(cfj_r, lamb_r)

        # fill y
        self.fill_y(cfj_r)

        # fill z
        self.fill_z(cfj_r)

    def get_memory(self, layer, node):  # consider the presence of instances
        total_memory = 0
        _, _, _, mf, _, _, _ = self.basic_fill_data()
        for f in layer:
            if not self.cfj[f, node]:
                total_memory = total_memory + mf[f]
        return total_memory

    def get_total_memory(self, layer):  # do not consider the presence of instances
        total_memory = 0
        _, _, _, mf, _, _, _ = self.basic_fill_data()
        for f in layer:
            total_memory = total_memory + mf[f]
        return total_memory

    def get_cpu_demand(self, layer, request, i):
        total_cpu = 0

        _, _, _, _, _, ufj, _ = self.basic_fill_data()
        for f in layer:
            total_cpu = total_cpu + self.w[request][f][i]*ufj[f, i]
        return total_cpu

    def resource_usage(self):
        total_nodes = 0
        memory = 0
        cpus = 0
        memories=[]
        cores=[]
        nodes = len(self.cfj[0])
        functions = len(self.cfj)
        for i in range(nodes):
            memories.append(self.data.node_memories[i]-self.node_memory_available[i])
            cores.append(self.data.node_cores[i] - self.node_cpu_available[i])
            for f in range(functions):
                if self.cfj[f, i] == 1:
                    total_nodes = total_nodes+1
                    break
        for m in memories:
            memory = memory+m
        for cpu in cores:
            cpus = cpus + cpu
        return total_nodes, memory, cpus


    def heuristic_placement(self, request, i, parallel_scheduler):
        start_time = time.time()
        self.fill_w(request)
        self.place_app(request, i, parallel_scheduler)
        end_time = time.time()
        self.fill_xyz(request)
        # Calculate the elapsed time
        decision_time = (end_time - start_time)*1000  # milliseconds
        return self.x[request], self.y[request], self.z[request], self.w[request],  self.list_r_cfj, self.cfj, \
            decision_time

    def get_max_coldstart(self, new_instances):
        max_coldstart = 0
        coldstart = self.data.function_cold_starts
        for f in new_instances:
            if coldstart[f] > max_coldstart:
                max_coldstart = coldstart[f]
        return max_coldstart

    def get_coldstart(self):
        return self.coldstart

    def exist(self, candidate, first_layer):
        return self.cfj[first_layer[0], candidate]
