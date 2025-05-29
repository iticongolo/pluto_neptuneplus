import math

import numpy as np

# from core.utils.util import *


class Cluster:
    def __init__(self, id, cores=0, memory=0, cores_available=0, memory_available=0, cluster_servers=None,
                 all_servers=None, centroid=(float('inf'), float('inf')), functions=None, status=1):
        self.id = id
        self.status = status
        self.original_cores = cores  # exclude cores allocated from another clusters by sharing cluster_servers
        self.original_memory = memory  # exclude memory allocated to another clusters by sharing cluster_servers
        self.cores_available = cores_available
        self.predicted_cores_available = cores_available
        self.memory_available = memory_available
        self.minimum_capacity_cores = 0  # recommended 20% of the cluster capacity
        self.minimum_capacity_memory = 0  # recommended 20% of the cluster capacity
        self.servers = cluster_servers if cluster_servers is not None else []  # a list
        self.all_servers = all_servers if all_servers is not None else []
        self.total_workload_per_f = []
        self.cores_needed_per_f = []

        self.centroid = centroid  # (x,y)
        # e.g.: [n0|2ms|, n1|1ms|, n3|10ms|, ...] i.e. delay from centroid to server 0 is 2ms
        # we initialize the delays by infinity
        self.centroid_servers_network_delay = [float('inf') for _ in range(len(self.all_servers))]
        self.received_servers = []

        # include all the cluster_servers here and put zero by default
        self.received_servers_allocated_cores = [0 for _ in range(len(self.all_servers))]
        self.received_servers_allocated_memory = [0 for _ in range(len(self.all_servers))]
        self.received_servers_available_cores = [0 for _ in range(len(self.all_servers))]
        self.received_servers_available_memory = [0 for _ in range(len(self.all_servers))]

        self.capacity_cores = cores  # include cores received from shared cluster_servers
        self.capacity_memory = memory  # include memory received from shared cluster_servers
        self.additional_cores_needed = 0
        self.functions = functions if functions is not None else []
        self.network_delays = []
        self.cpu_allocation = {}  # to be used by neptune

    def set_centroid(self, centroid):
        self.centroid = centroid

    def set_status(self, status):
        self.status = status

    def set_cores_available(self, cores):
        self.cores_available = cores

    def set_functions(self, functions):
        self.functions = functions

    def set_memory_available(self, memory):
        self.memory_available = memory

    def set_received_server(self, server):
        self.received_servers.append(server)

    def has_received_shared_servers(self):
        return len(self.received_servers) > 0

    def update_available_cores(self, new_cores):
        self.cores_available = self.cores_available + new_cores
        self.predicted_cores_available = self.predicted_cores_available + new_cores
        self.update_status_()

    def update_capacity_cores(self, new_cores):
        self.capacity_cores = self.capacity_cores + new_cores

    def update_original_resources(self, new_cores, new_memory):
        self.original_cores = self.original_cores+new_cores
        self.original_memory = self.original_memory + new_memory

    def update_available_memory(self, new_memory):
        self.memory_available = self.memory_available + new_memory

    def update_capacity_memory(self, new_memory):
        self.capacity_memory = self.capacity_memory + new_memory

    def update_status(self, required_cores):
        # print(f'required_cores AAAA={required_cores}')
        # print(f'capacity-cores AAAA={self.capacity_cores}')
        remained_cores = self.capacity_cores-required_cores
        if remained_cores < 0:
            self.status = -1  # overloaded
        else:
            if remained_cores <= 0.2*self.capacity_cores:
                self.status = 0  # fine
            else:
                self.status = 1  # underloaded

    def update_status_(self):
        if self.cores_available <= 0.2*self.capacity_cores:
            self.status = 0  # fine
        else:
            self.status = 1  # underloaded

    def initialize_cluster(self):
        for server in self.servers:
            self.update_capacity_cores(server.cores)
            self.update_available_cores(server.cores)
            self.update_capacity_memory(server.memory)
            self.update_available_memory(server.memory)
            self.update_centroid()

    # Add a new shared server or only add the new cores allocation corresponding to the existing shared server
    def update_received_servers(self, new_server, new_cores, new_memory):
        server_not_exists = 1
        for server in self.received_servers:
            if server.id == new_server.id:
                server_not_exists = 0
                break
        if server_not_exists:
            self.received_servers.append(new_server)
        self.add_server(new_server, new_cores, new_server.memory_available)
        # self.update_capacity_cores(new_cores)
        self.received_servers_allocated_cores[new_server.id] = self.received_servers_allocated_cores[new_server.id] + new_cores
        self.received_servers_allocated_memory[new_server.id] = self.received_servers_allocated_memory[
                                                                   new_server.id] + new_memory
        new_server.update_shared_resources(self.id, new_cores=new_cores, new_memory=new_memory,
                                           new_cores_available=new_cores, new_memory_available=new_memory)
        # update the

    def add_server(self, new_server, new_cores, new_memory, cluster_generation=0):
        server_not_exists = 1
        for server in self.servers:
            if server.id == new_server.id:
                server_not_exists = 0
                break
        if server_not_exists:
            self.servers.append(new_server)
        if cluster_generation:
            # print(f'+########## new_cores={new_cores}, new_memory={new_memory}')
            self.update_resources(new_cores=new_cores, new_memory=new_memory)
        else:
            self.update_resources_capacity(new_cores=new_cores, new_memory=new_memory)

    # we only remove servers shared with this cluster and never the servers from initial cluster
    # when removing a server, the capacity of the cluster changes
    def remove(self, server):
        # print(f'Cluster [{self.id}]-servers BF={self.servers}')
        pos = self.get_server_position(self.servers, server)
        if pos >= 0:
            self.received_server_reset(server)
            self.servers.pop(pos)

        # print(f'Cluster [{self.id}]-servers AF={self.servers}')

    def update_centroid(self):
        sum_x = 0
        sum_y = 0
        for server in self.servers:
            sum_x = sum_x+server.location[0]
            sum_y = sum_y + server.location[1]
        self.centroid = (sum_x/len(self.servers), sum_y / len(self.servers))

    def update_centroid_servers_network_delay(self, new_servers):
        for server in new_servers:
            eu_distance = round(
                math.sqrt((self.centroid[0] - server.location[0]) ** 2 + (self.centroid[1] - server.location[1]) ** 2), 2)
            self.centroid_servers_network_delay[server.id] = eu_distance

    def list_servers(self):
        list_servers = []
        for server in self.servers:
            list_servers.append(f'server{server.id}')
        return list_servers
    # # update all the resources that will be avai
    # lable after serving the cluster
    # # workload on cluster_servers received from other clusters. Remove the received cluster_servers that will not be used
    # def update_cores_received_servers(self, remain_cores_needed):  # TODO check later, seems to be inconcistente.
    #     # structure of cores_received_servers [[server_id,cores allocated to this cluster],...]
    #     # e.g.: [[0,200],[11,500],[4,100],...]
    #     cores_needed = remain_cores_needed
    #     i = 0
    #     while i < len(self.received_servers) and cores_needed > 0:
    #         diff = cores_needed - self.received_servers[i].cores_available
    #         if diff > 0:  # the cores needed are more than available on the current server
    #             self.received_servers_allocated_cores.append([self.received_servers[i].id, self.received_servers[i].cores_available])
    #             cores_needed = cores_needed - self.received_servers[i].cores_available
    #             self.received_servers[i].set_cores_available(0)  # all the remained cores were used
    #         else:
    #             self.received_servers_allocated_cores.append(
    #                 [self.received_servers[i].id, self.received_servers[i].cores_available])
    #             self.received_servers[i].set_cores_available(
    #                 self.received_servers[i].cores_available - cores_needed)  # use all the remained cores
    #             cores_needed = 0
    #             self.received_servers = self.received_servers[:i+1]
    #             break
    #         i = i+1
    #     self.additional_cores_needed = cores_needed

    # def remove_received_server(self, server_id):
    #     for i in range(self.received_servers.len):
    #         if self.received_servers[i].id == server_id:
    #             self.received_servers.pop(i)
    #             break

    # Check whether the cluster is predicted to be overloaded, fine or underloaded NOTE: DONE

    def update_resources_capacity(self, new_cores=0, new_memory=0):
        self.update_capacity_cores(new_cores)
        self.update_capacity_memory(new_memory)

    def update_resources(self, new_cores=0, new_memory=0):
        self.update_capacity_cores(new_cores)
        self.update_capacity_memory(new_memory)
        self.update_available_cores(new_cores)
        self.update_available_memory(new_memory)

    def update(self, cores_requested=0):
        if cores_requested > self.cores_available:
            self.cores_available = 0
        else:
            self.cores_available = self.cores_available-cores_requested

    def update_predicted_cores_available(self, requested_cores):
        self.predicted_cores_available = self.capacity_cores-requested_cores

    def get_server_predicted_available_resources(self, server):
        cores_available, _ = self.get_server_available_resources(server)
        return min(cores_available, self.predicted_cores_available)

    @staticmethod
    def get_server_position(servers, server):
        for i in range(len(servers)):
            if server.id == servers[i].id:
                return i
        return -1

    def get_network_delay(self, topology_network_delays):
        self.compute_network_delays(topology_network_delays)
        return self.network_delays

    def compute_network_delays(self, topology_network_delays):
        if len(self.servers) == 0:
            self.network_delays = []
            return
        self.network_delays = np.zeros((len(self.servers), len(self.servers)))
        i = 0
        for server_i in self.servers:
            j = 0
            for server_j in self.servers:
                self.network_delays[i][j] = topology_network_delays[server_i.id][server_j.id]
                j = j + 1
            i = i + 1

    def is_received_server(self, server):
        for s in self.received_servers:
            if s.id == server.id:
                return True
        return False

    def received_server_reset(self, server):
        allocated_cores = self.received_servers_allocated_cores[server.id]
        allocated_memory = self.received_servers_allocated_memory[server.id]
        self.update_resources(new_cores=-allocated_cores, new_memory=-allocated_memory)
        server.release_server(self.id)
        self.received_servers_allocated_cores[server.id] = 0
        self.received_servers_allocated_memory[server.id] = 0
        self.received_servers_available_cores[server.id] = 0
        self.received_servers_available_memory[server.id] = 0

    def get_server_available_resources(self, server):
        if server.shared_cores[self.id] > 0:  # this server is shared by other clusters
            cores_available = server.shared_available_cores[self.id]
            memory_available = server.shared_available_memory[self.id]
        else:
            cores_available = server.cores_available
            memory_available = server.memory_available
        return cores_available, memory_available

    def get_server_allocated_resources(self, server):
        if server.shared_cores[self.id] > 0:  # this server is shared by other clusters
            cores = server.shared_cores[self.id]
            memory = server.shared_memory[self.id]
        else:
            cores = server.cores
            memory = server.memory
        return cores, memory

    def update_server_available_resources(self, server, new_cores=0, new_memory=0):
        if server.shared_cores[self.id] > 0:  # this server is shared by other clusters
            server.shared_available_cores[self.id] = server.shared_available_cores[self.id] + new_cores
            server.shared_available_memory[self.id] = server.shared_available_memory[self.id] + new_memory
        else:
            server.cores_available = server.cores_available + new_cores
            server.memory_available = server.memory_available + new_memory
        self.update_resources(new_cores=new_cores, new_memory=new_memory)

    def reset(self):
        self.initialize_cluster()

