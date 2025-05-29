import copy
import math

from cluster import Cluster
from math import radians, sin, cos, sqrt, atan2


class Topology:

    def __init__(self, server_locations=(0, 0), status=1, servers=None, network_delays=None, initial_clusters=None):
        self.status = status
        self.servers = servers if servers is not None else []  # a list
        self.server_locations = server_locations
        # e.g.: [[0,3,1],[3,0,5],[1,5,0]] delays of servers n0, n1, n2
        self.network_delays = network_delays if network_delays is not None else []

        # e.g.: [n0|c1|, n1|c3|, n2|c5|, n3|c1|...] server is the index
        self.static_server_cluster = [None for _ in range(len(servers))]
        self.initial_clusters = initial_clusters if initial_clusters is not None else []  # a list
        self.current_clusters = copy.deepcopy(self.initial_clusters)

    def generate_clusters(self, delay_threshold=10):
        clusters = []
        servers = copy.deepcopy(self.servers)
        cluster_id = 0
        while len(servers) > 0:
            cluster = Cluster(cluster_id, all_servers=self.servers)
            i = 0
            j = 1
            cluster.add_server(servers[0], servers[0].cores, servers[0].memory, cluster_generation=1)
            servers.pop(0)
            while i < j:
                k = 0
                for near_server in servers:
                    if self.network_delays[cluster.servers[i].id][near_server.id] <= delay_threshold:
                        cluster.add_server(near_server, near_server.cores, near_server.memory, cluster_generation=1)
                        servers.pop(k)
                        j = j + 1
                    k = k+1
                i = i+1
            cluster.initialize_cluster()
            clusters.append(cluster)
            cluster_id = cluster_id+1
        self.generate_static_server_cluster(clusters)
        return clusters

    def generate_balanced_clusters(self, delay_threshold=10, max_servers_threshold=10):
        clusters = []
        servers = copy.deepcopy(self.servers)
        cluster_id = 0
        while len(servers) > 0:
            cluster = Cluster(cluster_id, all_servers=self.servers)
            i = 0
            j = 1
            cluster.add_server(servers[0], servers[0].cores, servers[0].memory, cluster_generation=1)
            cluster.update_original_resources(servers[0].cores, servers[0].memory)
            servers.pop(0)
            while i < j:
                k = 0
                for near_server in servers:
                    if len(cluster.servers) >= max_servers_threshold:
                        break
                    if self.network_delays[cluster.servers[i].id][near_server.id] <= delay_threshold:
                        cluster.add_server(near_server, near_server.cores, near_server.memory, cluster_generation=1)
                        cluster.update_original_resources(near_server.cores, near_server.memory)
                        servers.pop(k)
                        j = j + 1
                    k = k+1
                i = i+1

            cluster.minimum_capacity_cores = 0.2 * cluster.original_cores
            cluster.minimum_capacity_memory = 0.2 * cluster.original_memory
            # cluster.initialize_cluster()
            cluster.update_centroid()
            clusters.append(cluster)
            cluster_id = cluster_id+1
        self.generate_static_server_cluster(clusters)

        first_list = [0 for _ in clusters]
        for c in clusters:
            for server in c.servers:
                server.shared_cores = copy.deepcopy(first_list)
                server.shared_memory = copy.deepcopy(first_list)
                server.shared_available_cores = copy.deepcopy(first_list)
                server.shared_available_memory = copy.deepcopy(first_list)

        return clusters

    # def re_balancing_clusters(self, ):

    def set_status(self, status):
        self.status = status

    def generate_static_server_cluster(self, clusters):
        for c in copy.deepcopy(clusters):
            for server in c.servers:
                self.static_server_cluster[server.id] = c

    def set_network_delays(self, delays):
        self.network_delays = delays

    def generate_network_delays(self, hop_delays=None, negligible_delays=0):
        speed_of_light = 300000 # 300.000 km/s
        latency_of_two_hops_between_nodes = 0.001 if hop_delays is None else 0.0  # 1s from server 1 to switch 1, from switch 1 to switch 2
        # and from switch 2 to server 2
        self. network_delays = [[0 for _ in range(len(self.servers))] for _ in range(len(self.servers))]
        if not negligible_delays:
            for server1 in self.servers:
                for server2 in self.servers:
                    hop_d = 0.001 if hop_delays is None else hop_delays[server1.id][server2.id]  # use 1 for constant hop delay
                    if server1.id == server2.id:
                        self.network_delays[server1.id][server2.id] = 0
                    else:
                        # euclidian_distance = round(math.sqrt((server1.location[0] - server2.location[0]) ** 2 +
                        #                                      (server1.location[1] - server2.location[1]) ** 2), 2)
                        haversine_distance = get_distance(server1.location, server2.location)
                        delay = 2.1*haversine_distance/speed_of_light + latency_of_two_hops_between_nodes+hop_d
                        print(f'2.1*haversine_distance/speed_of_light + latency_of_two_hops_between_nodes+hop_d=2.1*{haversine_distance / speed_of_light} + {latency_of_two_hops_between_nodes}+{hop_d}={delay}')
                        self.network_delays[server1.id][server2.id] = round(delay, 8)

    def set_functions(self, functions):
        for c in self.current_clusters:
            c.set_functions(functions)
        for c in self.initial_clusters:
            c.set_functions(functions)


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
    unit = 1  # for seconds

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

