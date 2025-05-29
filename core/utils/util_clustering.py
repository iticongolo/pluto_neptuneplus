import copy

from core.utils.util import get_severs_list
from dynamic_clustering import DynamicClustering
from network_topology import Topology


def generate_dynamic_clustering(server_locations, old_servers=None, cores_range=(16000, 32000), hop_delays=None, delay_threshold=100,
                                max_servers_threshold=10000, functions=None, cores=None, step=1000):

    servers = get_severs_list(old_servers=old_servers, total_servers=len(server_locations), cores_range=cores_range, cores=cores, step=step,
                              location=server_locations)

    topology = Topology(servers=servers)
    topology.server_locations = server_locations
    topology.generate_network_delays(hop_delays=hop_delays)

    # for node in topology.nodes:
    #     print(f'Node{node.id}-Location={node.location}')
    clusters = topology.generate_balanced_clusters(delay_threshold=delay_threshold, max_servers_threshold=max_servers_threshold)

    for c in topology.current_clusters:
        c.compute_network_delays(topology.network_delays)

    for c in clusters:
        c.functions = functions

    topology.initial_clusters = copy.deepcopy(clusters)
    topology.current_clusters = clusters

    dynamic_clustering = DynamicClustering(topology, functions)
    return dynamic_clustering
