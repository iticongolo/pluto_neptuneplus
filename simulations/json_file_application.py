# from core import nodes, network


class JsonFileApplication:
    def __init__(self, functions_list, function_memory_needed, function_cold_starts, cluster,
                 functions_execution_time, function_max_delays, functions_network, functions_edges,
                 call_times, heu_xu_map=0, qty_f=0, app_name=None):
        self.qty_f = qty_f
        self.cpu_allocation = {}
        self.functions = functions_list
        self.app_name = app_name
        self. input = {
        "with_db": False,
        "solver": {
            "type": "NeptuneMinDelay",
            "args": {"alpha": 0.0, "verbose": True}
        },
        "cpu_coeff": 1,
        "community": "community-test",
        "namespace": "namespace-test",
        "node_names": [server.name for server in cluster.servers],
        "node_delay_matrix": cluster.network_delays,

        "workload_on_source_matrix": [],

        "node_memories": cluster.get_servers_memory_available(),

        "execution_time": functions_execution_time,
        "node_cores": cluster.get_servers_cores_available(),

        "gpu_node_names": [
        ],
        "gpu_node_memories": [
        ],
        "function_names": self.functions,
        "function_memories": function_memory_needed,
        "function_cold_starts": function_cold_starts,
        "function_max_delays": function_max_delays,
        "gpu_function_names": [
        ],
        "gpu_function_memories": [
        ],
        "actual_cpu_allocations": self.cpu_allocation,
        "app": "hotel",
        "actual_gpu_allocations": {
        },
        "nodes": functions_network,
        "edges": functions_edges,
        "m": call_times,
        "parallel_scheduler": heu_xu_map
        }
