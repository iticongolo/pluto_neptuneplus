from core import nodes, network


class JsonfileHotel:
    def __init__(self, id, cores=0, memory=0, cores_available=0, memory_available=0, cluster_servers=None,
                 all_servers=None, centroid=(float('inf'), float('inf')), functions=None, status=1):
        self.id = id
        self.status = status
    qty_f = 4
    cpu_allocation = {}
    position = 0

    functions = ["ns/f0(Sch)", "ns/f1(Geo)", "ns/f2(Prof)", "ns/f3(Rat)"]

    input = {
        "with_db": False,
        "solver": {
            "type": "NeptuneMinDelay",
            "args": {"alpha": 0.0, "verbose": True}
        },
        "cpu_coeff": 1,
        "community": "community-test",
        "namespace": "namespace-test",
        "node_names": nodes[position],
        "node_delay_matrix": network[position]['node_delay_matrix'],

        "workload_on_source_matrix": "\"[0]\"",

        "node_memories": network[position]["node_memories"],

        "execution_time": [
            78, 13, 33, 16
        ],
        "node_cores": network[position]["node_cores"],

        "gpu_node_names": [
        ],
        "gpu_node_memories": [
        ],
        "function_names": functions,
        "function_memories": [
            512, 512, 512, 512
        ],
        "function_cold_starts": [
            702, 177, 297, 144
        ],
        "function_max_delays": [
            5, 5, 5, 5
        ],
        "gpu_function_names": [
        ],
        "gpu_function_memories": [
        ],
        "actual_cpu_allocations": cpu_allocation,
        "app": "hotel",
        "actual_gpu_allocations": {
        },
        "nodes": [
            {"name": "f0(Sch)", "users": 0, "nrt": 10},
            {"name": "f1(Geo)", "users": 2, "nrt": 15},
            {"name": "f2(Prof)", "users": 2, "nrt": 10},
            {"name": "f3(Rat)", "users": 2, "nrt": 10},
        ],
        "edges": [
            {"source": "f0(Sch)", "target": "f1(Geo)", "sync": 1, "times": 1},
            {"source": "f0(Sch)", "target": "f2(Prof)", "sync": 2, "times": 1},
            {"source": "f0(Sch)", "target": "f3(Rat)", "sync": 3, "times": 1},
        ],
        "m": [[0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        "parallel_scheduler": [[0], [1], [2], [3]]
    }
