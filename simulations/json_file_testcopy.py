class JsonfileTestCopy:

    qty_f = 2
    cpu_allocation = {}
    position = 0

    functions = ["ns/f0", "ns/f1"]

    input = {
        "with_db": False,
        "solver": {
            "type": "NeptuneMinDelay",
            "args": {"alpha": 0.0, "verbose": True}
        },
        "cpu_coeff": 1,
        "community": "community-test",
        "namespace": "namespace-test",
        "node_names": [],
        "node_delay_matrix": [],

        "workload_on_source_matrix": "\"[0]\"",

        "node_memories": [],

        "execution_time": [
            50, 50
        ],
        "node_cores": [],

        "gpu_node_names": [
        ],
        "gpu_node_memories": [
        ],
        "function_names": functions,
        "function_memories": [
            200, 200
        ],
        "function_cold_starts": [
            500, 500
        ],
        "function_max_delays": [
            1000, 1000
        ],
        "gpu_function_names": [
        ],
        "gpu_function_memories": [
        ],
        "actual_cpu_allocations": cpu_allocation,
        "app": "test",
        "actual_gpu_allocations": {
        },
        "nodes": [
            {"name": "f0", "users": 0, "nrt": 10},
            {"name": "f1", "users": 2, "nrt": 10},
        ],
        "edges": [
            {"source": "f0", "target": "f1", "sync": 1, "times": 1},
        ],
        "m": [[0, 0], [0, 0]],
    }
