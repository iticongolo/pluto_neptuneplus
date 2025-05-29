class ApplicationsVaryFunctionsInputs:

    cpu_allocation = {}
    position = 0
    functions = [["ns/f0", "ns/f1", "ns/f2", "ns/f3", "ns/f4"],

                 ["ns/f0", "ns/f1", "ns/f2", "ns/f3", "ns/f4", "ns/f5", "ns/f6", "ns/f7", "ns/f8", "ns/f9"],

                 ["ns/f0", "ns/f1", "ns/f2", "ns/f3", "ns/f4", "ns/f5", "ns/f6", "ns/f7", "ns/f8",
                  "ns/f9", "ns/f10", "ns/f11", "ns/f12", "ns/f13", "ns/f14"],

                 ["ns/f0", "ns/f1", "ns/f2", "ns/f3", "ns/f4", "ns/f5", "ns/f6", "ns/f7", "ns/f8",
                  "ns/f9", "ns/f10", "ns/f11", "ns/f12", "ns/f13", "ns/f14", "ns/f15", "ns/f16", "ns/f17", "ns/f18",
                  "ns/f19"],

                 ["ns/f0", "ns/f1", "ns/f2", "ns/f3", "ns/f4", "ns/f5", "ns/f6", "ns/f7", "ns/f8",
                  "ns/f9", "ns/f10", "ns/f11", "ns/f12", "ns/f13", "ns/f14", "ns/f15", "ns/f16", "ns/f17", "ns/f18",
                  "ns/f19", "ns/f20", "ns/f21", "ns/f22", "ns/f23", "ns/f24"],

                 ["ns/f0", "ns/f1", "ns/f2", "ns/f3", "ns/f4", "ns/f5", "ns/f6", "ns/f7", "ns/f8",
                  "ns/f9", "ns/f10", "ns/f11", "ns/f12", "ns/f13", "ns/f14", "ns/f15", "ns/f16", "ns/f17", "ns/f18",
                  "ns/f19", "ns/f20", "ns/f21", "ns/f22", "ns/f23", "ns/f24", "ns/f25", "ns/f26", "ns/f27", "ns/f28",
                  "ns/f29"],
                 ]

    inputs = [{
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
            50, 50, 50, 50, 50
        ],
        "node_cores": [],

        "gpu_node_names": [
        ],
        "gpu_node_memories": [
        ],
        "function_names": functions[0],
        "function_memories": [
            200, 200, 200, 200, 200
        ],
        "function_cold_starts": [
            500, 500, 500, 500, 500
        ],
        "function_max_delays": [
            100, 100, 100, 100, 100
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
            {"name": "f0", "users": 2, "nrt": 10},
            {"name": "f1", "users": 2, "nrt": 10},
            {"name": "f2", "users": 2, "nrt": 10},
            {"name": "f3", "users": 2, "nrt": 10},
            {"name": "f4", "users": 2, "nrt": 10},

        ],
        "edges": [
            {"source": "f0", "target": "f1", "sync": 1, "times": 1},
            {"source": "f0", "target": "f2", "sync": 2, "times": 1},
            {"source": "f0", "target": "f3", "sync": 2, "times": 1},
            {"source": "f1", "target": "f4", "sync": 3, "times": 1},
        ],
        "m": [
            [0, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]

        ],
        "parallel_scheduler": [
            [0], [1], [2], [3], [4]
        ]
    },

        {
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
            50, 50, 50, 50, 50, 50, 50, 50, 50, 50
        ],
        "node_cores": [],

        "gpu_node_names": [
        ],
        "gpu_node_memories": [
        ],
        "function_names": functions[1],
        "function_memories": [
            200, 200, 200, 200, 200, 200, 200, 200, 200, 200
        ],
        "function_cold_starts": [
            500, 500, 500, 500, 500, 500, 500, 500, 500, 500
        ],
        "function_max_delays": [
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100
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
            {"name": "f0", "users": 2, "nrt": 10},
            {"name": "f1", "users": 2, "nrt": 10},
            {"name": "f2", "users": 2, "nrt": 10},
            {"name": "f3", "users": 2, "nrt": 10},
            {"name": "f4", "users": 2, "nrt": 10},

            {"name": "f5", "users": 2, "nrt": 10},
            {"name": "f6", "users": 2, "nrt": 10},
            {"name": "f7", "users": 2, "nrt": 10},
            {"name": "f8", "users": 2, "nrt": 10},
            {"name": "f9", "users": 2, "nrt": 10}

        ],
        "edges": [
            {"source": "f0", "target": "f1", "sync": 1, "times": 1},
            {"source": "f0", "target": "f2", "sync": 2, "times": 1},
            {"source": "f0", "target": "f3", "sync": 2, "times": 1},
            {"source": "f1", "target": "f4", "sync": 3, "times": 1},

            {"source": "f1", "target": "f5", "sync": 4, "times": 1},
            {"source": "f2", "target": "f6", "sync": 5, "times": 1},
            {"source": "f2", "target": "f7", "sync": 6, "times": 1},
            {"source": "f3", "target": "f8", "sync": 7, "times": 1},
            {"source": "f3", "target": "f9", "sync": 7, "times": 1}
        ],
        "m": [
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        ],
        "parallel_scheduler": [
            [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]
        ]
    },

        {
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
            50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50
        ],
        "node_cores": [],

        "gpu_node_names": [
        ],
        "gpu_node_memories": [
        ],
        "function_names": functions[2],
        "function_memories": [
            200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200
        ],
        "function_cold_starts": [
            500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500
        ],
        "function_max_delays": [
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100
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
            {"name": "f0", "users": 2, "nrt": 10},
            {"name": "f1", "users": 2, "nrt": 10},
            {"name": "f2", "users": 2, "nrt": 10},
            {"name": "f3", "users": 2, "nrt": 10},
            {"name": "f4", "users": 2, "nrt": 10},
            {"name": "f5", "users": 2, "nrt": 10},
            {"name": "f6", "users": 2, "nrt": 10},
            {"name": "f7", "users": 2, "nrt": 10},
            {"name": "f8", "users": 2, "nrt": 10},
            {"name": "f9", "users": 2, "nrt": 10},
            {"name": "f10", "users": 2, "nrt": 10},
            {"name": "f11", "users": 2, "nrt": 10},
            {"name": "f12", "users": 2, "nrt": 10},
            {"name": "f13", "users": 2, "nrt": 10},
            {"name": "f14", "users": 2, "nrt": 10},

        ],
        "edges": [
            {"source": "f0", "target": "f1", "sync": 1, "times": 1},
            {"source": "f0", "target": "f2", "sync": 2, "times": 1},
            {"source": "f0", "target": "f3", "sync": 2, "times": 1},
            {"source": "f1", "target": "f4", "sync": 3, "times": 1},

            {"source": "f1", "target": "f5", "sync": 4, "times": 1},
            {"source": "f2", "target": "f6", "sync": 5, "times": 1},
            {"source": "f2", "target": "f7", "sync": 6, "times": 1},
            {"source": "f3", "target": "f8", "sync": 7, "times": 1},
            {"source": "f3", "target": "f9", "sync": 7, "times": 1},

            {"source": "f4", "target": "f10", "sync": 8, "times": 1},
            {"source": "f4", "target": "f11", "sync": 9, "times": 1},
            {"source": "f4", "target": "f12", "sync": 10, "times": 1},
            {"source": "f8", "target": "f13", "sync": 11, "times": 1},
            {"source": "f8", "target": "f14", "sync": 12, "times": 1}
        ],
        "m": [

            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],

            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        "parallel_scheduler": [
            [0], [1], [2], [3], [4], [5], [6], [7], [8], [9],  [10], [11], [12], [13], [14]
        ]
    },

        {
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
                50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50
            ],
            "node_cores": [],

            "gpu_node_names": [
            ],
            "gpu_node_memories": [
            ],
            "function_names": functions[3],
            "function_memories": [
                200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200
            ],
            "function_cold_starts": [
                500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500
            ],
            "function_max_delays": [
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100
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
                {"name": "f0", "users": 2, "nrt": 10},
                {"name": "f1", "users": 2, "nrt": 10},
                {"name": "f2", "users": 2, "nrt": 10},
                {"name": "f3", "users": 2, "nrt": 10},
                {"name": "f4", "users": 2, "nrt": 10},
                {"name": "f5", "users": 2, "nrt": 10},
                {"name": "f6", "users": 2, "nrt": 10},
                {"name": "f7", "users": 2, "nrt": 10},
                {"name": "f8", "users": 2, "nrt": 10},
                {"name": "f9", "users": 2, "nrt": 10},
                {"name": "f10", "users": 2, "nrt": 10},
                {"name": "f11", "users": 2, "nrt": 10},
                {"name": "f12", "users": 2, "nrt": 10},
                {"name": "f13", "users": 2, "nrt": 10},
                {"name": "f14", "users": 2, "nrt": 10},
                {"name": "f15", "users": 2, "nrt": 10},
                {"name": "f16", "users": 2, "nrt": 10},
                {"name": "f17", "users": 2, "nrt": 10},
                {"name": "f18", "users": 2, "nrt": 10},
                {"name": "f19", "users": 2, "nrt": 10},

            ],
            "edges": [
                {"source": "f0", "target": "f1", "sync": 1, "times": 1},
                {"source": "f0", "target": "f2", "sync": 2, "times": 1},
                {"source": "f0", "target": "f3", "sync": 2, "times": 1},
                {"source": "f1", "target": "f4", "sync": 3, "times": 1},

                {"source": "f1", "target": "f5", "sync": 4, "times": 1},
                {"source": "f2", "target": "f6", "sync": 5, "times": 1},
                {"source": "f2", "target": "f7", "sync": 6, "times": 1},
                {"source": "f3", "target": "f8", "sync": 7, "times": 1},
                {"source": "f3", "target": "f9", "sync": 7, "times": 1},

                {"source": "f4", "target": "f10", "sync": 8, "times": 1},
                {"source": "f4", "target": "f11", "sync": 9, "times": 1},
                {"source": "f4", "target": "f12", "sync": 10, "times": 1},
                {"source": "f6", "target": "f15", "sync": 13, "times": 1},
                {"source": "f6", "target": "f16", "sync": 14, "times": 1},
                {"source": "f6", "target": "f17", "sync": 14, "times": 1},
                {"source": "f8", "target": "f13", "sync": 11, "times": 1},
                {"source": "f8", "target": "f14", "sync": 12, "times": 1},


                {"source": "f9", "target": "f18", "sync": 15, "times": 1},
                {"source": "f9", "target": "f19", "sync": 16, "times": 1}
            ],
            "m": [

                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            "parallel_scheduler": [
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11],
                [12], [13], [14], [15], [16], [17], [18], [19]
            ]
        },

        {
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
                50, 50, 50, 50, 50,
                50, 50, 50, 50, 50,
                50, 50, 50, 50, 50,
                50, 50, 50, 50, 50,
                50, 50, 50, 50, 50
            ],
            "node_cores": [],

            "gpu_node_names": [
            ],
            "gpu_node_memories": [
            ],
            "function_names": functions[4],
            "function_memories": [
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200
            ],
            "function_cold_starts": [
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500
            ],
            "function_max_delays": [
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100
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
                {"name": "f0", "users": 2, "nrt": 10},
                {"name": "f1", "users": 2, "nrt": 10},
                {"name": "f2", "users": 2, "nrt": 10},
                {"name": "f3", "users": 2, "nrt": 10},
                {"name": "f4", "users": 2, "nrt": 10},
                {"name": "f5", "users": 2, "nrt": 10},
                {"name": "f6", "users": 2, "nrt": 10},
                {"name": "f7", "users": 2, "nrt": 10},
                {"name": "f8", "users": 2, "nrt": 10},
                {"name": "f9", "users": 2, "nrt": 10},
                {"name": "f10", "users": 2, "nrt": 10},
                {"name": "f11", "users": 2, "nrt": 10},
                {"name": "f12", "users": 2, "nrt": 10},
                {"name": "f13", "users": 2, "nrt": 10},
                {"name": "f14", "users": 2, "nrt": 10},
                {"name": "f15", "users": 2, "nrt": 10},
                {"name": "f16", "users": 2, "nrt": 10},
                {"name": "f17", "users": 2, "nrt": 10},
                {"name": "f18", "users": 2, "nrt": 10},
                {"name": "f19", "users": 2, "nrt": 10},
                {"name": "f20", "users": 2, "nrt": 10},
                {"name": "f21", "users": 2, "nrt": 10},
                {"name": "f22", "users": 2, "nrt": 10},
                {"name": "f23", "users": 2, "nrt": 10},
                {"name": "f24", "users": 2, "nrt": 10},

            ],
            "edges": [
                {"source": "f0", "target": "f1", "sync": 1, "times": 1},
                {"source": "f0", "target": "f2", "sync": 2, "times": 1},
                {"source": "f0", "target": "f3", "sync": 2, "times": 1},
                {"source": "f1", "target": "f4", "sync": 3, "times": 1},

                {"source": "f1", "target": "f5", "sync": 4, "times": 1},
                {"source": "f2", "target": "f6", "sync": 5, "times": 1},
                {"source": "f2", "target": "f7", "sync": 6, "times": 1},
                {"source": "f3", "target": "f8", "sync": 7, "times": 1},
                {"source": "f3", "target": "f9", "sync": 7, "times": 1},

                {"source": "f4", "target": "f10", "sync": 8, "times": 1},
                {"source": "f4", "target": "f11", "sync": 9, "times": 1},
                {"source": "f4", "target": "f12", "sync": 10, "times": 1},
                {"source": "f6", "target": "f15", "sync": 13, "times": 1},
                {"source": "f6", "target": "f16", "sync": 14, "times": 1},
                {"source": "f6", "target": "f17", "sync": 14, "times": 1},
                {"source": "f8", "target": "f13", "sync": 11, "times": 1},
                {"source": "f8", "target": "f14", "sync": 12, "times": 1},

                {"source": "f9", "target": "f18", "sync": 15, "times": 1},
                {"source": "f9", "target": "f19", "sync": 16, "times": 1},

                {"source": "f11", "target": "f20", "sync": 17, "times": 1},
                {"source": "f11", "target": "f21", "sync": 17, "times": 1},
                {"source": "f16", "target": "f22", "sync": 18, "times": 1},
                {"source": "f16", "target": "f23", "sync": 18, "times": 1},
                {"source": "f16", "target": "f24", "sync": 19, "times": 1}

            ],
            "m": [

                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            ],
            "parallel_scheduler": [
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11],
                [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24]
            ]
        },

        {
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
                50, 50, 50, 50, 50,
                50, 50, 50, 50, 50,
                50, 50, 50, 50, 50,
                50, 50, 50, 50, 50,
                50, 50, 50, 50, 50,
                50, 50, 50, 50, 50
            ],
            "node_cores": [],

            "gpu_node_names": [
            ],
            "gpu_node_memories": [
            ],
            "function_names": functions[5],
            "function_memories": [
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200
            ],
            "function_cold_starts": [
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500
            ],
            "function_max_delays": [
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500
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
                {"name": "f0", "users": 2, "nrt": 10},
                {"name": "f1", "users": 2, "nrt": 10},
                {"name": "f2", "users": 2, "nrt": 10},
                {"name": "f3", "users": 2, "nrt": 10},
                {"name": "f4", "users": 2, "nrt": 10},
                {"name": "f5", "users": 2, "nrt": 10},
                {"name": "f6", "users": 2, "nrt": 10},
                {"name": "f7", "users": 2, "nrt": 10},
                {"name": "f8", "users": 2, "nrt": 10},
                {"name": "f9", "users": 2, "nrt": 10},
                {"name": "f10", "users": 2, "nrt": 10},
                {"name": "f11", "users": 2, "nrt": 10},
                {"name": "f12", "users": 2, "nrt": 10},
                {"name": "f13", "users": 2, "nrt": 10},
                {"name": "f14", "users": 2, "nrt": 10},
                {"name": "f15", "users": 2, "nrt": 10},
                {"name": "f16", "users": 2, "nrt": 10},
                {"name": "f17", "users": 2, "nrt": 10},
                {"name": "f18", "users": 2, "nrt": 10},
                {"name": "f19", "users": 2, "nrt": 10},
                {"name": "f20", "users": 2, "nrt": 10},
                {"name": "f21", "users": 2, "nrt": 10},
                {"name": "f22", "users": 2, "nrt": 10},
                {"name": "f23", "users": 2, "nrt": 10},
                {"name": "f24", "users": 2, "nrt": 10},
                {"name": "f25", "users": 2, "nrt": 10},
                {"name": "f26", "users": 2, "nrt": 10},
                {"name": "f27", "users": 2, "nrt": 10},
                {"name": "f28", "users": 2, "nrt": 10},
                {"name": "f29", "users": 2, "nrt": 10},

            ],
            "edges": [
                {"source": "f0", "target": "f1", "sync": 1, "times": 1},
                {"source": "f0", "target": "f2", "sync": 2, "times": 1},
                {"source": "f0", "target": "f3", "sync": 2, "times": 1},
                {"source": "f1", "target": "f4", "sync": 3, "times": 1},

                {"source": "f1", "target": "f5", "sync": 4, "times": 1},
                {"source": "f2", "target": "f6", "sync": 5, "times": 1},
                {"source": "f2", "target": "f7", "sync": 6, "times": 1},
                {"source": "f3", "target": "f8", "sync": 7, "times": 1},
                {"source": "f3", "target": "f9", "sync": 7, "times": 1},

                {"source": "f4", "target": "f10", "sync": 8, "times": 1},
                {"source": "f4", "target": "f11", "sync": 9, "times": 1},
                {"source": "f4", "target": "f12", "sync": 10, "times": 1},
                {"source": "f6", "target": "f15", "sync": 13, "times": 1},
                {"source": "f6", "target": "f16", "sync": 14, "times": 1},
                {"source": "f6", "target": "f17", "sync": 14, "times": 1},
                {"source": "f8", "target": "f13", "sync": 11, "times": 1},
                {"source": "f8", "target": "f14", "sync": 12, "times": 1},

                {"source": "f9", "target": "f18", "sync": 15, "times": 1},
                {"source": "f9", "target": "f19", "sync": 16, "times": 1},

                {"source": "f11", "target": "f20", "sync": 17, "times": 1},
                {"source": "f11", "target": "f21", "sync": 17, "times": 1},
                {"source": "f16", "target": "f22", "sync": 18, "times": 1},
                {"source": "f16", "target": "f23", "sync": 18, "times": 1},
                {"source": "f16", "target": "f24", "sync": 19, "times": 1},

                {"source": "f17", "target": "f25", "sync": 20, "times": 1},
                {"source": "f17", "target": "f26", "sync": 21, "times": 1},
                {"source": "f18", "target": "f27", "sync": 22, "times": 1},
                {"source": "f18", "target": "f28", "sync": 22, "times": 1},
                {"source": "f18", "target": "f29", "sync": 23, "times": 1}
            ],
            "m": [
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            "parallel_scheduler": [
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11],
                [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24],
                [25], [26], [27], [28], [29]
            ]
        }
    ]
