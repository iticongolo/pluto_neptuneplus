class ApplicationsInputs:

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

                 ["ns/f0", "ns/f1", "ns/f2", "ns/f3", "ns/f4", "ns/f5", "ns/f6", "ns/f7", "ns/f8",
                  "ns/f9", "ns/f10", "ns/f11", "ns/f12", "ns/f13", "ns/f14", "ns/f15", "ns/f16", "ns/f17", "ns/f18",
                  "ns/f19", "ns/f20", "ns/f21", "ns/f22", "ns/f23", "ns/f24", "ns/f25", "ns/f26", "ns/f27", "ns/f28",
                  "ns/f29", "ns/f30", "ns/f31", "ns/f32", "ns/f33", "ns/f34"],

                 ["ns/f0", "ns/f1", "ns/f2", "ns/f3", "ns/f4", "ns/f5", "ns/f6", "ns/f7", "ns/f8",
                  "ns/f9", "ns/f10", "ns/f11", "ns/f12", "ns/f13", "ns/f14", "ns/f15", "ns/f16", "ns/f17", "ns/f18",
                  "ns/f19", "ns/f20", "ns/f21", "ns/f22", "ns/f23", "ns/f24", "ns/f25", "ns/f26", "ns/f27", "ns/f28",
                  "ns/f29", "ns/f30", "ns/f31", "ns/f32", "ns/f33", "ns/f34", "ns/f35", "ns/f36", "ns/f37", "ns/f38",
                  "ns/f39"],

                 ["ns/f0", "ns/f1", "ns/f2", "ns/f3", "ns/f4", "ns/f5", "ns/f6", "ns/f7", "ns/f8",
                  "ns/f9", "ns/f10", "ns/f11", "ns/f12", "ns/f13", "ns/f14", "ns/f15", "ns/f16", "ns/f17", "ns/f18",
                  "ns/f19", "ns/f20", "ns/f21", "ns/f22", "ns/f23", "ns/f24", "ns/f25", "ns/f26", "ns/f27", "ns/f28",
                  "ns/f29", "ns/f30", "ns/f31", "ns/f32", "ns/f33", "ns/f34", "ns/f35", "ns/f36", "ns/f37", "ns/f38",
                  "ns/f39", "ns/f40", "ns/f41", "ns/f42", "ns/f43", "ns/f44"],

                 ["ns/f0", "ns/f1", "ns/f2", "ns/f3", "ns/f4", "ns/f5", "ns/f6", "ns/f7", "ns/f8", "ns/f9",
                  "ns/f10", "ns/f11", "ns/f12", "ns/f13", "ns/f14", "ns/f15", "ns/f16", "ns/f17", "ns/f18",
                  "ns/f19", "ns/f20", "ns/f21", "ns/f22", "ns/f23", "ns/f24", "ns/f25", "ns/f26", "ns/f27", "ns/f28",
                  "ns/f29", "ns/f30", "ns/f31", "ns/f32", "ns/f33", "ns/f34", "ns/f35", "ns/f36", "ns/f37", "ns/f38",
                  "ns/f39", "ns/f40", "ns/f41", "ns/f42", "ns/f43", "ns/f44", "ns/f45", "ns/f46", "ns/f47", "ns/f48",
                  "ns/f49"]
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
            {"source": "f0", "target": "f2", "sync": 2, "times": 2},
            {"source": "f1", "target": "f3", "sync": 3, "times": 1},
            {"source": "f1", "target": "f4", "sync": 3, "times": 1},
        ],
        "m": [
            [0, 1, 2, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]

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
            {"name": "f9", "users": 2, "nrt": 10},

        ],
        "edges": [
            {"source": "f0", "target": "f1", "sync": 1, "times": 1},
            {"source": "f0", "target": "f2", "sync": 2, "times": 2},
            {"source": "f1", "target": "f3", "sync": 3, "times": 1},
            {"source": "f1", "target": "f4", "sync": 3, "times": 1},

            {"source": "f5", "target": "f6", "sync": 1, "times": 1},
            {"source": "f5", "target": "f7", "sync": 2, "times": 2},
            {"source": "f6", "target": "f8", "sync": 3, "times": 1},
            {"source": "f6", "target": "f9", "sync": 3, "times": 1},
        ],
        "m": [
            [0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
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
            {"source": "f0", "target": "f2", "sync": 2, "times": 2},
            {"source": "f1", "target": "f3", "sync": 3, "times": 1},
            {"source": "f1", "target": "f4", "sync": 3, "times": 1},

            {"source": "f5", "target": "f6", "sync": 1, "times": 1},
            {"source": "f5", "target": "f7", "sync": 2, "times": 2},
            {"source": "f6", "target": "f8", "sync": 3, "times": 1},
            {"source": "f6", "target": "f9", "sync": 3, "times": 1},

            {"source": "f10", "target": "f11", "sync": 1, "times": 1},
            {"source": "f10", "target": "f12", "sync": 2, "times": 2},
            {"source": "f11", "target": "f13", "sync": 3, "times": 1},
            {"source": "f11", "target": "f14", "sync": 3, "times": 1},
        ],
        "m": [
            [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
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
                {"source": "f0", "target": "f2", "sync": 2, "times": 2},
                {"source": "f1", "target": "f3", "sync": 3, "times": 1},
                {"source": "f1", "target": "f4", "sync": 3, "times": 1},

                {"source": "f5", "target": "f6", "sync": 1, "times": 1},
                {"source": "f5", "target": "f7", "sync": 2, "times": 2},
                {"source": "f6", "target": "f8", "sync": 3, "times": 1},
                {"source": "f6", "target": "f9", "sync": 3, "times": 1},

                {"source": "f10", "target": "f11", "sync": 1, "times": 1},
                {"source": "f10", "target": "f12", "sync": 2, "times": 2},
                {"source": "f11", "target": "f13", "sync": 3, "times": 1},
                {"source": "f11", "target": "f14", "sync": 3, "times": 1},

                {"source": "f15", "target": "f16", "sync": 1, "times": 1},
                {"source": "f15", "target": "f17", "sync": 2, "times": 2},
                {"source": "f16", "target": "f18", "sync": 3, "times": 1},
                {"source": "f16", "target": "f19", "sync": 3, "times": 1},
            ],
            "m": [
                [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
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
                {"source": "f0", "target": "f2", "sync": 2, "times": 2},
                {"source": "f1", "target": "f3", "sync": 3, "times": 1},
                {"source": "f1", "target": "f4", "sync": 3, "times": 1},

                {"source": "f5", "target": "f6", "sync": 1, "times": 1},
                {"source": "f5", "target": "f7", "sync": 2, "times": 2},
                {"source": "f6", "target": "f8", "sync": 3, "times": 1},
                {"source": "f6", "target": "f9", "sync": 3, "times": 1},

                {"source": "f10", "target": "f11", "sync": 1, "times": 1},
                {"source": "f10", "target": "f12", "sync": 2, "times": 2},
                {"source": "f11", "target": "f13", "sync": 3, "times": 1},
                {"source": "f11", "target": "f14", "sync": 3, "times": 1},

                {"source": "f15", "target": "f16", "sync": 1, "times": 1},
                {"source": "f15", "target": "f17", "sync": 2, "times": 2},
                {"source": "f16", "target": "f18", "sync": 3, "times": 1},
                {"source": "f16", "target": "f19", "sync": 3, "times": 1},

                {"source": "f20", "target": "f21", "sync": 1, "times": 1},
                {"source": "f20", "target": "f22", "sync": 2, "times": 2},
                {"source": "f21", "target": "f23", "sync": 3, "times": 1},
                {"source": "f21", "target": "f24", "sync": 3, "times": 1},
            ],
            "m": [
                [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
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
                {"source": "f0", "target": "f2", "sync": 2, "times": 2},
                {"source": "f1", "target": "f3", "sync": 3, "times": 1},
                {"source": "f1", "target": "f4", "sync": 3, "times": 1},

                {"source": "f5", "target": "f6", "sync": 1, "times": 1},
                {"source": "f5", "target": "f7", "sync": 2, "times": 2},
                {"source": "f6", "target": "f8", "sync": 3, "times": 1},
                {"source": "f6", "target": "f9", "sync": 3, "times": 1},

                {"source": "f10", "target": "f11", "sync": 1, "times": 1},
                {"source": "f10", "target": "f12", "sync": 2, "times": 2},
                {"source": "f11", "target": "f13", "sync": 3, "times": 1},
                {"source": "f11", "target": "f14", "sync": 3, "times": 1},

                {"source": "f15", "target": "f16", "sync": 1, "times": 1},
                {"source": "f15", "target": "f17", "sync": 2, "times": 2},
                {"source": "f16", "target": "f18", "sync": 3, "times": 1},
                {"source": "f16", "target": "f19", "sync": 3, "times": 1},

                {"source": "f20", "target": "f21", "sync": 1, "times": 1},
                {"source": "f20", "target": "f22", "sync": 2, "times": 2},
                {"source": "f21", "target": "f23", "sync": 3, "times": 1},
                {"source": "f21", "target": "f24", "sync": 3, "times": 1},

                {"source": "f25", "target": "f26", "sync": 1, "times": 1},
                {"source": "f25", "target": "f27", "sync": 2, "times": 2},
                {"source": "f26", "target": "f28", "sync": 3, "times": 1},
                {"source": "f26", "target": "f29", "sync": 3, "times": 1},
            ],
            "m": [
                [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            "parallel_scheduler": [
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11],
                [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24],
                [25], [26], [27], [28], [29]
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
                50, 50, 50, 50, 50,
                50, 50, 50, 50, 50
            ],
            "node_cores": [],

            "gpu_node_names": [
            ],
            "gpu_node_memories": [
            ],
            "function_names": functions[6],
            "function_memories": [
                200, 200, 200, 200, 200,
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
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500
            ],
            "function_max_delays": [
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                500, 500, 500, 500, 500,
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
                {"name": "f30", "users": 2, "nrt": 10},
                {"name": "f31", "users": 2, "nrt": 10},
                {"name": "f32", "users": 2, "nrt": 10},
                {"name": "f33", "users": 2, "nrt": 10},
                {"name": "f34", "users": 2, "nrt": 10},

            ],
            "edges": [
                {"source": "f0", "target": "f1", "sync": 1, "times": 1},
                {"source": "f0", "target": "f2", "sync": 2, "times": 2},
                {"source": "f1", "target": "f3", "sync": 3, "times": 1},
                {"source": "f1", "target": "f4", "sync": 3, "times": 1},

                {"source": "f5", "target": "f6", "sync": 1, "times": 1},
                {"source": "f5", "target": "f7", "sync": 2, "times": 2},
                {"source": "f6", "target": "f8", "sync": 3, "times": 1},
                {"source": "f6", "target": "f9", "sync": 3, "times": 1},

                {"source": "f10", "target": "f11", "sync": 1, "times": 1},
                {"source": "f10", "target": "f12", "sync": 2, "times": 2},
                {"source": "f11", "target": "f13", "sync": 3, "times": 1},
                {"source": "f11", "target": "f14", "sync": 3, "times": 1},

                {"source": "f15", "target": "f16", "sync": 1, "times": 1},
                {"source": "f15", "target": "f17", "sync": 2, "times": 2},
                {"source": "f16", "target": "f18", "sync": 3, "times": 1},
                {"source": "f16", "target": "f19", "sync": 3, "times": 1},

                {"source": "f20", "target": "f21", "sync": 1, "times": 1},
                {"source": "f20", "target": "f22", "sync": 2, "times": 2},
                {"source": "f21", "target": "f23", "sync": 3, "times": 1},
                {"source": "f21", "target": "f24", "sync": 3, "times": 1},

                {"source": "f25", "target": "f26", "sync": 1, "times": 1},
                {"source": "f25", "target": "f27", "sync": 2, "times": 2},
                {"source": "f26", "target": "f28", "sync": 3, "times": 1},
                {"source": "f26", "target": "f29", "sync": 3, "times": 1},

                {"source": "f30", "target": "f31", "sync": 1, "times": 1},
                {"source": "f30", "target": "f32", "sync": 2, "times": 2},
                {"source": "f31", "target": "f33", "sync": 3, "times": 1},
                {"source": "f31", "target": "f34", "sync": 3, "times": 1},
            ],
            "m": [
                [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            "parallel_scheduler": [
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11],
                [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24],
                [25], [26], [27], [28], [29], [30], [31], [32], [33], [34]
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
                50, 50, 50, 50, 50,
                50, 50, 50, 50, 50,
                50, 50, 50, 50, 50
            ],
            "node_cores": [],

            "gpu_node_names": [
            ],
            "gpu_node_memories": [
            ],
            "function_names": functions[7],
            "function_memories": [
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200,
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
                500, 500, 500, 500, 500,
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
                {"name": "f30", "users": 2, "nrt": 10},
                {"name": "f31", "users": 2, "nrt": 10},
                {"name": "f32", "users": 2, "nrt": 10},
                {"name": "f33", "users": 2, "nrt": 10},
                {"name": "f34", "users": 2, "nrt": 10},
                {"name": "f35", "users": 2, "nrt": 10},
                {"name": "f36", "users": 2, "nrt": 10},
                {"name": "f37", "users": 2, "nrt": 10},
                {"name": "f38", "users": 2, "nrt": 10},
                {"name": "f39", "users": 2, "nrt": 10},
            ],
            "edges": [
                {"source": "f0", "target": "f1", "sync": 1, "times": 1},
                {"source": "f0", "target": "f2", "sync": 2, "times": 2},
                {"source": "f1", "target": "f3", "sync": 3, "times": 1},
                {"source": "f1", "target": "f4", "sync": 3, "times": 1},

                {"source": "f5", "target": "f6", "sync": 1, "times": 1},
                {"source": "f5", "target": "f7", "sync": 2, "times": 2},
                {"source": "f6", "target": "f8", "sync": 3, "times": 1},
                {"source": "f6", "target": "f9", "sync": 3, "times": 1},

                {"source": "f10", "target": "f11", "sync": 1, "times": 1},
                {"source": "f10", "target": "f12", "sync": 2, "times": 2},
                {"source": "f11", "target": "f13", "sync": 3, "times": 1},
                {"source": "f11", "target": "f14", "sync": 3, "times": 1},

                {"source": "f15", "target": "f16", "sync": 1, "times": 1},
                {"source": "f15", "target": "f17", "sync": 2, "times": 2},
                {"source": "f16", "target": "f18", "sync": 3, "times": 1},
                {"source": "f16", "target": "f19", "sync": 3, "times": 1},

                {"source": "f20", "target": "f21", "sync": 1, "times": 1},
                {"source": "f20", "target": "f22", "sync": 2, "times": 2},
                {"source": "f21", "target": "f23", "sync": 3, "times": 1},
                {"source": "f21", "target": "f24", "sync": 3, "times": 1},

                {"source": "f25", "target": "f26", "sync": 1, "times": 1},
                {"source": "f25", "target": "f27", "sync": 2, "times": 2},
                {"source": "f26", "target": "f28", "sync": 3, "times": 1},
                {"source": "f26", "target": "f29", "sync": 3, "times": 1},

                {"source": "f30", "target": "f31", "sync": 1, "times": 1},
                {"source": "f30", "target": "f32", "sync": 2, "times": 2},
                {"source": "f31", "target": "f33", "sync": 3, "times": 1},
                {"source": "f31", "target": "f34", "sync": 3, "times": 1},

                {"source": "f35", "target": "f36", "sync": 1, "times": 1},
                {"source": "f35", "target": "f37", "sync": 2, "times": 2},
                {"source": "f36", "target": "f38", "sync": 3, "times": 1},
                {"source": "f36", "target": "f39", "sync": 3, "times": 1},
            ],
            "m": [
                [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0]
            ],
            "parallel_scheduler": [
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11],
                [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24],
                [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39]
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
            "function_names": functions[8],
            "function_memories": [
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200,
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
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500,
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200
            ],
            "function_max_delays": [
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500,
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
                {"name": "f30", "users": 2, "nrt": 10},
                {"name": "f31", "users": 2, "nrt": 10},
                {"name": "f32", "users": 2, "nrt": 10},
                {"name": "f33", "users": 2, "nrt": 10},
                {"name": "f34", "users": 2, "nrt": 10},
                {"name": "f35", "users": 2, "nrt": 10},
                {"name": "f36", "users": 2, "nrt": 10},
                {"name": "f37", "users": 2, "nrt": 10},
                {"name": "f38", "users": 2, "nrt": 10},
                {"name": "f39", "users": 2, "nrt": 10},
                {"name": "f40", "users": 2, "nrt": 10},
                {"name": "f41", "users": 2, "nrt": 10},
                {"name": "f42", "users": 2, "nrt": 10},
                {"name": "f43", "users": 2, "nrt": 10},
                {"name": "f44", "users": 2, "nrt": 10},
            ],
            "edges": [
                {"source": "f0", "target": "f1", "sync": 1, "times": 1},
                {"source": "f0", "target": "f2", "sync": 2, "times": 2},
                {"source": "f1", "target": "f3", "sync": 3, "times": 1},
                {"source": "f1", "target": "f4", "sync": 3, "times": 1},

                {"source": "f5", "target": "f6", "sync": 1, "times": 1},
                {"source": "f5", "target": "f7", "sync": 2, "times": 2},
                {"source": "f6", "target": "f8", "sync": 3, "times": 1},
                {"source": "f6", "target": "f9", "sync": 3, "times": 1},

                {"source": "f10", "target": "f11", "sync": 1, "times": 1},
                {"source": "f10", "target": "f12", "sync": 2, "times": 2},
                {"source": "f11", "target": "f13", "sync": 3, "times": 1},
                {"source": "f11", "target": "f14", "sync": 3, "times": 1},

                {"source": "f15", "target": "f16", "sync": 1, "times": 1},
                {"source": "f15", "target": "f17", "sync": 2, "times": 2},
                {"source": "f16", "target": "f18", "sync": 3, "times": 1},
                {"source": "f16", "target": "f19", "sync": 3, "times": 1},

                {"source": "f20", "target": "f21", "sync": 1, "times": 1},
                {"source": "f20", "target": "f22", "sync": 2, "times": 2},
                {"source": "f21", "target": "f23", "sync": 3, "times": 1},
                {"source": "f21", "target": "f24", "sync": 3, "times": 1},

                {"source": "f25", "target": "f26", "sync": 1, "times": 1},
                {"source": "f25", "target": "f27", "sync": 2, "times": 2},
                {"source": "f26", "target": "f28", "sync": 3, "times": 1},
                {"source": "f26", "target": "f29", "sync": 3, "times": 1},

                {"source": "f30", "target": "f31", "sync": 1, "times": 1},
                {"source": "f30", "target": "f32", "sync": 2, "times": 2},
                {"source": "f31", "target": "f33", "sync": 3, "times": 1},
                {"source": "f31", "target": "f34", "sync": 3, "times": 1},

                {"source": "f35", "target": "f36", "sync": 1, "times": 1},
                {"source": "f35", "target": "f37", "sync": 2, "times": 2},
                {"source": "f36", "target": "f38", "sync": 3, "times": 1},
                {"source": "f36", "target": "f39", "sync": 3, "times": 1},

                {"source": "f40", "target": "f41", "sync": 1, "times": 1},
                {"source": "f40", "target": "f42", "sync": 2, "times": 2},
                {"source": "f41", "target": "f43", "sync": 3, "times": 1},
                {"source": "f41", "target": "f44", "sync": 3, "times": 1},
            ],
            "m": [
                [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            "parallel_scheduler": [
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11],
                [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24],
                [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39],
                [40], [41], [42], [43], [44]
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
            "function_names": functions[9],
            "function_memories": [
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200,
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
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500,
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200,
                200, 200, 200, 200, 200
            ],
            "function_max_delays": [
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100,
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500,
                500, 500, 500, 500, 500,
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
                {"name": "f30", "users": 2, "nrt": 10},
                {"name": "f31", "users": 2, "nrt": 10},
                {"name": "f32", "users": 2, "nrt": 10},
                {"name": "f33", "users": 2, "nrt": 10},
                {"name": "f34", "users": 2, "nrt": 10},
                {"name": "f35", "users": 2, "nrt": 10},
                {"name": "f36", "users": 2, "nrt": 10},
                {"name": "f37", "users": 2, "nrt": 10},
                {"name": "f38", "users": 2, "nrt": 10},
                {"name": "f39", "users": 2, "nrt": 10},
                {"name": "f40", "users": 2, "nrt": 10},
                {"name": "f41", "users": 2, "nrt": 10},
                {"name": "f42", "users": 2, "nrt": 10},
                {"name": "f43", "users": 2, "nrt": 10},
                {"name": "f44", "users": 2, "nrt": 10},
                {"name": "f45", "users": 2, "nrt": 10},
                {"name": "f46", "users": 2, "nrt": 10},
                {"name": "f47", "users": 2, "nrt": 10},
                {"name": "f48", "users": 2, "nrt": 10},
                {"name": "f49", "users": 2, "nrt": 10},
            ],
            "edges": [
                {"source": "f0", "target": "f1", "sync": 1, "times": 1},
                {"source": "f0", "target": "f2", "sync": 2, "times": 2},
                {"source": "f1", "target": "f3", "sync": 3, "times": 1},
                {"source": "f1", "target": "f4", "sync": 3, "times": 1},

                {"source": "f5", "target": "f6", "sync": 1, "times": 1},
                {"source": "f5", "target": "f7", "sync": 2, "times": 2},
                {"source": "f6", "target": "f8", "sync": 3, "times": 1},
                {"source": "f6", "target": "f9", "sync": 3, "times": 1},

                {"source": "f10", "target": "f11", "sync": 1, "times": 1},
                {"source": "f10", "target": "f12", "sync": 2, "times": 2},
                {"source": "f11", "target": "f13", "sync": 3, "times": 1},
                {"source": "f11", "target": "f14", "sync": 3, "times": 1},

                {"source": "f15", "target": "f16", "sync": 1, "times": 1},
                {"source": "f15", "target": "f17", "sync": 2, "times": 2},
                {"source": "f16", "target": "f18", "sync": 3, "times": 1},
                {"source": "f16", "target": "f19", "sync": 3, "times": 1},

                {"source": "f20", "target": "f21", "sync": 1, "times": 1},
                {"source": "f20", "target": "f22", "sync": 2, "times": 2},
                {"source": "f21", "target": "f23", "sync": 3, "times": 1},
                {"source": "f21", "target": "f24", "sync": 3, "times": 1},

                {"source": "f25", "target": "f26", "sync": 1, "times": 1},
                {"source": "f25", "target": "f27", "sync": 2, "times": 2},
                {"source": "f26", "target": "f28", "sync": 3, "times": 1},
                {"source": "f26", "target": "f29", "sync": 3, "times": 1},

                {"source": "f30", "target": "f31", "sync": 1, "times": 1},
                {"source": "f30", "target": "f32", "sync": 2, "times": 2},
                {"source": "f31", "target": "f33", "sync": 3, "times": 1},
                {"source": "f31", "target": "f34", "sync": 3, "times": 1},

                {"source": "f35", "target": "f36", "sync": 1, "times": 1},
                {"source": "f35", "target": "f37", "sync": 2, "times": 2},
                {"source": "f36", "target": "f38", "sync": 3, "times": 1},
                {"source": "f36", "target": "f39", "sync": 3, "times": 1},

                {"source": "f40", "target": "f41", "sync": 1, "times": 1},
                {"source": "f40", "target": "f42", "sync": 2, "times": 2},
                {"source": "f41", "target": "f43", "sync": 3, "times": 1},
                {"source": "f41", "target": "f44", "sync": 3, "times": 1},

                {"source": "f45", "target": "f46", "sync": 1, "times": 1},
                {"source": "f45", "target": "f47", "sync": 2, "times": 2},
                {"source": "f46", "target": "f48", "sync": 3, "times": 1},
                {"source": "f46", "target": "f49", "sync": 3, "times": 1},
            ],
            "m": [
                [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],

            "parallel_scheduler": [
                [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11],
                [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24],
                [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39],
                [40], [41], [42], [43], [44], [45], [46], [47], [48], [49]
            ]
        },
    ]
