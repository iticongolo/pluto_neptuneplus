M = 10**6
epsilon = 10**-6

# If a function `f` is deployed on node i then c[f,i] is True
def constrain_c_according_to_x(data, solver, c, x):
    for f in range(len(data.functions)):
        for j in range(len(data.nodes)):
            solver.Add(
                solver.Sum([
                    x[f, i, j] for i in range(len(data.sources))
                ]) <= c[f, j] * M)
            solver.Add(
                solver.Sum([
                    x[f, i,  j] for i in range(len(data.sources))
                ]) + epsilon >= c[f, j])

# The sum of the memory of functions deployed on a node is less than its capacity
def constrain_memory_usage(data, solver, c):
    for j in range(len(data.nodes)):
        solver.Add(
            solver.Sum([
                c[f, j] * data.function_memory_matrix[f] for f in range(len(data.functions))
            ]) <= data.node_memory_matrix[j])


# All requests in a node are rerouted somewhere else
def constrain_handle_all_requests(data, solver, x, eq=True):
    op = lambda x: x == 1 if eq else lambda x: x <= 1
    for f in range(len(data.functions)):
        for i in range(len(data.sources)):
            solver.Add(
                op(solver.Sum([
                    x[f, i, j] for j in range(len(data.nodes))
                ])))


# All requests except the ones managed by GPUs in a node are rerouted somewhere else
def constrain_handle_only_remaining_requests(data, solver, x):
    for f in range(len(data.functions)):
        for i in range(len(data.sources)):
            solver.Add(
                solver.Sum([
                    x[f, i, j] for j in range(len(data.nodes))
                ]) == 1 - data.prev_x[i][f].sum())


def constrain_handle_required_requests(data, solver, x):
    # Handle all requests
    if data.prev_x.shape == (0,):
        constrain_handle_all_requests(data, solver, x)
    # Handle all requests except for the ones handled by GPUs
    else:
        constrain_handle_only_remaining_requests(data, solver, x)


# Do not overload nodes' CPUs
def constrain_CPU_usage(data, solver, x):
    # print(f'data.CORES={data.core_per_req_matrix}')
    # for f in range(len(data.functions)):
    #     for i in range(len(data.nodes)):
            #print(f'data.CORES[{f}, {i}]={data.core_per_req_matrix[f, i]}')
    # print(f'ata.nodes={data.nodes}')
    # print(f'ata.source={data.sources}')
    for j in range(len(data.nodes)):
        solver.Add(
            solver.Sum([
                x[f, i, j] * data.workload_matrix[f, i] * data.core_per_req_matrix[f, j] for f in
                range(len(data.functions)) for i in
                range(len(data.sources))
            ]) <= data.node_cores_matrix[j]
        )


# If a node i contains one or more functions then n[i] is True
def constrain_n_according_to_c(data, solver, n, c):
    for i in range(len(data.nodes)):
        solver.Add(
            solver.Sum([
                c[f, i] for f in range(len(data.functions))
            ]) <= n[i] * M)
        solver.Add(
            solver.Sum([
                c[f, i] for f in range(len(data.functions))
            ]) + epsilon >= n[i])


# The sum of the memory of functions deployed on a gpu device is less than its capacity
def constrain_GPU_memory_usage(data, solver, c):
    for j in range(len(data.nodes)):
        solver.Add(
            solver.Sum([
                c[f, j] * data.gpu_function_memory_matrix[f] for f in range(len(data.functions))
            ]) <= data.gpu_node_memory_matrix[j])


# Do not overload nodes' GPUs
def constrain_GPU_usage(data, solver, x):
    for f in range(len(data.functions)):
        for j in range(len(data.nodes)):
            solver.Add(
                solver.Sum([
                    x[f, i, j] * data.workload_matrix[f, i] * data.response_time_matrix[f, j] for i in
                    range(len(data.sources))
                ]) <= 1000)


def constrain_budget(data, solver, n):
    for j in range(len(data.nodes)):
        solver.Add(n[j] * data.node_costs[j] <= data.node_budget)
