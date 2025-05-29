from logging.config import dictConfig
from core.solvers import *
from flask import Flask, request
from core import Statistics as st
# from core import *
# from core.solvers import *
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# from core import Statistics as st
from core import Parameters as param
from core.utils.heu_xu_et_al import HeuXu
from core.utils.pluto_heuristic import PLUTO
from core.utils.topology_generator import NetworkGenerator
from core.utils.util import *
import matplotlib.pyplot as plt

from core.utils.util_clustering import generate_dynamic_clustering
# from real_world_parameters import RealWorldParameters as param_real_world
import os

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://sys.stdout',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

app = Flask(__name__)
app.app_context()


def fill_formal_optimization(output_matrix_lines, servers_ids, function_ids, lambs,
                             input, sumlamb, max_delay, clusterings=None, reunse_clusterings=False, server_locations=None, cores=None,
                             scale=10, scale_optional=5, qty_values=1, has_coldstart=True, has_decision_delay=True, lambs_var_app=None, fixed_lamb=True,
                             vary_topologies=True, var_apps_topology=False, vary_functions=False, real_world=False, hop_delays=None, solver_type='NeptuneMinDelayAndUtilization'):
    clusts = copy.deepcopy(clusterings)
    response = None
    list_nodes_qty = []
    formal_approach_possible_total_delay = np.zeros(qty_values)
    formal_approach_possible_coldstart = np.zeros(qty_values)
    formal_approach_possible_network_delay = np.zeros(qty_values)
    formal_approach_possible_memory = np.zeros(qty_values)
    delays_formal_approach = np.zeros((output_matrix_lines, 2))
    nodes_formal_approach = np.zeros((output_matrix_lines, 2))
    memory_formal_approach = np.zeros((output_matrix_lines, 2))

    formal_approach_possible_decision_delay = np.zeros(qty_values)
    decision_delay_formal_approach = np.zeros((output_matrix_lines, 2))
    avg_decision_delay_formal_approach = np.zeros((output_matrix_lines, 2))

    network_delay_formal_approach = np.zeros((output_matrix_lines, 2))
    avg_network_delay_formal_approach = np.zeros((output_matrix_lines, 2))
    coldstart_delay_formal_approach = np.zeros((output_matrix_lines, 2))

    # f = len(inp["function_names"])
    lamb = fixed_lamb  # fix workload for entrypoint function
    # workload[0][0] = lamb
    # json_workload = json.dumps(workload)
    # inp["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0
    memories = []
    first_sum_lamb = 1
    for i in range(output_matrix_lines):
        sumlamb1 = 0
        sumlamb, universal_param, data_list, _, _, _, func_ids, clusts = update_resources(copy.deepcopy(clusterings), copy.deepcopy(servers_ids),
                                                                                          copy.deepcopy(function_ids), lambs, i, scale, sumlamb1,
                                                                                          input, scale_optional=scale_optional, reunse_clusterings=reunse_clusterings, original_clusterings=clusterings,
                                                                                          server_locations=server_locations, cores=cores, lambs_var_app=lambs_var_app, fixed_lamb=fixed_lamb,
                                                                                          vary_topologies=vary_topologies, var_apps_and_topology=var_apps_topology, vary_functions=vary_functions, real_world=real_world, hop_delays=hop_delays)
        inp = input
        if var_apps_topology or (fixed_lamb and (not vary_topologies)):
            inp = input[i]
        if i == 0:
            first_sum_lamb = max(first_sum_lamb, sumlamb)

        print(f'{solver_type} sumlamb={sumlamb}')

        total_nodes_used = []
        total_memory_used = []
        cores_used = []
        total_cold_starts = []
        total_network_delay = []
        total_placement_time = []
        previous_memory = 0

        data_list_temp = copy.deepcopy(data_list)
        for data in data_list_temp:
            c = data.cluster
            for server in c.servers:
                cores_available, memory_available = c.get_server_available_resources(server)
                print(f'[T1-{i}]ServerAAA[{server.id}]={cores_available, memory_available}')

            hide_functions_without_workload(data, func_ids)
            memory_usage_formal_approach = 0
            for j in range(qty_values):
                solver = inp.get("solver", {'type': solver_type})

                # "NeptuneMinDelayAndUtilization",
                # "NeptuneMinDelay",
                # "NeptuneMinUtilization",
                # "VSVBP",
                # "Criticality",
                # "CriticalityHeuristic",
                # "MCF"
                solver_type = solver.get("type")
                solver_args = solver.get("args", {})
                with_db = inp.get("with_db", True)
                solver = eval(solver_type)(**solver_args)
                # print(f'###########################')
                solver.load_data(data_to_solver_input(inp, with_db=with_db, data_temp=data, cpu_coeff=inp.get("cpu_coeff", 1.3)))
                # print(f'@@@@@@@@@@@@@@@@@@@@@@@@')
                solver.solve()
                x, c = solver.results()
                qty_nodes, _ = solver.get_resources_usage()
                list_nodes_qty.append(qty_nodes)
                total_delay, coldstart, network_delay, decision_time = solver.object_function_global_results()

                decision_delay = round(decision_time)
                # print(f'NEPTUNE  decision_delay= {decision_delay}')
                delay_formal_approach = coldstart*has_coldstart + network_delay + decision_delay*has_decision_delay

                formal_approach_possible_total_delay[j] = delay_formal_approach
                formal_approach_possible_coldstart[j] = coldstart
                formal_approach_possible_network_delay[j] = network_delay
                formal_approach_possible_decision_delay[j] = decision_delay
                score = solver.score()
                if j == qty_values-1:
                    # memory_usage_formal_approach = solver.get_memory_used(inp["function_memories"])
                    memory_usage_formal_approach = solver.get_memory_used(data.function_memories)
                    memories.append(data.function_memories)

                response = app.response_class(
                    response=json.dumps({
                        "cpu_routing_rules": x,
                        "cpu_allocations": c,
                        "gpu_routing_rules": {},
                        "gpu_allocations": {},
                        "score": score
                    }),
                    status=200,
                    mimetype='application/json'
                )
            c = data.cluster
            for server in c.servers:
                cores_available, memory_available = c.get_server_available_resources(server)
                print(f'[T2-{i}]ServerAAA[{server.id}]={cores_available, memory_available}')

            # total_memory_used.append(memory_usage_formal_approach)
            total_memory_used.append(memory_usage_formal_approach)
            total_nodes_used.append(np.mean(list_nodes_qty))
            total_cold_starts.append(np.mean(formal_approach_possible_coldstart))
            total_network_delay.append(np.mean(formal_approach_possible_network_delay))
            total_placement_time.append(np.mean(formal_approach_possible_decision_delay))

        # print(f'total_memory_used, total_nodes_used, total_cold_starts, total_network_delay, total_placement_time')
        # print(f'{total_memory_used}-{total_nodes_used}- {total_cold_starts}- {total_network_delay}- {total_placement_time}')

        memory = sum(total_memory_used)
        network_delay = sum(total_network_delay)
        placement_delay = sum(total_placement_time)
        coldstart = np.max(total_cold_starts)
        total_nodes = sum(total_nodes_used)
        if sumlamb==0:
            sumlamb=1
        avg_network_delay = network_delay/sumlamb
        avg_placement_delay = placement_delay / sumlamb

        total_delay_formal_approach = fill_output(first_sum_lamb, sumlamb, coldstart, coldstart_delay_formal_approach, placement_delay, avg_placement_delay,
                                          decision_delay_formal_approach,avg_decision_delay_formal_approach, delays_formal_approach, i, memory,
                                          memory_formal_approach, network_delay, avg_network_delay, network_delay_formal_approach,
                                          avg_network_delay_formal_approach, nodes_formal_approach, total_nodes, universal_param,
                                          vary_topologies, has_coldstart=has_coldstart, has_decision_delay=has_decision_delay,
                                          real_world=real_world)

        if total_delay_formal_approach > max_delay:
          max_delay = total_delay_formal_approach

    if response is None:
        response = app.response_class(
            response=json.dumps({
                "cpu_routing_rules": {},
                "cpu_allocations": {},
                "gpu_routing_rules": {},
                "gpu_allocations": {},
                "score": {}
            }),
            status=200,
            mimetype='application/json'
        )

    return response, sumlamb, delays_formal_approach, memory_formal_approach, coldstart_delay_formal_approach, network_delay_formal_approach,\
        avg_network_delay_formal_approach, decision_delay_formal_approach, avg_decision_delay_formal_approach, nodes_formal_approach, max_delay, clusts


# inputs:
# output_matrix_lines defines the set of topologies, servers_ids sets, etc. TODO REVISE RESOURCE USAGE METHOD
# output_matrix_lines (number points to plot. e.g.: 10 points)
# clusterings (number of infrastructures. Each infrastructure includes Topology and in topology we have clusters.
# e.g.: clusters=[dynamic_clustering1, dynamic_clustering2,..., dynamic_clustering_n] )
# servers_ids (set of servers receiving requests for each topology. e.g.: servers_id=[[2,4,9,11], [2,4,9,11,1,0], ...)
# function_ids (set of entry-point function ids receiving requests for each set of application represente by a DAG
# e.g.: function_ids=[[0,1,5,7], [0,1,5,7, 15, 20], ...)
# lambs ( set of workload for each set of entry_point functions of each set of applications.
# e.g.: lam=[[10,30,20,50], [10,30,20,50,40,5], ...]
def fill_pluto(output_matrix_lines, servers_ids, function_ids, lambs,
                             input, sumlamb, max_delay, scale_optional=5, clusterings=None, reunse_clusterings=False, server_locations=None, cores=None, scale=10, qty_values=1, has_coldstart=True,
                             has_decision_delay=True, lambs_var_app=None, fixed_lamb=True,  vary_topologies=True, var_apps_topology=False, vary_functions=False, real_world=False, hop_delays=None):
    # print(f'@@@@@@@@@@@@ clusterings= {[[(s.id, s.cores, s.memory) for s in dy.topology.servers] for dy in clusterings]}')
    # print(
    #     f'@@@@@@@@@@@@ clusterings= {[[s.id for s in dy.topology.servers] for dy in clusterings]}')
    clusts = copy.deepcopy(clusterings)
    pluto_possible_placement_delay = np.zeros(qty_values)
    delays_pluto = np.zeros((output_matrix_lines, 2))
    nodes_pluto = np.zeros((output_matrix_lines, 2))
    memory_pluto = np.zeros((output_matrix_lines, 2))
    network_delay_pluto = np.zeros((output_matrix_lines, 2))
    avg_network_delay_pluto = np.zeros((output_matrix_lines, 2))
    coldstart_delay_pluto = np.zeros((output_matrix_lines, 2))
    decision_delay_pluto = np.zeros((output_matrix_lines, 2))
    avg_decision_delay_pluto = np.zeros((output_matrix_lines, 2))
    first_sum_lamb = 1
    # inp["workload_on_source_matrix"] = json_workload  # send workload to function f0 in node 0
    # fix workload and applications, vary topologies
    for i in range(output_matrix_lines):
        sumlamb1 =0
        sumlamb, universal_param, data_list, _, _, _, func_ids, clusts = \
            update_resources(copy.deepcopy(clusterings), copy.deepcopy(servers_ids), copy.deepcopy(function_ids),
                             lambs, i, scale, sumlamb1, input, scale_optional=scale_optional,
                             reunse_clusterings=reunse_clusterings, original_clusterings=clusterings,
                             server_locations=server_locations, cores=cores, lambs_var_app=lambs_var_app,
                             fixed_lamb=fixed_lamb, vary_topologies=vary_topologies,
                             var_apps_and_topology=var_apps_topology, vary_functions=vary_functions,
                             real_world=real_world, hop_delays=hop_delays)
        if i == 0:
            first_sum_lamb = max(first_sum_lamb, sumlamb)
        print(f'PLUTO sumlamb={sumlamb}')
        perc_workload_balance = 1.0
        total_nodes_used = []
        memory_used = []
        cores_used = []
        total_cold_starts = []
        total_network_delay = []
        total_placement_time = []
        data_list_temp = copy.deepcopy(data_list)
        for data in data_list_temp:
            asc = PLUTO(data)
            asc.compute_resources_contention(data.cluster.total_workload_per_f, data.cluster.cores_needed_per_f)
            c = data.cluster
            for server in c.servers:
                cores_available, memory_available = c.get_server_available_resources(server)
                print(f'[{i}]ServerAAA[{server.id}]={cores_available, memory_available}')

            x, y, z, w, node_cpu_available, node_memory_available, instance_fj, decision_time = \
                asc.heuristic_placement()
            _, coldstart, network_delay = asc.object_function_heuristic(w, x, y, z)
            total_nodes, memory, cores = asc.resource_usage()
            total_cold_starts.append(coldstart)
            total_network_delay.append(network_delay)
            total_nodes_used.append(total_nodes)
            memory_used.append(memory)
            cores_used.append(cores)
            pluto_possible_placement_delay[0] = decision_time
            for j in range(1, qty_values):
                data_list_temp = copy.deepcopy(data_list)
                clusters_decision_time = 0
                for data in data_list_temp:
                    asc = PLUTO(data)
                    _, _, _, _, _, _, _, decision_time = \
                        asc.heuristic_placement()
                    clusters_decision_time = clusters_decision_time + decision_time
                pluto_possible_placement_delay[j] = clusters_decision_time

        placement_delay = np.mean(pluto_possible_placement_delay) + 0.0
        total_nodes = sum(total_nodes_used)
        memory = sum(memory_used)
        cores = sum(cores_used)
        coldstart = max(total_cold_starts)
        network_delay = sum(total_network_delay)
        avg_network_delay = network_delay/sumlamb
        avg_placement_delay=placement_delay/sumlamb
        print(f'avg_network_delay = {network_delay}/{sumlamb}')
        # delay_pluto = coldstart * has_coldstart + network_delay + round(placement_delay) * has_decision_delay

        total_delay_pluto = fill_output(first_sum_lamb, sumlamb, coldstart, coldstart_delay_pluto, placement_delay, avg_placement_delay, decision_delay_pluto,
                                        avg_decision_delay_pluto, delays_pluto, i, memory, memory_pluto, network_delay, avg_network_delay,
                                        network_delay_pluto, avg_network_delay_pluto, nodes_pluto,
                                        total_nodes, universal_param, vary_topologies, has_coldstart=has_coldstart,
                                        has_decision_delay=has_decision_delay, real_world=real_world)

        if total_delay_pluto > max_delay:
            max_delay = total_delay_pluto
    response = app.response_class(
        response=json.dumps({
            "cpu_routing_rules": {},
            "cpu_allocations": {},
            "gpu_routing_rules": {},
            "gpu_allocations": {},
        }),
        status=200,
        mimetype='application/json'
    )

    return response, sumlamb, delays_pluto, memory_pluto, coldstart_delay_pluto, network_delay_pluto, \
        avg_network_delay_pluto, decision_delay_pluto, avg_decision_delay_pluto, nodes_pluto, max_delay, clusts

# NOTE DONE


# NOTE: DONE
def update_resources(clusterings, servers_ids, function_ids,
                     lambs, i, scale, sumlamb, input, scale_optional=5, reunse_clusterings=False, original_clusterings=None, server_locations=None, cores=None,
                     lambs_var_app=None, fixed_lamb=True, vary_topologies=True, is_heu=False, var_apps_and_topology=False, vary_functions=False, real_world=False, hop_delays=None):
    # Scale-optional is used for number of servers to add in each step in case of varying both, workload and topology
    if var_apps_and_topology:
        inp = input[i]
        funct = input[i]["function_names"] # returns ["ns/f0", "ns/f1", "ns/f2", "ns/f3", "ns/f4" ...]
        # remove ns/
        func= get_cleaned_functions(funct)
        functions=generate_functions(func)

        if len(original_clusterings) < i+1:

            old_dynamic_clusters = copy.deepcopy(original_clusterings[i-1])
            old_server_locations = copy.deepcopy(old_dynamic_clusters.topology.server_locations)
            # for cases that we are not using topology generated before by other approaches
            if not reunse_clusterings:
                network = NetworkGenerator()
                server_locations, has_changed = network.append_servers_positions(i, old_server_locations, qty=scale,
                                                                                 server_locations=server_locations)
                if has_changed:
                    dynamic_clustering = generate_dynamic_clustering(server_locations, old_servers=copy.deepcopy(old_dynamic_clusters.topology.servers),
                                                                     hop_delays=hop_delays, cores=cores, functions=functions)
                    clusterings.append(copy.deepcopy(dynamic_clustering))
                    original_clusterings.append(dynamic_clustering)
            else:
                original_clusterings.append(clusterings[i])
        clusterings[i].functions = functions
        clusterings[i].topology.set_functions(functions)
        clusts = copy.deepcopy(clusterings)
        topology = clusterings[i].topology
        topo = copy.deepcopy(topology)

        lamb = lambs_var_app[i]
        lambs = lambs_var_app[i]  # the workload for each set of applications
        sumlamb = sum(lambs)
        servers_id = servers_ids[i]
        func_ids = function_ids[i]

        param = i+1  # it defines the complexity
        data_list = get_data_list(clusterings[i], input[i])
        # in this case clusterings has multiple topologies and servers_ids
        # a list of servers receiving workload for each topology
        # NOTE check servers_ids[i]
        workloads = set_workload(clusterings[i], data_list, servers_ids[i], function_ids[i],
                                 lambs, input[i], is_heu=is_heu)
    else:
        inp = input
        if fixed_lamb:
            if vary_topologies:
                if len(original_clusterings) < i+1:
                    old_dynamic_clusters = original_clusterings[i-1]
                    old_server_locations = old_dynamic_clusters.topology.server_locations
                    # for cases that we are not using topology generated before by other approaches
                    if not reunse_clusterings:
                        network = NetworkGenerator()
                        server_locations = network.append_servers_positions(old_server_locations, qty=scale,server_locations=server_locations)
                        dynamic_clustering = generate_dynamic_clustering(server_locations, old_servers=copy.deepcopy(old_dynamic_clusters.topology.servers),
                                                                         functions=old_dynamic_clusters.functions)
                        clusterings.append(copy.deepcopy(dynamic_clustering))
                        original_clusterings.append(dynamic_clustering)
                    else:
                        original_clusterings.append(clusterings[i])
                clusts = copy.deepcopy(clusterings)
                topology = clusterings[i].topology
                topo = copy.deepcopy(topology)
                lamb = lambs[0]
                sumlamb =sum(lamb)
                servers_id = servers_ids[0]
                func_ids = function_ids[0]
                param = len(topology.servers)
                data_list = get_data_list(clusterings[i], input)
                # in this case clusterings has multiple topologies and servers_ids
                # a list of servers receiving workload for each topology
                # NOTE check servers_ids[i]
                workloads = set_workload(clusterings[i], data_list, servers_ids[0], function_ids[0],
                                         lambs[0], input, is_heu=is_heu)
            else:   # vary_applications
                funct = input[i]["function_names"]  # returns ["ns/f0", "ns/f1", "ns/f2", "ns/f3", "ns/f4" ...]
                # remove ns/
                func = get_cleaned_functions(funct)
                functions = generate_functions(func)
                clusterings[0].functions = functions
                clusterings[0].topology.set_functions(functions)
                clusts = copy.deepcopy(clusterings)
                topo = copy.deepcopy(clusterings[0].topology)
                if vary_functions: # since we have a single app at a time, we use only one
                    # fixed entrypoint function, server and fixed workload
                    lamb = lambs_var_app[0]
                    lambs = lambs_var_app[0]  # the workload for each set of applications
                    param = len(funct)  # qty of applications on each cluster of the topology. Only the ones receiving workload
                    servers_id = servers_ids[0]
                    func_ids = function_ids[0]
                else:
                    lamb = lambs_var_app[i]
                    lambs = lambs_var_app[i]  # the workload for each set of applications
                    param = i + 1  # qty of applications on each cluster of the topology. Only the ones receiving workload
                    servers_id = servers_ids[i]
                    func_ids = function_ids[i]

                sumlamb = sum(lambs)

                data_list = get_data_list(clusterings[0], input[i])
                # function_ids is a list of set of entry-point functions receiving workload
                inp = input[i]
                if vary_functions:
                    workloads = set_workload(clusterings[0], data_list, servers_ids[0], function_ids[0],
                                             lambs, input[i], is_heu=is_heu)
                else:
                    workloads = set_workload(clusterings[0], data_list, servers_ids[i], function_ids[i],
                                         lambs, input[i], is_heu=is_heu)
        else:
            if real_world:  # fix the topology (number of nodes), number of applications and vary the workload for gateway servers in real world
                clusts = copy.deepcopy(clusterings)
                topo = copy.deepcopy(clusterings[0].topology)
                lamb = lambs_var_app[i]
                lambs = lambs_var_app[i]  # the workload for each scale
                sumlamb = sum(lambs)
                servers_id = servers_ids
                func_ids = function_ids
                param = scale * i  # 5min, 10min, 15min, ..., 60min
                data_list = get_data_list(clusterings[0], input)

                # print(f'function_ids={function_ids}')
                # print(f'servers_ids={servers_ids}')
                # print(f'lambs[{i}]={lambs}')
                # function_ids is a list of set of entry-point functions receiving workload
                workloads = set_workload(clusterings[0], data_list, servers_ids, function_ids,
                                         lambs, input, is_heu=is_heu)

                # print(f'WORKLOADS')
                # for i in range(len(workloads)):
                #     print(workloads[i])
            else:
                if vary_topologies:  # vary the topology (number of nodes) and the workload
                    funct = input["function_names"]  # returns ["ns/f0", "ns/f1", "ns/f2", "ns/f3", "ns/f4" ...]
                    # remove ns/
                    func = get_cleaned_functions(funct)
                    functions = generate_functions(func)

                    if len(original_clusterings) < i + 1:
                        old_dynamic_clusters = original_clusterings[i - 1]
                        old_server_locations = old_dynamic_clusters.topology.server_locations
                        # for cases that we are not using topology generated before by other approaches
                        if not reunse_clusterings:

                            network = NetworkGenerator()
                            server_locations_aux, has_changed = network.append_servers_positions(i, old_servers_positions=old_server_locations, qty=scale_optional, server_locations=server_locations)
                            # print(f'++++++++++++++ server_locations={server_locations_aux}')
                            if has_changed:
                                dynamic_clustering = generate_dynamic_clustering(server_locations_aux,
                                                                                 old_servers=copy.deepcopy(
                                                                                     old_dynamic_clusters.topology.servers),
                                                                                 cores=cores, functions=old_dynamic_clusters.functions)
                                clusterings.append(copy.deepcopy(dynamic_clustering))
                                original_clusterings.append(dynamic_clustering)
                        else:
                            original_clusterings.append(clusterings[i])

                    clusterings[i].functions = functions
                    clusterings[i].topology.set_functions(functions)
                    clusts = copy.deepcopy(clusterings)
                    topology = clusterings[i].topology
                    topo = copy.deepcopy(topology)
                    lamb = scale * (i + 1)
                    lambs = np.full(len(lambs[0]), lamb)
                    lamb = lambs
                    servers_id = servers_ids[0]
                    func_ids = function_ids[0]
                    param = i+1  # it defines the complexity (workload x servers)
                    data_list = get_data_list(clusterings[i], input)
                    # in this case clusterings has multiple topologies and servers_ids
                    # a list of servers receiving workload for each topology
                    # NOTE check servers_ids[i]

                    workloads = set_workload(clusterings[i], data_list, servers_ids[0], function_ids[0],
                                             lambs, input, is_heu=is_heu)
                    sumlamb = sum(lambs)
                    # print(f'$$$$$$$$$$ workload={workloads}')

                else:  # fix the topology (number of nodes), number of applications and vary the workload
                    clusts = copy.deepcopy(clusterings)
                    topo = copy.deepcopy(clusterings[0].topology)
                    lamb = scale * (i+1)
                    if lamb == 0:
                        lamb = 1
                    lambs = np.full(len(lambs[0]), lamb)
                    lamb=lambs
                    servers_id = servers_ids[0]
                    func_ids = function_ids[0]
                    data_list = get_data_list(clusterings[0], input)
                    # clusterings[0] means only one topology, the same go to servers_ids[0] and function_ids[0]
                    workloads = set_workload(clusterings[0], data_list, servers_ids[0], function_ids[0],
                                             lambs, input, is_heu=is_heu)
                    # update_data(json, data_list, workloads, inp)
                    sumlamb = sum(lambs)
                    param = sum(lambs)
    update_data(json, data_list, workloads, inp, is_heu=is_heu)
    universal_param = param

    return sumlamb, universal_param, data_list, topo, lamb, servers_id, func_ids, clusts


def fill_heu_xu(output_matrix_lines, servers_ids, function_ids, lambs,
                input, sumlamb, max_delay, scale_optional=5, clusterings=None, reunse_clusterings=False, server_locations=None, cores=None, scale=10, qty_values=1, has_coldstart=True,
                has_decision_delay=True, lambs_var_app=None, fixed_lamb=True,  vary_topologies=True, var_apps_topology=False, vary_functions=False, real_world=False, hop_delays=None):

    heu_xu_possible_decision_delay = np.zeros(qty_values)
    delays_heu_xu = np.zeros((output_matrix_lines, 2))
    nodes_heu_xu = np.zeros((output_matrix_lines, 2))
    memory_heu_xu = np.zeros((output_matrix_lines, 2))
    network_delay_heu_xu = np.zeros((output_matrix_lines, 2))
    coldstart_delay_heu_xu = np.zeros((output_matrix_lines, 2))
    decision_delay_heu_xu = np.zeros((output_matrix_lines, 2))
    avg_network_delay_heu_xu = np.zeros((output_matrix_lines, 2))
    avg_decision_delay_heu_xu = np.zeros((output_matrix_lines, 2))
    test_net_delays = []
    real_net_delays=[]
    clusts = copy.deepcopy(clusterings)
    first_sum_lamb = 1
    for i in range(output_matrix_lines):
        sumlamb=0
        sumlamb, universal_param, data_list, topology, lamb, servers_id, func_ids, clusts = update_resources(copy.deepcopy(clusterings), copy.deepcopy(servers_ids),
                                                                                                             copy.deepcopy(function_ids),
                                                                                                             lambs, i, scale, sumlamb, input, scale_optional=scale_optional, reunse_clusterings=reunse_clusterings,
                                                                                                             original_clusterings=clusterings, server_locations=server_locations, cores=cores,
                                                                                                             lambs_var_app=lambs_var_app, fixed_lamb=fixed_lamb, vary_topologies=vary_topologies,
                                                                                                             is_heu=True, var_apps_and_topology=var_apps_topology, vary_functions=vary_functions, real_world=real_world, hop_delays=hop_delays)
        if i == 0:
            first_sum_lamb = max(first_sum_lamb, sumlamb)

        print(f'HEU sumlamb={sumlamb}')
        total_nodes_used = []
        memory_used = []
        cores_used = []
        total_cold_starts = []
        total_network_delay = []
        total_placement_time = []
        data_list_temp = copy.deepcopy(data_list)
        cluster_reqs = np.zeros(len(data_list), dtype=int)
        j = 0
        # print(f'@@@@@@@@@@ servers_id={servers_id}')
        total_decision_time = 0
        for server_id in servers_id:

            c_id = get_initial_cluster_by_id(topology.initial_clusters, server_id).id
            current_cluster = topology.current_clusters[c_id]
            servers = current_cluster.servers
            pos_server = get_position(servers, server_id)

            # print(f'@@@@@pos_server = get_position({[s.id for s in servers]}, {server_id})={pos_server}')
            data = Data()
            for item in data_list_temp:
                if item.cluster.id == c_id:
                    data = item
                break

            hide_functions_without_workload(data, func_ids)

            heu = HeuXu(data)
            parallel_scheduler = data.parallel_scheduler
            # new_lambs = data.workload_on_source_matrix
            # pos_init = cluster_reqs[c_id]
            # pos_last = cluster_reqs[c_id] + lamb[j]
            # print(f'@@@@@@@range({pos_init}, {pos_last})')
            # for req in range(pos_init, pos_last):
            for req in range(lamb[j]):
                # print(f'%%%%%%%%%%%%%%%%% REQ={req}')
                x, y, z, w, list_cfj, cfj, decision_delay = heu.heuristic_placement(req, pos_server, parallel_scheduler)

                network_delay = heu.object_function_heu(req)
                test_net_delays.append(network_delay)
                total_decision_time = total_decision_time + decision_delay
            # cluster_reqs[c_id] = pos_last  # the next set of requests(lamb)
            # incoming to the same cluster must be set from position pos_last
            j = j + 1
            _, coldstart, network_delay = heu.object_function_global_results()
            real_net_delays.append(network_delay)
            # print(f'########### data.function_memories[f]={data.function_memories}')
            total_nodes, memory, cores = heu.resource_usage()
            total_cold_starts.append(coldstart)
            total_network_delay.append(network_delay)
            total_nodes_used.append(total_nodes)
            memory_used.append(memory)
            cores_used.append(cores)
        heu_xu_possible_decision_delay[0] = total_decision_time

        for k in range(1, qty_values):
            data_list_temp = copy.deepcopy(data_list)
            total_decision_time = 0
            cluster_reqs = np.zeros(len(data_list), dtype=int)
            j = 0
            for server_id in servers_id:
                c_id = get_initial_cluster_by_id(topology.initial_clusters, server_id).id
                current_cluster = topology.current_clusters[c_id]
                servers = current_cluster.servers
                pos_server = get_position(servers, server_id)
                data = Data()
                for item in data_list_temp:
                    if item.cluster.id == c_id:
                        data = item
                    break
                heu = HeuXu(data)
                parallel_scheduler = data.parallel_scheduler
                # new_lambs = data.workload_on_source_matrix
                # pos_init = cluster_reqs[c_id]
                # pos_last = cluster_reqs[c_id] + lamb[j]

                # for req in range(pos_init, pos_last):
                for req in range(lamb[j]):
                    x, y, z, w, list_cfj, cfj, decision_delay = heu.heuristic_placement(req, pos_server,
                                                                                        parallel_scheduler)
                    total_decision_time = total_decision_time + decision_delay
                # cluster_reqs[c_id] = pos_last  # the next set of requests(lamb)
                # incoming to the same cluster must be set from position pos_last
                j = j+1
            heu_xu_possible_decision_delay[k] = total_decision_time

        total_decision_time = np.mean(heu_xu_possible_decision_delay)
        placement_delay = round((total_decision_time) + 0.0, 0)
        total_nodes = sum(total_nodes_used)
        memory = sum(memory_used)
        cores = sum(cores_used)
        coldstart = max(total_cold_starts)
        network_delay = sum(total_network_delay)
        avg_network_delay = network_delay/sumlamb
        avg_placement_delay = round((total_decision_time) + 0.0, 0)/sumlamb
        total_delay_heu = fill_output(first_sum_lamb, sumlamb, coldstart, coldstart_delay_heu_xu, placement_delay, avg_placement_delay,
                                      decision_delay_heu_xu, avg_decision_delay_heu_xu, delays_heu_xu, i, memory,
                                      memory_heu_xu, network_delay, avg_network_delay, network_delay_heu_xu,
                                      avg_network_delay_heu_xu, nodes_heu_xu, total_nodes, universal_param,
                                      vary_topologies, has_coldstart=has_coldstart,
                                      has_decision_delay=has_decision_delay, real_world=real_world)

        if total_delay_heu > max_delay:
            max_delay = total_delay_heu
    response = app.response_class(
        response=json.dumps({
            "cpu_routing_rules": {},
            "cpu_allocations": {},
            "gpu_routing_rules": {},
            "gpu_allocations": {},
        }),
        status=200,
        mimetype='application/json'
    )
    # print(f'@@@@@@@@@@test_net_delays={test_net_delays}')
    # print(f'@@@@@@@@@@real_net_delays={real_net_delays}')
    return response, sumlamb, delays_heu_xu, memory_heu_xu, coldstart_delay_heu_xu, network_delay_heu_xu,\
        avg_network_delay_heu_xu, decision_delay_heu_xu, avg_decision_delay_heu_xu, nodes_heu_xu, max_delay, clusts


@app.route('/')
def serve():

    # # print("Request received")
    input = request.json

    # f = len(inp["function_names"])
    # nod = len(inp["node_names"])

    # Define the size of the PDF sheet
    pdf_sheet_width = 4.2  # Width in inches
    pdf_sheet_height = 3.0  # Height in inches

    # application = inp["app"]
    application = "syntetic"
    save_path = 'plots'

    network_delays = []
    avg_network_delays = []
    coldstart_delays = []
    memories = []
    decision_delays = []
    avg_decision_delays = []
    total_delays = []
    nodes = []
    labels = []
    uses_real_workload = False
    has_coldstart = True
    log_directory = 'logs'
    has_decision_delay = True
    cold_start_inclusion = '(with cold start)' if has_coldstart else '(no cold start)'

    print(f'Fixe Number of nodes and applications, and vary the workload')
# NOTE ******++++FIX NUMBER OF SERVERS AND APPLICATIONS AND VARY THE WORKLOAD +++++++++++++++++++++++++
    print(f'---------------------------------------------------------------')
    """
    inputs:
    output_matrix_lines defines the set of topologies, servers_ids sets, etc. TODO REVISE RESOURCE USAGE METHOD
    output_matrix_lines (number points to plot. e.g.: 10 points)
    clusterings (number of infrastructures. Each infrastructure includes Topology and in topology we have clusters.
    e.g.: clusters=[dynamic_clustering1, dynamic_clustering2,..., dynamic_clustering_n] )
    servers_ids (set of servers receiving requests for each topology. e.g.: servers_id=[[2,4,9,11], [2,4,9,11,1,0], ...)
    function_ids (set of entry-point function ids receiving requests for each set of application represente by a DAG
    e.g.: function_ids=[[0,1,5,7], [0,1,5,7, 15, 20], ...)
    lambs ( set of workload for each set of entry_point functions of each set of applications.
    e.g.: lamb=[[10,30,20,50], [10,30,20,50,40,5], ...] for instance we take only the lamb[o]
    """

    output_matrix_lines = 10
    clusterings = [param.dynamic_clustering] # 50 servers
    servers_ids = param.servers_ids
    function_ids = param.function_ids
    lambs = param.lambs
    hop_delays = param.hop_delays
    # print(f'@@@@@@lambs={lambs}')
    sumlamb = 0
    max_delay = 100
    scale = 100  # workload from 100 to 1000
    qty_values = 1
    has_coldstart = True
    has_decision_delay = True
    lambs_var_app = [param.lambs]
    fixed_lamb = False
    vary_topologies = False
    vary_apps_topologies = False
    vary_functions = False
    x_label = 'Workload'
    variation_param = 'varyWorkload(100_1000)'
    cores = 20000
    scale_optional = 0
    server_locations = None
    chart_param = '(varyWorkload100-1000)'
    log_subdirectory = 'var_workload100-1000(50servers20cores)'
    is_log_scale = False

    print(f'FIX THE NUMBER OF APPLICATIONS AND WORKLOAD, AND VARY THE NUMBER OF SERVERS - ONE APP ')
    # TODO revise fix the one application and vary the workload and number of servers
    #  ( application of 5 functions, start by 5 servers and 100 requests.
    #  Add 5 servers and 100 requests until reaching 50 servers)
    # NOTE ****** FIX THE NUMBER OF APPLICATIONS AND VARY WORKLOAD AND THE NUMBER OF SERVERS - ONE APP ****
    print(f'---------------------------------------------------------------')
    # output_matrix_lines = 10
    # clusterings = [param.dynamic_clustering]
    # servers_ids = param.servers_ids
    # function_ids = param.function_ids
    # lambs = param.lambs
    # # print(f'@@@@@@lambs={lambs}')
    # sumlamb = 0
    # max_delay = 100
    # scale = 100
    # scale_optional = 5  # for number of servers variation
    # qty_values = 1
    # has_coldstart = True
    # has_decision_delay = True
    # lambs_var_app = [param.lambs]
    # server_locations = param.server_italy_locations50
    # fixed_lamb = False
    # vary_topologies = True
    # vary_apps_topologies = False
    # cores = 20000
    # # server_locations = None
    # vary_functions = False
    # chart_param = '(varyServers5-50andWorkload100-1000)'
    # x_label = 'Complexity(servers x workload)'
    # variation_param = 'varyNumberServers5-50_and_Workload100-1000_one_app'
    # log_subdirectory = 'varyServers5-50andWorkload100-1000'
    # is_log_scale = False
    # cold_start_inclusion = '(with cold start)' if has_coldstart else '(no cold start)'
    # print(f'FIX THE NUMBER OF APPLICATIONS AND WORKLOAD, AND VARY THE NUMBER OF SERVERS - 10 APPS')

    print(f'FIX THE NETWORK TOPOLOGY, THE WORKLOAD AND VARY THE NUMBER OF FUNCTIONS '
          f'(ASSESS HOW THE SOLUTIONS PERFORM INCRESING THE COMPLEXITY OF THE APPLICATIONS )')
    # NOTE **** FIX THE NETWORK TOPOLOGY, THE WORKLOAD AND VARY NUMBER OF FUNCTIONS *******
    print(f'---------------------------------------------------------------')
    # output_matrix_lines = 6
    # clusterings = [param.dynamic_clustering]
    # servers_ids = param.servers_ids_var_f
    # function_ids = param.function_ids_var_f
    # lambs = param.lambs_var_f
    # # print(f'@@@@@@lambs={lambs}')
    # sumlamb = 0
    # max_delay = 100
    # scale = 1
    # qty_values = 10
    # has_coldstart = True
    # has_decision_delay = True
    # lambs_var_app = param.lambs_var_f
    # apps_number = 6
    # fixed_lamb = True
    # vary_topologies = False
    # vary_apps_topologies = False
    # cores = 20000
    # scale_optional = 0
    # server_locations = None
    # vary_functions = True
    # cold_start_inclusion = '(with cold start)' if has_coldstart else '(no cold start)'
    #
    # x_label = 'Number of functions'
    # variation_param = 'varyNumberFunctions_200users_SameLocation(5-10-15--30functions)'
    # chart_param ='(varyNumberFunctions5-30)'
    # log_subdirectory = 'vary_number_functions(workload500)'
    # is_log_scale = False


    print(f'FIX THE NETWORK TOPOLOGY, THE WORKLOAD FOR EACH APPLICATION AND VARY THE NUMBER OF APPLICATIONS')
    # NOTE **** FIX THE NETWORK TOPOLOGY, THE WORKLOAD FOR EACH APPLICATION AND VARY THE NUMBER OF APPLICATIONS*********
    print(f'---------------------------------------------------------------')
    # output_matrix_lines = 10
    # clusterings = [param.dynamic_clustering]
    # servers_ids = generate_random_server_ids(10, 20, fix_server_id=0)
    # function_ids = param.function_ids
    # lambs = param.lambs
    # # print(f'@@@@@@lambs={lambs}')
    # sumlamb = 0
    # max_delay = 100
    # scale = 1
    # qty_values = 1
    # has_coldstart = True
    # has_decision_delay = True
    # lambs_var_app = param.lambs
    # apps_number = 10
    # fixed_lamb = True
    # vary_topologies = False
    # vary_apps_topologies = False
    # cold_start_inclusion = '(with cold start)' if has_coldstart else '(no cold start)'
    # cores = 20000
    # scale_optional = 0
    # server_locations = None
    # vary_functions = False
    # chart_param = '(varyApps1-10-50servers)'
    #
    #
    # x_label = 'Number of Applications'
    # variation_param = 'varyNumberApplications_100users_50servers_singleStartServer'
    # chart_param = '(varyApps1-10-50servers)'
    # log_subdirectory = '50servers_20cores_app5f_100reqeachApp_VarApps1-10'
    # is_log_scale = False

    print(f'VARY THE NETWORK TOPOLOGY, VARY THE NUMBER OF APPLICATIONS AND FIX THE  WORKLOAD FOR EACH APPLICATION')
    # NOTE **** VARY THE NETWORK TOPOLOGY, VARY THE NUMBER OF APPLICATIONS AND FIX THE
    #  WORKLOAD FOR EACH APPLICATION *********
    print(f'---------------------------------------------------------------')

    # output_matrix_lines = 10
    # clusterings = [param.dynamic_clustering]
    # servers_ids = param.servers_ids  #  10 apps, 10 initial servers, score=10
    # save_list_of_lists(log_directory, servers_ids, 'servers_ids.txt')  # use to process in parts to prevent desconections
    # function_ids = param.function_ids
    # lambs = param.lambs
    # # print(f'@@@@@@lambs={lambs}')
    # sumlamb = 0
    # max_delay = 100
    # scale = 10
    # qty_values = 1
    # has_coldstart = True
    # has_decision_delay = True
    # lambs_var_app = param.lambs
    # apps_number = 10
    # fixed_lamb = True
    # vary_apps_topologies = True
    # vary_topologies = False
    # hop_delays = param.hop_delays
    # cores = 20000
    # scale_optional = 0
    # server_locations = param.server_italy_locations
    # vary_functions = False
    # chart_param = '(varyApps1-10-varytTopology10-100servers)'
    # log_subdirectory = 'varyApps1-10-varytTopology10-100servers'
    # cold_start_inclusion = '(with cold start)' if has_coldstart else '(no cold start)'
    #
    #
    #
    # x_label = 'Complexity (Servers x Applications)'
    # variation_param = 'varyNumberApplications1-10AndNumberOfServers-10-100(200req each app)'

    print(f'FIX THE NETWORK TOPOLOGY, THE NUMBER OF APPLICATIONS AND VARY THE WORKLOAD TIME'
          f' IN SCALE RANGE (real world workload)')
    # NOTE: ***** FIX THE NETWORK TOPOLOGY, THE NUMBER OF APPLICATIONS AND VARY THE WORKLOAD TIME IN SCALE RANGE
    #  (real world workload) *********************
    print(f'---------------------------------------------------------------')
    # output_matrix_lines = 12
    # clusterings = [param.dynamic_clustering]
    # servers_ids = param.servers_id
    # # save_list_of_lists(log_directory, servers_ids,'servers_ids.txt')  # use to process in parts to prevent desconections
    # function_ids = param.function_ids
    # lambs = param.real_workload_per_server
    # # print(f'@@@@@@lambs={lambs}')
    # sumlamb = 0
    #
    # max_delay = 100
    # scale = 5  # 5min
    # qty_values = 10
    # has_coldstart = True
    # has_decision_delay = True
    # lambs_var_app = param.real_workload_per_server
    # fixed_lamb = False
    # vary_topologies = False
    # uses_real_workload = True
    # vary_apps_topologies = False
    # cold_start_inclusion = '(with cold start)' if has_coldstart else '(no cold start)'
    #
    # hop_delays = None
    # cores = 100000  # No need
    # scale_optional = 0
    # server_locations = param.server_locations  # No need
    # vary_functions = False
    # chart_param = '(real_world_varyWorkload_Topology50servers)'
    # log_subdirectory = 'new_realworld26012025'
    # x_label = 'Time (min)'
    # variation_param = 'real_workload_varyTime_and_Workload-all_approaches(timeout_30min)'
    # is_log_scale = True


    # NOTE ALL THE APPROACHES HERE
    print(f'START HERE')
    response, sumlamb, delays_vsvbp, memory_vsvbp, coldstart_delay_vsvbp, network_delay_vsvbp, avg_network_delay_vsvbp, decision_delay_vsvbp, avg_decision_delay_vsvbp, nodes_vsvbp, \
        max_delay, clusts = fill_formal_optimization(output_matrix_lines, servers_ids, function_ids, lambs, input, sumlamb,
                                                max_delay, scale_optional=scale_optional, clusterings=clusterings, reunse_clusterings=False, server_locations=server_locations, cores=cores, scale=scale,
                                 qty_values=qty_values, has_coldstart=has_coldstart, has_decision_delay=has_decision_delay, lambs_var_app=lambs_var_app,
                                 fixed_lamb=fixed_lamb, vary_topologies=vary_topologies, var_apps_topology=vary_apps_topologies, vary_functions=vary_functions, real_world=uses_real_workload, hop_delays=hop_delays, solver_type='VSVBP')

    network_delays.append(network_delay_vsvbp), avg_network_delays.append(avg_network_delay_vsvbp), memories.append(memory_vsvbp), coldstart_delays.append(coldstart_delay_vsvbp)
    decision_delays.append(decision_delay_vsvbp), avg_decision_delays.append(avg_decision_delay_vsvbp), total_delays.append(delays_vsvbp), nodes.append(nodes_vsvbp), labels.append('VSVBP')

    save_dynamic_clusters(log_directory, clusterings, 's_ids.txt', 'cores.txt', 'memories.txt', 'server_locations.txt', 'f_ids.txt', 'f_names.txt')
    save_matrixes(log_directory, chart_param, network_delay_vsvbp, 'network_delay_vsvbp.txt', avg_network_delay_vsvbp,
                  'avg_network_delay_vsvbp.txt',
                  memory_vsvbp, 'memory_vsvbp.txt', coldstart_delay_vsvbp, 'coldstart_delay_vsvbp.txt',
                  decision_delay_vsvbp, 'decision_delay_vsvbp.txt', avg_decision_delay_vsvbp,
                  'avg_decision_delay_vsvbp.txt', delays_vsvbp, 'delays_vsvbp.txt', nodes_vsvbp, 'nodes_vsvbp.txt')

    response, sumlamb, delays_neptune, memory_neptune, coldstart_delay_neptune, network_delay_neptune, avg_network_delay_neptune, decision_delay_neptune, avg_decision_delay_neptune, nodes_neptune, \
        max_delay, _ = fill_formal_optimization(output_matrix_lines, servers_ids, function_ids, lambs, input, sumlamb,
                                                max_delay, scale_optional=scale_optional, clusterings=clusts, reunse_clusterings=True, server_locations=server_locations, cores=cores, scale=scale, qty_values=qty_values,
                                                has_coldstart=has_coldstart, has_decision_delay=has_decision_delay,
                                                lambs_var_app=lambs_var_app, fixed_lamb=fixed_lamb,
                                                vary_topologies=vary_topologies, var_apps_topology=vary_apps_topologies, vary_functions=vary_functions, real_world=uses_real_workload, hop_delays=hop_delays, solver_type='NeptuneMinDelayAndUtilization')

    # servers = [(s.id, s.cores, s.memory) for s in clusts[len(clusts)-1].topology.servers]
    # print(f'Total Topo-NEP={len(clusts)}')
    # print(f'Servers-NEP={servers}')

    network_delays.append(network_delay_neptune), avg_network_delays.append(avg_network_delay_neptune), \
        memories.append(memory_neptune), coldstart_delays.append(coldstart_delay_neptune)
    decision_delays.append(decision_delay_neptune), avg_decision_delays.append(avg_decision_delay_neptune),total_delays.append(delays_neptune), nodes.append(
        nodes_neptune), labels.append('NEPTUNE')

    save_matrixes(log_directory,  chart_param, network_delay_neptune, 'network_delay_neptune.txt', avg_network_delay_neptune,
                  'avg_network_delay_neptune.txt',
                  memory_neptune, 'memory_neptune.txt', coldstart_delay_neptune, 'coldstart_delay_neptune.txt',
                  decision_delay_neptune, 'decision_delay_neptune.txt', avg_decision_delay_neptune,
                  'avg_decision_delay_neptune.txt', delays_neptune, 'delays_neptune.txt', nodes_neptune,
                  'nodes_neptune.txt')


    response, sumlamb, delays_cr_eua, memory_cr_eua, coldstart_delay_cr_eua, network_delay_cr_eua, avg_network_delay_cr_eua, decision_delay_cr_eua, avg_decision_delay_cr_eua, nodes_cr_eua, \
        max_delay, _ = fill_formal_optimization(output_matrix_lines, servers_ids, function_ids, lambs, input,
                                                sumlamb, max_delay, scale_optional=scale_optional, clusterings=clusts, reunse_clusterings=True, server_locations=server_locations, cores=cores, scale=scale,
                                                qty_values=qty_values, has_coldstart=has_coldstart,
                                                has_decision_delay=has_decision_delay, lambs_var_app=lambs_var_app,
                                                fixed_lamb=fixed_lamb, vary_topologies=vary_topologies, var_apps_topology=vary_apps_topologies, vary_functions=vary_functions, real_world=uses_real_workload, hop_delays=hop_delays,  solver_type='CriticalityHeuristic')

    network_delays.append(network_delay_cr_eua), avg_network_delays.append(avg_network_delay_cr_eua), memories.append(memory_cr_eua), coldstart_delays.append(coldstart_delay_cr_eua)
    decision_delays.append(decision_delay_cr_eua), avg_decision_delays.append(avg_decision_delay_cr_eua), total_delays.append(delays_cr_eua), nodes.append(nodes_cr_eua), labels.append('CR-EUA')

    save_matrixes(log_directory,  chart_param, network_delay_cr_eua, 'network_delay_cr_eua.txt', avg_network_delay_cr_eua,
                  'avg_network_delay_cr_eua.txt',
                  memory_cr_eua, 'memory_cr_eua.txt', coldstart_delay_cr_eua, 'coldstart_delay_cr_eua.txt',
                  decision_delay_cr_eua, 'decision_delay_cr_eua.txt', avg_decision_delay_cr_eua,
                  'avg_decision_delay_cr_eua.txt', delays_cr_eua, 'delays_cr_eua.txt', nodes_cr_eua, 'nodes_cr_eua.txt')


    response, sumlamb, delays_mcf, memory_mcf, coldstart_delay_mcf, network_delay_mcf, avg_network_delay_mcf, decision_delay_mcf, avg_decision_delay_mcf, nodes_mcf, \
        max_delay, _ = fill_formal_optimization(output_matrix_lines, servers_ids, function_ids, lambs, input,
                                                sumlamb, max_delay, scale_optional=scale_optional, clusterings=clusts, reunse_clusterings=True, server_locations=server_locations, cores=cores, scale=scale,
                                                qty_values=qty_values, has_coldstart=has_coldstart,
                                                has_decision_delay=has_decision_delay, lambs_var_app=lambs_var_app,
                                                fixed_lamb=fixed_lamb, vary_topologies=vary_topologies, var_apps_topology=vary_apps_topologies, vary_functions=vary_functions,  real_world=uses_real_workload, hop_delays=hop_delays, solver_type='MCF')

    network_delays.append(network_delay_mcf), avg_network_delays.append(avg_network_delay_mcf), memories.append(memory_mcf), coldstart_delays.append(coldstart_delay_mcf)
    decision_delays.append(decision_delay_mcf), avg_decision_delays.append(avg_decision_delay_mcf), total_delays.append(delays_mcf), nodes.append(nodes_mcf), labels.append('MCF')

    save_matrixes(log_directory,  chart_param, network_delay_mcf, 'network_delay_mcf.txt', avg_network_delay_mcf,
                  'avg_network_delay_mcf.txt',
                  memory_mcf, 'memory_mcf.txt', coldstart_delay_mcf, 'coldstart_delay_mcf.txt',
                  decision_delay_mcf, 'decision_delay_mcf.txt', avg_decision_delay_mcf,
                  'avg_decision_delay_mcf.txt', delays_mcf, 'delays_mcf.txt', nodes_mcf, 'nodes_mcf.txt')



    response, sumlamb, delays_pluto, memory_pluto, coldstart_delay_pluto, network_delay_pluto, avg_network_delay_pluto, decision_delay_pluto, \
        avg_decision_delay_pluto, nodes_pluto, max_delay, _ = \
        fill_pluto(output_matrix_lines, servers_ids, function_ids, lambs, input, sumlamb,
                                                max_delay, scale_optional=scale_optional, clusterings=clusterings, reunse_clusterings=True, server_locations=server_locations, cores=cores, scale=scale, qty_values=qty_values,
                                                has_coldstart=has_coldstart, has_decision_delay=has_decision_delay,
                                                lambs_var_app=lambs_var_app, fixed_lamb=fixed_lamb,
                                                vary_topologies=vary_topologies, var_apps_topology=vary_apps_topologies, vary_functions=vary_functions, real_world=uses_real_workload, hop_delays=hop_delays)

    network_delays.append(network_delay_pluto), avg_network_delays.append(avg_network_delay_pluto), memories.append(
        memory_pluto), coldstart_delays.append(coldstart_delay_pluto)
    decision_delays.append(decision_delay_pluto), avg_decision_delays.append(
        avg_decision_delay_pluto), total_delays.append(delays_pluto), nodes.append(nodes_pluto), labels.append('PLUTO')

    save_matrixes(log_directory, chart_param, network_delay_pluto, 'network_delay_pluto.txt', avg_network_delay_pluto, 'avg_network_delay_pluto.txt',
                  memory_pluto, 'memory_pluto.txt', coldstart_delay_pluto, 'coldstart_delay_pluto.txt',
                  decision_delay_pluto, 'decision_delay_pluto.txt', avg_decision_delay_pluto,
                  'avg_decision_delay_pluto.txt', delays_pluto, 'delays_pluto.txt', nodes_pluto, 'nodes_pluto.txt')


    print(f'-------------------------------------------------------------')

    response, sumlamb, delays_heu_xu, memory_heu_xu, coldstart_delay_heu_xu, network_delay_heu_xu, \
        avg_network_delay_heu_xu, decision_delay_heu_xu, avg_decision_delay_heu_xu, nodes_heu_xu, max_delay, _ = fill_heu_xu(
        output_matrix_lines, servers_ids, function_ids, lambs, input, sumlamb, max_delay,scale_optional=scale_optional,
        clusterings=clusts, reunse_clusterings=True, server_locations=server_locations, cores=cores, scale=scale, qty_values=qty_values,
        has_coldstart=has_coldstart, has_decision_delay=has_decision_delay,
        lambs_var_app=lambs_var_app, fixed_lamb=fixed_lamb, vary_topologies=vary_topologies,
        var_apps_topology=vary_apps_topologies, vary_functions=vary_functions, real_world=uses_real_workload, hop_delays=hop_delays)

    network_delays.append(network_delay_heu_xu), avg_network_delays.append(avg_network_delay_heu_xu), memories.append(
        memory_heu_xu), coldstart_delays.append(coldstart_delay_heu_xu)
    decision_delays.append(decision_delay_heu_xu), avg_decision_delays.append(
        avg_decision_delay_heu_xu), total_delays.append(delays_heu_xu), nodes.append(nodes_heu_xu), labels.append('HEU')

    save_matrixes(log_directory,  chart_param, network_delay_heu_xu, 'network_delay_heu_xu.txt', avg_network_delay_heu_xu,
                  'avg_network_delay_heu_xu.txt',
                  memory_heu_xu, 'memory_heu_xu.txt', coldstart_delay_heu_xu, 'coldstart_delay_heu_xu.txt',
                  decision_delay_heu_xu, 'decision_delay_heu_xu.txt', avg_decision_delay_heu_xu,
                  'avg_decision_delay_heu_xu.txt', delays_heu_xu, 'delays_heu_xu.txt', nodes_heu_xu, 'nodes_heu_xu.txt')

    print()
    # NOTE RESTORE RESULTS STORED ON APPROACH FILES
    # NOTE: MANUALLY CALCULATE THE AVGs
    print(f'STARTS HERE ')
    # response, network_delay_vsvbp, avg_network_delay_vsvbp, memory_vsvbp, coldstart_delay_vsvbp, decision_delay_vsvbp, \
    #     avg_decision_delay_vsvbp, delays_vsvbp, nodes_vsvbp = \
    #     get_matrixes(app, log_directory, log_subdirectory, chart_param, 'network_delay_vsvbp.txt', 'avg_network_delay_vsvbp.txt', 'memory_vsvbp.txt',
    #                  'coldstart_delay_vsvbp.txt', 'decision_delay_vsvbp.txt', 'avg_decision_delay_vsvbp.txt',
    #                  'delays_vsvbp.txt', 'nodes_vsvbp.txt')
    #
    # print(f'Restore results stored on NEPTUNE files ')
    # response, network_delay_neptune, avg_network_delay_neptune, memory_neptune, coldstart_delay_neptune, decision_delay_neptune, \
    #     avg_decision_delay_neptune, delays_neptune, nodes_neptune = \
    #     get_matrixes(app, log_directory, log_subdirectory, chart_param, 'network_delay_neptune.txt', 'avg_network_delay_neptune.txt',
    #                  'memory_neptune.txt', 'coldstart_delay_neptune.txt', 'decision_delay_neptune.txt', 'avg_decision_delay_neptune.txt',
    #                  'delays_neptune.txt', 'nodes_neptune.txt')
    #
    # print(f'Restore results stored on CR-EUA files ')
    # response, network_delay_cr_eua, avg_network_delay_cr_eua, memory_cr_eua, coldstart_delay_cr_eua, decision_delay_cr_eua, \
    #     avg_decision_delay_cr_eua, delays_cr_eua, nodes_cr_eua = \
    #     get_matrixes(app, log_directory, log_subdirectory, chart_param, 'network_delay_cr_eua.txt', 'avg_network_delay_cr_eua.txt',
    #                  'memory_cr_eua.txt',
    #                  'coldstart_delay_cr_eua.txt', 'decision_delay_cr_eua.txt', 'avg_decision_delay_cr_eua.txt',
    #                  'delays_cr_eua.txt', 'nodes_cr_eua.txt')
    #
    # print(f'Restore results stored on MCF files ')
    # response, network_delay_mcf, avg_network_delay_mcf, memory_mcf, coldstart_delay_mcf, decision_delay_mcf, \
    #     avg_decision_delay_mcf, delays_mcf, nodes_mcf = \
    #     get_matrixes(app, log_directory, log_subdirectory, chart_param, 'network_delay_mcf.txt', 'avg_network_delay_mcf.txt', 'memory_mcf.txt',
    #                  'coldstart_delay_mcf.txt', 'decision_delay_mcf.txt', 'avg_decision_delay_mcf.txt',
    #                  'delays_mcf.txt', 'nodes_mcf.txt')
    #
    # print(f'Restore results stored on PLUTO files')
    # response, network_delay_pluto, avg_network_delay_pluto, memory_pluto, coldstart_delay_pluto, decision_delay_pluto, \
    #     avg_decision_delay_pluto, delays_pluto, nodes_pluto = \
    #     get_matrixes(app, log_directory, log_subdirectory, chart_param, 'network_delay_pluto.txt',
    #                  'avg_network_delay_pluto.txt', 'memory_pluto.txt',
    #                  'coldstart_delay_pluto.txt', 'decision_delay_pluto.txt', 'avg_decision_delay_pluto.txt',
    #                  'delays_pluto.txt', 'nodes_pluto.txt')
    #
    # print(f'Restore results stored on HEU files')
    # response, network_delay_heu_xu, avg_network_delay_heu_xu, memory_heu_xu, coldstart_delay_heu_xu, decision_delay_heu_xu, \
    #     avg_decision_delay_heu_xu, delays_heu_xu, nodes_heu_xu = \
    #     get_matrixes(app, log_directory, log_subdirectory, chart_param, 'network_delay_heu_xu.txt', 'avg_network_delay_heu_xu.txt',
    #                  'memory_heu_xu.txt',
    #                  'coldstart_delay_heu_xu.txt', 'decision_delay_heu_xu.txt', 'avg_decision_delay_heu_xu.txt',
    #                  'delays_heu_xu.txt', 'nodes_heu_xu.txt')
    # print("BEFORE")
    # for i in range(len(delays_vsvbp)):
    #     print(delays_vsvbp[i])
    # print("AFTER")
    # delays_vsvbp = compute_total_delays(network_delay_vsvbp, coldstart_delay_vsvbp, decision_delay_vsvbp)
    # for i in range(len(delays_vsvbp)):
    #     print(delays_vsvbp[i])
    # network_delays.append(network_delay_vsvbp), avg_network_delays.append(avg_network_delay_vsvbp), memories.append(memory_vsvbp), coldstart_delays.append(coldstart_delay_vsvbp)
    # decision_delays.append(decision_delay_vsvbp), avg_decision_delays.append(avg_decision_delay_vsvbp), total_delays.append(delays_vsvbp), nodes.append(nodes_vsvbp), labels.append('VSVBP')
    #
    # delays_neptune = compute_total_delays(network_delay_neptune, coldstart_delay_neptune, decision_delay_neptune)
    # network_delays.append(network_delay_neptune), avg_network_delays.append(avg_network_delay_neptune), \
    #     memories.append(memory_neptune), coldstart_delays.append(coldstart_delay_neptune)
    # decision_delays.append(decision_delay_neptune), avg_decision_delays.append(avg_decision_delay_neptune),total_delays.append(delays_neptune), nodes.append(
    #     nodes_neptune), labels.append('NEPTUNE')
    #
    # delays_cr_eua = compute_total_delays(network_delay_cr_eua, coldstart_delay_cr_eua, decision_delay_cr_eua)
    # network_delays.append(network_delay_cr_eua), avg_network_delays.append(avg_network_delay_cr_eua), memories.append(memory_cr_eua), coldstart_delays.append(coldstart_delay_cr_eua)
    # decision_delays.append(decision_delay_cr_eua), avg_decision_delays.append(avg_decision_delay_cr_eua), total_delays.append(delays_cr_eua), nodes.append(nodes_cr_eua), labels.append('CR-EUA')
    #
    # delays_mcf = compute_total_delays(network_delay_mcf, coldstart_delay_mcf, decision_delay_mcf)
    # network_delays.append(network_delay_mcf), avg_network_delays.append(avg_network_delay_mcf), memories.append(memory_mcf), coldstart_delays.append(coldstart_delay_mcf)
    # decision_delays.append(decision_delay_mcf), avg_decision_delays.append(avg_decision_delay_mcf), total_delays.append(delays_mcf), nodes.append(nodes_mcf), labels.append('MCF')
    #
    # delays_pluto = compute_total_delays(network_delay_pluto, coldstart_delay_pluto, decision_delay_pluto)
    # network_delays.append(network_delay_pluto), avg_network_delays.append(avg_network_delay_pluto), memories.append(
    #     memory_pluto), coldstart_delays.append(coldstart_delay_pluto)
    # decision_delays.append(decision_delay_pluto), avg_decision_delays.append(
    #     avg_decision_delay_pluto), total_delays.append(delays_pluto), nodes.append(nodes_pluto), labels.append('PLUTO')
    #
    # delays_heu_xu = compute_total_delays(network_delay_heu_xu, coldstart_delay_heu_xu, decision_delay_heu_xu)
    # network_delays.append(network_delay_heu_xu), avg_network_delays.append(avg_network_delay_heu_xu), memories.append(
    #     memory_heu_xu), coldstart_delays.append(coldstart_delay_heu_xu)
    # decision_delays.append(decision_delay_heu_xu), avg_decision_delays.append(
    #     avg_decision_delay_heu_xu), total_delays.append(delays_heu_xu), nodes.append(nodes_heu_xu), labels.append('HEU')
    print(f'-------------------------------------------------------------')

    # # servers = [(s.id, s.cores, s.memory) for s in clusts1[len(clusts1) - 1].topology.servers]
    # # print(f'Total Topo-HEU={len(clusts1)}')
    # # print(f'Servers-HEU={servers}')
    # #
    # # print(f'-------------------------------------------------------------')
    # #
    # # print(f' Memory_Pluto={memory_pluto}')
    # # print(f' Memory_Heu={memory_heu_xu}')
    # #
    # # # Create subplots with 1 row and 3 columns
    # #
    # #
    # # # # NOTE VALID ...........................................................
    # # # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10.8, 2.8))
    # # #
    # # # # Plot network delay data
    # # # axes[0].plot(network_delay_neptune[:, 0], network_delay_neptune[:, 1], label='NEPTUNE', color='blue', marker='*',
    # # #              linestyle='--')
    # # # axes[0].plot(network_delay_heu_xu[:, 0], network_delay_heu_xu[:, 1], label='HEU', color='black', marker='x',
    # # #              linestyle='--')
    # # # axes[0].plot(network_delay_pluto[:, 0], network_delay_pluto[:, 1], label='PLUTO', color='green', marker='o',
    # # #              linestyle='-')
    # # # axes[0].set_xlabel(x_label)
    # # # axes[0].set_ylabel('Network delay(ms)')
    # # # axes[0].legend()
    # # # description = "(a) Network delay"
    # # # axes[0].text(0.5, -0.26, description, ha='center', va='center', transform=axes[0].transAxes, fontsize=12)
    # # #
    # # # # Plot cold start data
    # # # axes[1].plot(coldstart_delay_neptune[:, 0], coldstart_delay_neptune[:, 1], label='NEPTUNE', color='blue', marker='*', linestyle='--')
    # # # axes[1].plot(coldstart_delay_heu_xu[:, 0], coldstart_delay_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    # # # axes[1].plot(coldstart_delay_pluto[:, 0], coldstart_delay_pluto[:, 1], label='PLUTO', color='green', marker='o',
    # # #              linestyle='-')
    # # # axes[1].set_xlabel(x_label)
    # # # axes[1].set_ylabel('Coldstart delay(ms)')
    # # # axes[1].legend()
    # # # description = "(b) Coldstart delay"
    # # # axes[1].text(0.5, -0.26, description, ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
    # # #
    # # # # Plot decision delay data
    # # # axes[2].plot(decision_delay_neptune[:, 0], decision_delay_neptune[:, 1], label='NEPTUNE', color='blue', marker='*', linestyle='-')
    # # # axes[2].plot(decision_delay_heu_xu[:, 0], decision_delay_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    # # # axes[2].plot(decision_delay_pluto[:, 0], decision_delay_pluto[:, 1], label='PLUTO', color='green', marker='o',
    # # #              linestyle='-')
    # # # axes[2].set_xlabel(x_label)
    # # # axes[2].set_ylabel('Decision delay(ms)')
    # # # axes[2].legend()
    # # # description = "(c) Decision delay"
    # # # axes[2].text(0.5, -0.26, description, ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
    # # # plt.tight_layout()
    # # # plt.subplots_adjust(bottom=0.25)
    # # # if not os.path.exists(save_path):
    # # #     os.makedirs(save_path)
    # # # file_name = f'plot_{application}-10nod200req_{variation_param}-net-cold-decision.pdf'
    # # # plt.savefig(os.path.join(save_path, file_name), format='pdf')
    # # #
    # # # plt.clf()  # clear the current graph
    # # #
    # # # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.4, 3.0))
    # # # axes[0].plot(delays_neptune[:, 0], delays_neptune[:, 1], label='NEPTUNE', color='blue', marker='*', linestyle='--')
    # # # axes[0].plot(delays_heu_xu[:, 0], delays_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    # # # axes[0].plot(delays_pluto[:, 0], delays_pluto[:, 1], label='PLUTO', color='green', marker='o', linestyle='-')
    # # # axes[0].set_xlabel(x_label)
    # # # axes[0].set_ylabel('Total delay(ms)')
    # # # axes[0].legend()
    # # # description = "(d) Total delay"
    # # # axes[0].text(0.5, -0.26, description, ha='center', va='center', transform=axes[0].transAxes, fontsize=12)
    # # #
    # # # # Plot memory consumption
    # # # axes[1].plot(memory_neptune[:, 0], memory_neptune[:, 1], label='NEPTUNE', color='blue', marker='*',
    # # #              linestyle='--')
    # # # axes[1].plot(memory_heu_xu[:, 0], memory_heu_xu[:, 1], label='HEU', color='black', marker='x',
    # # #              linestyle='--')
    # # # axes[1].plot(memory_pluto[:, 0], memory_pluto[:, 1], label='PLUTO', color='green', marker='o',
    # # #              linestyle='-')
    # # # axes[1].set_xlabel(x_label)
    # # # axes[1].set_ylabel('Memory consumption(MB)')
    # # # axes[1].legend()
    # # # description = "(e) Memory consumption"
    # # # axes[1].text(0.5, -0.26, description, ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
    # # # # Adjust layout to prevent overlapping
    # # # plt.tight_layout()
    # # # plt.subplots_adjust(bottom=0.25)
    # # # if not os.path.exists(save_path):
    # # #     os.makedirs(save_path)
    # # # file_name = f'plot_{application}-10nod200req_{variation_param}-total-delay-memory.pdf'
    # # # plt.savefig(os.path.join(save_path, file_name), format='pdf')
    # # #
    # # # plt.clf()  # clear the current graph
    # # #
    # # # # --------------------- SEPARATED FIGURES --------------------------------
    # # plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # # Plot network delay data
    #
    # markers = ['o', 'x', '+', 's',  'v', 'P']
    # linstyles = ['-', (0, (1, 1)), '-.','--',  ':', (0, (5, 1))]
    # colors = ['green', 'black', 'red', 'blue', 'purple', 'orange']
    plt.clf()  # clear the current graph
    markers = ['+', 's', 'v', 'P', 'o', 'x']
    linstyles = ['-.', '--', ':', (0, (5, 1)), '-', (0, (1, 1))]
    colors = ['red', 'blue', 'purple', 'orange', 'green', 'black']

    # markers = ['o', 'x']
    # linstyles = ['-', (0, (1, 1))]
    # colors = ['green', 'black']

    plt.rcParams.update({
        'font.size': 18,  # Global font size
        'axes.titlesize': 18,  # Font size for plot titles
        'axes.labelsize': 18,  # Font size for axis labels
        'xtick.labelsize': 18,  # Font size for x-tick labels
        'ytick.labelsize': 18,  # Font size for y-tick labels
        'legend.fontsize': 12,  # Font size for legend
        'lines.markersize': 8,  # Size of markers
        'lines.linewidth': 2,  # Line width
    })

    for i in range(len(labels)):
        plt.plot(network_delays[i][:, 0], network_delays[i][:, 1], label=labels[i], color=colors[i], marker=markers[i],
                 linestyle=linstyles[i])

    plt.xlabel(x_label)
    plt.ylabel('Total Network delay(ms)')
    plt.legend()

    if is_log_scale:
        # plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.yscale('log')

    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'single-plot_{application}-{variation_param}-total-network-delay.pdf'
    plt.savefig(os.path.join(save_path, file_name), format='pdf')

    plt.clf()  # clear the current graph

    for i in range(len(labels)):
        plt.plot(avg_network_delays[i][:, 0], avg_network_delays[i][:, 1], label=labels[i], color=colors[i], marker=markers[i],
                 linestyle=linstyles[i])

    # plt.plot(network_delay_vsvbp[:, 0], network_delay_vsvbp[:, 1], label='VSVBP', color='red', marker='+',
    #              linestyle='--')
    #
    # plt.plot(network_delay_heu_xu[:, 0], network_delay_heu_xu[:, 1], label='HEU', color='black', marker='x',
    #              linestyle='--')
    # plt.plot(network_delay_pluto[:, 0], network_delay_pluto[:, 1], label='PLUTO', color='green', marker='o',
    #              linestyle='-')
    plt.xlabel(x_label)
    plt.ylabel('Average network delay(ms)')

    if is_log_scale:
        # plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.yscale('log')

    plt.legend()
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'single-plot_{application}-{variation_param}-avg-network-delay.pdf'
    plt.savefig(os.path.join(save_path, file_name), format='pdf')

    plt.clf()  # clear the current graph

    # Plot cold start data
    for i in range(len(labels)):
        plt.plot(coldstart_delays[i][:, 0], coldstart_delays[i][:, 1], label=labels[i], color=colors[i], marker=markers[i],
                 linestyle=linstyles[i])

    plt.xlabel(x_label)
    plt.ylabel('Coldstart delay(ms)')

    if is_log_scale:
        # plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.yscale('log')

    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'single-plot_{application}-{variation_param}-coldstart-delay.pdf'
    plt.savefig(os.path.join(save_path, file_name), format='pdf')

    plt.clf()  # clear the current graph
    #
    # Plot decision delay data
    for i in range(len(labels)):
        plt.plot(decision_delays[i][:, 0], decision_delays[i][:, 1], label=labels[i], color=colors[i],
                 marker=markers[i],
                 linestyle=linstyles[i])

    plt.xlabel(x_label)
    plt.ylabel('Placement delay(ms)')

    if is_log_scale:
        # plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'single-plot_{application}-{variation_param}-decision-delay.pdf'
    plt.savefig(os.path.join(save_path, file_name), format='pdf')

    plt.clf()  # clear the current graph

    for i in range(len(labels)):
        plt.plot(avg_decision_delays[i][:, 0], avg_decision_delays[i][:, 1], label=labels[i], color=colors[i],
                 marker=markers[i],
                 linestyle=linstyles[i])

    plt.xlabel(x_label)
    plt.ylabel('Average placement delay (ms)')

    if is_log_scale:
        # plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'single-plot_{application}-{variation_param}-avg_decision-delay.pdf'
    plt.savefig(os.path.join(save_path, file_name), format='pdf')

    plt.clf()  # clear the current graph

    for i in range(len(labels)):
        plt.plot(total_delays[i][:, 0], total_delays[i][:, 1], label=labels[i], color=colors[i],
                 marker=markers[i],
                 linestyle=linstyles[i])
    plt.xlabel(x_label)
    plt.ylabel('Total delay(ms)')

    if is_log_scale:
        # plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.yscale('log')

    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'single-plot_{application}-{variation_param}-total-delay.pdf'
    plt.savefig(os.path.join(save_path, file_name), format='pdf')

    plt.clf()  # clear the current graph
    #
    # Plot memory consumption
    for i in range(len(labels)):
        plt.plot(memories[i][:, 0], memories[i][:, 1], label=labels[i], color=colors[i],
                 marker=markers[i],
                 linestyle=linstyles[i])
    plt.xlabel(x_label)
    plt.ylabel('Memory consumption(MB)')

    # if is_log_scale:
    #     # plt.xscale('log')  # Set x-axis to logarithmic scale
    #     plt.yscale('log')

    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'single-plot_{application}-{variation_param}-memory-consumption.pdf'
    plt.savefig(os.path.join(save_path, file_name), format='pdf')
    plt.clf()  # clear the current graph

    st_file_name = f'statistic_{application}-{variation_param}-all.pdf'
    statistic(cold_start_inclusion, coldstart_delays,
              decision_delays, total_delays, memories, network_delays, avg_network_delays, nodes, st_file_name)

    # # # +++++++++++++++++++++++++ separate graphs coldstart and network delay++++++++++++++++++++++++++++++++
    # # plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # # if has_coldstart:
    # #     plt.plot(coldstart_delay_neptune[:, 0], coldstart_delay_neptune[:, 1], label='NEPTUNE(Cold start)', color='blue', marker='o', linestyle='-')
    # #     plt.plot(coldstart_delay_pluto[:, 0], coldstart_delay_pluto[:, 1], label='PLUTO(Cold start)', color='green', marker='*', linestyle='--')
    # #     plt.plot(coldstart_delay_heu_xu[:, 0], coldstart_delay_heu_xu[:, 1], label='HEU(Cold start)', color='black', marker='x', linestyle='--')
    # #
    # # plt.plot(network_delay_neptune[:, 0], network_delay_neptune[:, 1], label='NEPTUNE(Network delay)', color='blue',
    # #          marker='D', linestyle='-')
    # # plt.plot(network_delay_pluto[:, 0], network_delay_pluto[:, 1], label='PLUTO(Network delay)', color='green',
    # #          marker='s', linestyle='--')
    # # plt.plot(network_delay_heu_xu[:, 0], network_delay_heu_xu[:, 1], label='HEU(Network delay)', color='black',
    # #          marker='^', linestyle='--')
    # #
    # # if has_decision_delay:
    # #     plt.plot(decision_delay_neptune[:, 0], decision_delay_neptune[:, 1], label='NEPTUNE(Allocation delay)',
    # #              color='blue',
    # #              marker='P', linestyle='-')
    # #     plt.plot(decision_delay_pluto[:, 0], decision_delay_pluto[:, 1], label='PLUTO(Allocation delay)',
    # #              color='green',
    # #              marker='+', linestyle='--')
    # #     plt.plot(decision_delay_heu_xu[:, 0], decision_delay_heu_xu[:, 1], label='HEU(Allocation delay)', color='black',
    # #              marker='>', linestyle='--')
    # #
    # # plt.xlabel('Cores on node receiving direct calls (millicores)')
    # # plt.ylabel('Delay (ms)')
    # # plt.legend()
    # #
    # # # Customize legend and marker size
    # # plt.legend(fontsize='small')
    # # plt.setp(plt.gca().get_legend().get_texts(), fontsize='6')  # Adjust legend font size
    # # plt.setp(plt.gca().get_legend().get_title(), fontsize='6')  # Adjust legend title font size
    # # plt.setp(plt.gca().get_lines(), markersize=4)  # Adjust marker size
    # #
    # # if not os.path.exists(save_path):
    # #     os.makedirs(save_path)
    # # file_name = f'plot_{application}(10nod200req_varyCoresFistNode-delay{cold_start_inclusion}).pdf'
    # # plt.savefig(os.path.join(save_path, file_name), format='pdf')
    # #
    # # plt.clf()  # clear the current graph
    # #
    # #
    # # # +++++++++++++++++++++++++ Memory ++++++++++++++++++++++++++++++++
    # # plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # # # # Plot for matrix A with a specific color, marker, and format
    # # plt.plot(memory_neptune[:, 0], memory_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    # # plt.plot(memory_pluto[:, 0], memory_pluto[:, 1], label='PLUTO', color='green', marker='*', linestyle='--')
    # # plt.plot(memory_heu_xu[:, 0], memory_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    # # plt.xlabel('Cores on node receiving direct calls (millicores)')
    # # plt.ylabel('Total memory (MB)')
    # # plt.legend()
    # # if not os.path.exists(save_path):
    # #     os.makedirs(save_path)
    # # file_name = f'plot_{application}(10nod200req_varyCoresFistNode(memory)).pdf'
    # # plt.savefig(os.path.join(save_path, file_name), format='pdf')
    # #
    # # plt.clf()  # clear the current graph
    # #
    # # # +++++++++++++++++++++++++ TOTAL NODES USED ++++++++++++++++++++++++++++++++
    # # plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # # plt.plot(nodes_neptune[:, 0], nodes_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    # # plt.plot(nodes_pluto[:, 0], nodes_pluto[:, 1], label='PLUTO', color='green', marker='*', linestyle='--')
    # # plt.plot(nodes_heu_xu[:, 0], nodes_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    # # plt.xlabel('Cores on node receiving direct calls (millicores)')
    # # plt.ylabel('Total nodes used')
    # # plt.legend()
    # # if not os.path.exists(save_path):
    # #     os.makedirs(save_path)
    # # file_name = f'plot_{application}(10nod200req_varyCoresFistNode(total nodes).pdf'
    # #
    # # plt.savefig(os.path.join(save_path, file_name), format='pdf')

    # st_file_name = f'statistic(avg)_{application}' \
    #                f'(10nod200req_{variation_param}{cold_start_inclusion}.pdf'
    # statistic(cold_start_inclusion, coldstart_delay_pluto, coldstart_delay_heu_xu, coldstart_delay_neptune,
    #           decision_delay_pluto, decision_delay_heu_xu, decision_delay_neptune, delays_pluto, delays_heu_xu,
    #           delays_neptune, memory_pluto, memory_heu_xu, memory_neptune, network_delay_pluto, network_delay_heu_xu,
    #           network_delay_neptune, nodes_pluto, nodes_heu_xu, nodes_neptune, st_file_name)

    # # mean_delays = st.generate_by_count(st, [delays_neptune, delays_heu_xu, delays_pluto])
    # # mean_nodes = st.generate_by_count(st, [nodes_neptune, nodes_heu_xu, nodes_pluto])
    # # mean_memory = st.generate_by_count(st, [memory_neptune, memory_heu_xu, memory_pluto])
    # # mean_coldstart = st.generate_by_count(st, [coldstart_delay_neptune, coldstart_delay_heu_xu, coldstart_delay_pluto])
    # # mean_network_delay = st.generate_by_count(st, [network_delay_neptune, network_delay_heu_xu, network_delay_pluto])
    # # st.create_statistical_table(st, ['Delay', 'Nodes', 'Memory', 'Coldstart', 'Network Delay'], ['NEPTUNE', 'HEU', 'PLUTO'],
    # #                             [mean_delays, mean_nodes, mean_memory, mean_coldstart, mean_network_delay], 'plots', st_file_name)
    # plt.clf()  # clear the current graph
    #
    #
    # # # NOTE ****** VARY THE NUMBER OF NODES FROM 10 TO 50 ***************
    # # lamb = 200  # fix workload for entrypoint function
    # #
    # # # print("Request received")
    # # inp = request.json
    # #
    # # output_matrix_lines = 8
    # # topologies = [0, 1, 2, 3, 4, 5, 6, 7]
    # # response, sumlamb, delays_neptune, memory_neptune, nodes_neptune, max_delay = \
    # #     fill_neptune(output_matrix_lines, workload, inp, sumlamb, qty_values, fixed_lamb=lamb, scale=10, max_delay=0,
    # #                  topologies=topologies)
    # #
    # # sumlamb, delays_pluto, memory_pluto, nodes_pluto, max_delay = \
    # #     fill_pluto(output_matrix_lines, workload, inp, sumlamb, max_delay, fixed_lamb=lamb, scale=10, topologies=topologies)
    # #
    # # sumlamb, delays_heu_xu, memory_heu_xu, nodes_heu_xu, max_delay = \
    # #     fill_heu_xu(output_matrix_lines, workload, inp, sumlamb, max_delay, fixed_lamb=lamb, scale=10, topologies=topologies)
    # #
    # #
    # # # Plot for matrix A with a specific color, marker, and format
    # # plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # # plt.plot(delays_neptune[:, 0], delays_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    # # plt.plot(delays_pluto[:, 0], delays_pluto[:, 1], label='PLUTO', color='green', marker='*', linestyle='--')
    # # plt.plot(delays_heu_xu[:, 0], delays_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    # # plt.xlabel('Number of nodes')
    # # plt.ylabel('Total delay (ms)')
    # # plt.legend()
    # # if not os.path.exists(save_path):
    # #     os.makedirs(save_path)
    # # file_name = f'plot_{application}(200req_varyNode_Qty(delay).pdf'
    # # plt.savefig(os.path.join(save_path, file_name), format='pdf')
    # #
    # # plt.clf()  # clear the current graph
    # #
    # # # +++++++++++++++++++++++++ Memory ++++++++++++++++++++++++++++++++
    # # # # Plot for matrix A with a specific color, marker, and format
    # # plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # # plt.plot(memory_neptune[:, 0], memory_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    # # plt.plot(memory_pluto[:, 0], memory_pluto[:, 1], label='PLUTO', color='green', marker='*', linestyle='--')
    # # plt.plot(memory_heu_xu[:, 0], memory_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    # # plt.xlabel('Number of nodes')
    # # plt.ylabel('Total memory (MB)')
    # # plt.legend()
    # # if not os.path.exists(save_path):
    # #     os.makedirs(save_path)
    # # file_name = f'plot_{application}(200req_varyNode_Qty(memory).pdf'
    # # plt.savefig(os.path.join(save_path, file_name), format='pdf')
    # #
    # # plt.clf()  # clear the current graph
    # #
    # # # +++++++++++++++++++++++++ TOTAL NODES USED ++++++++++++++++++++++++++++++++
    # # plt.figure(figsize=(pdf_sheet_width, pdf_sheet_height))
    # # plt.plot(nodes_neptune[:, 0], nodes_neptune[:, 1], label='NEPTUNE', color='blue', marker='o', linestyle='-')
    # # plt.plot(nodes_pluto[:, 0], nodes_pluto[:, 1], label='PLUTO', color='green', marker='*', linestyle='--')
    # # plt.plot(nodes_heu_xu[:, 0], nodes_heu_xu[:, 1], label='HEU', color='black', marker='x', linestyle='--')
    # # plt.xlabel('Number of nodes')
    # # plt.ylabel('Total nodes used')
    # # plt.legend()
    # # if not os.path.exists(save_path):
    # #     os.makedirs(save_path)
    # # file_name = f'plot_{application}(200req_varyNode_Qty(nodes).pdf'
    # #
    # # st_file_name = f'statistic(avg)_{application}(200req_varyNode_Qty(nodes).pdf'
    # #
    # # plt.savefig(os.path.join(save_path, file_name), format='pdf')
    # #
    # # mean_delays = st.generate_by_count(st, [delays_neptune, delays_heu_xu, delays_pluto])
    # # mean_nodes = st.generate_by_count(st, [nodes_neptune, nodes_heu_xu, nodes_pluto])
    # # mean_memory = st.generate_by_count(st, [memory_neptune, memory_heu_xu, memory_pluto])
    # # print(f'mean_delays/mean_nodes/mean_memory ={mean_delays}/{mean_nodes}/{mean_memory}')
    # # st.create_statistical_table(st, ['Delay', 'Nodes', 'Memory'], ['NEPTUNE', 'HEU', 'PLUTO'],
    # #                             [mean_delays, mean_nodes, mean_memory], 'plots', st_file_name)
    plt.clf()  # clear the current graph
    return response


def statistic(cold_start_inclusion, coldstart_delays,
              decision_delays, total_delays, memories, network_delays, avg_network_delays, nodes, st_file_name):
    mean_delays = st.generate_by_count(st, total_delays)
    mean_allocation_delay = st.generate_by_count(st, decision_delays)
    mean_nodes = st.generate_by_count(st, nodes)
    mean_memory = st.generate_by_count(st, memories)
    mean_coldstart = st.generate_by_count(st, coldstart_delays)
    mean_network_delay = st.generate_by_count(st, network_delays)
    mean_avg_network_delay = st.generate_by_count(st, avg_network_delays)
    st.create_statistical_table(st, [f'Total delay{cold_start_inclusion}', 'Nodes', 'Memory', 'Coldstart',
                                     'Network delay', 'AVG network delay', 'Allocation delay'], ['VSVBP', 'NEPTUNE', 'CR-EUA','MCF', 'PLUTO', 'HEU'],
                                [mean_delays, mean_nodes, mean_memory, mean_coldstart, mean_network_delay,
                                 mean_avg_network_delay, mean_allocation_delay], 'plots', st_file_name)

# ['NEPTUNE', 'VSVBP', 'CR-EUA','MCF', 'PLUTO', 'HEU']
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000, debug=True)


app.run(host='0.0.0.0', port=5000, threaded=False, processes=10, debug=False)
