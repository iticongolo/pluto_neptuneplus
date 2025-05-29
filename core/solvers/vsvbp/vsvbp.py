
import time
from core.utils.util import *
import numpy as np
from ortools.linear_solver import pywraplp
from ..solver import Solver
from ortools.sat.python import cp_model
from .utils import *
from ..neptune.utils.output import convert_c_matrix, convert_x_matrix
from ...utils.pluto_heuristic import PLUTO


class VSVBP(Solver):

    def __init__(self, num_users=8, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.x, self.c, self.y = {}, {}, {}
        self.first_step = True
        self.computing_time = 0

    def load_data(self, data):
        self.prepare_data(data)
        super().load_data(data)

    def prepare_data(self, data):
        data.num_users = self.num_users
        data.node_coords = delay_to_geo(data.node_delay_matrix)
        data.user_coords = place_users_close_to_nodes(data.num_users, data.node_coords)
        prepare_requests(data)
        prepare_req_distribution(data)
        prepare_coverage(data)

    def init_vars(self):
        init_x(self.data, self.model, self.x)
        init_c(self.data, self.model, self.c)
        init_y(self.data, self.model, self.y)

    def init_constraints(self):
        if self.first_step:
            constrain_coverage(self.data, self.model, self.x)
            constrain_proximity(self.data, self.model, self.x)
            constrain_memory(self.data, self.model, self.c, self.y)
            constrain_cpu(self.data, self.model, self.x, self.y)
            constrain_request_handled(self.data, self.model, self.x)
            constrain_c_according_to_x(self.data, self.model, self.c, self.x)    
            constrain_y_according_to_x(self.data, self.model, self.y, self.x)
            constrain_amount_of_instances(self.data, self.model, self.c)
        else:
            add_hints(self.data, self.model, self.solver, self.x)
            constrain_previous_objective(self.data, self.model, self.x, self.solver.ObjectiveValue())
            
    
    def init_objective(self):
        if self.first_step:
            maximize_handled_requests(self.data, self.model, self.x)
        else:
            minimize_utilization(self.data, self.model, self.y)


    def solve(self):
        start_time = time.time()
        self.init_objective()
        self.solver.Solve(self.model)
        self.first_step = False

        self.init_constraints()
        self.init_objective()
        self.status = self.solver.Solve(self.model)
        end_time = time.time()
        self.computing_time = self.computing_time + end_time - start_time

        # self.log(f"Problem solved with status {self.status}")

    def results(self):
        xjr = output_xjr(self.data, self.solver, self.status, self.x, self.c, self.y)
        x, c = output_x_and_c(self.data, self.solver, self.c, xjr)
        return convert_x_matrix(x, self.data.nodes, self.data.functions, self.data.nodes), convert_c_matrix(c, self.data.functions, self.data.nodes)

    def dep_results(self):
        start_time = time.time()
        x, y, z = get_dep_results(self.data,  self.x, self.c)
        end_time = time.time()
        self.computing_time = self.computing_time+end_time-start_time
        return x, y, z

    def basic_data(self):
        function = get_functions(self.data)
        nodes = len(self.data.cluster.servers)
        funcs = len(self.data.functions)
        return funcs, function, nodes

    def internal_workload(self):
        w = get_internal_workload(self.data, self.x, self.c)
        return w

    def dep_network_delay(self):
        net_delay = get_net_delay(self.data, self.x, self.c)
        return net_delay

    def object_function_global_results(self):
        pluto = PLUTO(self.data)
        x, y, z = self.dep_results()
        w = self.internal_workload()
        lamb = self.data.workload_matrix
        pluto.set_coldstart()
        total_delay, coldstart, network_delay = pluto.object_function_heuristic(w, x, y, z, lambd=lamb,
                                                                                instances=self.c)
        decision_time = self.computing_time*1000  # in milliseconds
        return total_delay, coldstart, network_delay, decision_time

    def get_resources_usage(self):
        nodes = []
        instancef = []
        for f in range(len(self.c)):
            instances = 0
            for j in range(len(self.c[0])):
                if self.c[f, j]:
                    instances = instances+self.c[f, j]
                    if j not in nodes:
                        nodes.append(j)
            instancef.append([f, instances])

        return len(nodes), instancef

    def get_memory_used(self, mf):
        _, instances = self.get_resources_usage()
        # print(f'@@@@@@@@@@@@@ instances={instances}')
        memory = 0
        for f in range(len(instances)):
            memory = memory + instances[f][1]*mf[f]
        return memory

