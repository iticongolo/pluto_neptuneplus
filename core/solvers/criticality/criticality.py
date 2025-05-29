import time
from ..vsvbp import VSVBP
from .utils import *
from ..neptune.utils.output import convert_c_matrix, convert_x_matrix
from ..vsvbp.utils.output import output_x_and_c
from ...utils.pluto_heuristic import PLUTO
from ...utils.util import *


class Criticality(VSVBP):
    def __init__(self, danger_radius_km=0.5, **kwargs):
        super().__init__(**kwargs)
        self.danger_radius_km = danger_radius_km
        self.computing_time = 0

 
    def prepare_data(self, data):
        super().prepare_data(data)
        prepare_aux_vars(data, self.danger_radius_km)
        prepare_criticality(data)
        prepare_live_position(data)        
        prepare_coverage_live(data)


    def init_objective(self):
        if self.first_step:
            maximize_handled_most_critical_requests(self.data, self.model, self.x)
        else:
            super().init_objective()


class CriticalityHeuristic(Criticality):
    def init_vars(self): 
        self.x_jr, self.c_fj, self.y_j, self.S_active = init_all_vars(self.data)

    def init_constraints(self): pass
    def init_objective(self): pass

    def solve(self):
        start_time = time.time()
        criticality_heuristic(self.data, self.log, self.S_active, self.y_j, self.c_fj, self.x_jr)
        end_time = time.time()
        self.computing_time = self.computing_time + end_time - start_time

    def results(self):
        x, c = output_x_and_c(self.data, None, self.c_fj, self.x_jr)
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
