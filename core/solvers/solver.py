from ortools.linear_solver import pywraplp
from ..utils.data import Data
import datetime
import random
class Solver:
    def __init__(self, verbose: bool = True, **kwargs):
        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        self.verbose = verbose
        if verbose:
            self.solver.EnableOutput()
        self.objective = self.solver.Objective()
        self.data = None
        self.args = kwargs

    def load_data(self, data: Data):
        self.data = data
        self.log("Initializing variables...")
        self.init_vars()
        self.log("Initializing constraints...")
        self.init_constraints()

    def init_vars(self):
        raise NotImplementedError("Solvers must implement init_vars()")
    
    def init_constraints(self):
        raise NotImplementedError("Solvers must implement init_constraints()")
    
    def init_objective(self):
        raise NotImplementedError("Solvers must implement init_objective()")

    def log(self, msg: str):
        pass
        # if self.verbose:
        #     print(f"{datetime.datetime.now()}: {msg}")

    # def solve(self):
    #     # self.solver.set_time_limit(10)
    #     self.solver.SetSolverSpecificParametersAsString(f"randomization/randomseedshift = {random.randint(0, 10**6)}")
    #     self.init_objective()
    #     status = self.solver.Solve()
    #     value = self.solver.Objective().Value()
    #     self.log(f"Problem solved with status {status} and value {value}")
    #     return status == pywraplp.Solver.OPTIMAL

    # def solve(self):
    #     optimal_or_feasible = False
    #     self.solver.set_time_limit(30000000000) # use self.solver.set_time_limit(1800000) for few placements
    #     # self.solver.SetSolverSpecificParametersAsString(f"randomization/randomseedshift = {random.randint(0, 10**6)}")
    #
    #     self.solver.SetSolverSpecificParametersAsString(
    #         f"""
    #         randomization/randomseedshift = {random.randint(0, 10 ** 6)}
    #         limits/time = 25000000
    #         timing/clocktype = 1
    #         """
    #     )
    #     self.init_objective()
    #     status = self.solver.Solve()
    #     value = self.solver.Objective().Value()
    #     self.log(f"Problem solved with status {status} and value {value}")
    #     # NOTE new code to accommodate time_limit
    #     if status == pywraplp.Solver.OPTIMAL:
    #         print('Optimal solution:', value)
    #         optimal_or_feasible = True
    #     elif status == pywraplp.Solver.FEASIBLE:
    #         optimal_or_feasible = True
    #         print('Feasible solution found within time limit.', value)
    #     else:
    #         print('No solution found within time limit.')
    #         # return status == pywraplp.Solver.OPTIMAL
    #     return optimal_or_feasible

    def solve(self):
        optimal = False
        # self.solver.set_time_limit(100)  # Set the time limit
        self.solver.SetSolverSpecificParametersAsString(
            f"randomization/randomseedshift = {random.randint(0, 10 ** 6)}"
        )
        self.init_objective()
        status = self.solver.Solve()

        # Get the best value found so far (even if not optimal)
        if self.solver.Objective().BestBound() < float('inf'):
            value = self.solver.Objective().BestBound()
        else:
            value = None  # No feasible solution found

        self.log(f"Problem solved with status {status} and value {value}")

        # Handle solver statuses
        if status == pywraplp.Solver.OPTIMAL:
            print('Optimal solution:', value)
            optimal = True
        elif status == pywraplp.Solver.FEASIBLE:
            print('Feasible solution found within time limit:', value)
        else:
            print('No feasible solution found within time limit.')

        return optimal, value

    def results(self):
        raise NotImplementedError("Solvers must implement results()")
    def score(self) -> float:
        return self.objective.Value()
