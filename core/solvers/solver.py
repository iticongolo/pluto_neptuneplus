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

    def solve(self):

        self.solver.SetSolverSpecificParametersAsString(f"randomization/randomseedshift = {random.randint(0, 10**6)}")
        self.init_objective()
        status = self.solver.Solve()
        value = self.solver.Objective().Value()
        self.log(f"Problem solved with status {status} and value {value}")
        return status == pywraplp.Solver.OPTIMAL

    def results(self):
        raise NotImplementedError("Solvers must implement results()")
    def score(self) -> float:
        return self.objective.Value()
