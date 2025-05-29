import requests
import pprint

from simulations.apps_inputs import ApplicationsInputs as apps
# from simulations.apps_vary_functions_inputs import ApplicationsVaryFunctionsInputs as apps


input = apps.inputs


def get_input():
    return input


for inp in input:
    inp["cores_matrix"] = [[1] * len(inp["node_memories"])] * len(inp["function_names"])
    inp["workload_on_destination_matrix"] = [[1] * len(inp["node_memories"])] * len(inp["function_names"])

response = requests.request(method='get', url="http://localhost:5003/", json=input, timeout=86400)
# timeout =24h=86400s

pprint.pprint(response.json())
