from .topology import nodes, network
from .util import *


class Parameters:

    # THE ACTIONS BELOW WERE DONE ON THE APPLICATION FILE CONSTRUCTOR
    # app_input["node_names"] = [server.name for server in cluster.servers]
    # app_input["node_delay_matrix"] = cluster.network_delays
    # app_input["node_memories"] = network[pos]["node_memories"] Done on the constructor
    # app_input["node_cores"] = network[pos]["node_cores"]

    # e.g.: servers_ids= [2,1,9,0]  function_ids=[7,3,0,6]  lambs =[200, 50, 10, 100]
    # explanation: 200 requests are sent to the instance of function 7 on server 2
    # for simplicity, we consider the same application for all the clusters
    # for each time window (e.g: only Hotel reservation, or Sock-shop, etc.)

    def do_nothing(self):
        pass
