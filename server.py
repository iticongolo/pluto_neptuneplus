class Server:
    def __init__(self, id, name="", cores=0, memory=0, cores_available=0, memory_available=0, location=(float('inf'), float('inf')),
                 status=1, nrt=None, parallel_f=None, sequential_f=None):
        self.id = id
        self.name = name
        self.status = status
        self.cores = cores
        self.memory = memory
        self.cores_available = cores_available
        self.memory_available = memory_available
        self.location = location
        self.parallel_f = parallel_f
        self.sequential_f = sequential_f
        self.nrt = nrt
        self.shared_cores = []
        self.shared_available_cores = []
        self.shared_memory = []
        self.shared_available_memory = []

    def set_status(self, status):
        self.status = status

    def set_cores_available(self, cores):
        self.cores_available = cores
        self.update_status()

    def set_memory_available(self, memory):
        self.memory_available = memory

    def update_available_cores(self, new_cores):
        self.cores_available = self.cores_available + new_cores
        self.update_status()

    # status = 1 underloaded, else, not underloaded
    def update_status(self):
        if self.cores_available > 0.2*self.cores:
            self.status = 1
        else:
            self.status = 0

    def initialization(self):
        self.set_cores_available(self.cores)
        self.set_memory_available(self.memory)
        self.set_status(1)

    def release_server(self, cluster_id):
        self.cores = self.cores+self.shared_cores[cluster_id]
        self.cores_available = self.cores_available + self.shared_available_cores[cluster_id]

        self.shared_cores[cluster_id] = 0
        self.shared_available_cores[cluster_id] = 0
        self.shared_memory[cluster_id] = 0
        self.shared_available_memory[cluster_id] = 0

    def release_shared_resources(self, cluster_id, remove_cores=0, remove_memory=0, remove_cores_available=0,
                                 remove_memory_available=0):
        self.shared_cores[cluster_id] = self.shared_cores[cluster_id]-remove_cores
        self.shared_available_cores[cluster_id] = self.shared_available_cores[cluster_id]-remove_cores_available
        self.shared_memory[cluster_id] = self.shared_memory[cluster_id]-remove_memory
        self.shared_available_memory[cluster_id] = self.shared_available_memory[cluster_id]-remove_memory_available

    def update_shared_resources(self, cluster_id, new_cores=0, new_memory=0, new_cores_available=0,
                                new_memory_available=0):
        self.shared_cores[cluster_id] = self.shared_cores[cluster_id]+new_cores
        self.shared_available_cores[cluster_id] = self.shared_available_cores[cluster_id]+new_cores_available
        self.shared_memory[cluster_id] = self.shared_memory[cluster_id]+new_memory
        self.shared_available_memory[cluster_id] = self.shared_available_memory[cluster_id]+new_memory_available

    # Note Closed
