class DSL_State:
    def __init__(self, grid, trajectory=None, intermediate_results=None):
        self.grid = grid  # The current grid (I at the initial state)
        self.trajectory = trajectory if trajectory is not None else []  # List of DSL functions used so far
        self.intermediate_results = intermediate_results if intermediate_results is not None else {}  # Store intermediate results like objs, black_objs, etc.

    def add_to_trajectory(self, dsl_function):
        """Add a DSL function to the trajectory."""
        self.trajectory.append(dsl_function)
    
    def update_intermediate_results(self, key, value):
        """Update intermediate results."""
        self.intermediate_results[key] = value
    
    def get_intermediate_result(self, key):
        """Retrieve intermediate result by key."""
        return self.intermediate_results.get(key, None)
