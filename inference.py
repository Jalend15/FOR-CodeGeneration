import importlib.util
import inspect
import os
import random

def extract_dsl_functions_from_file(file_path):
    # Get the module name by stripping the '.py' and extracting the base name
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get all the functions defined in the module
    dsl_functions = {
        name: obj for name, obj in inspect.getmembers(module, inspect.isfunction)
    }

    return dsl_functions

# Example usage
dsl_functions = extract_dsl_functions_from_file('dsl.py')

# Print the function names and actual functions
for name, func in dsl_functions.items():
    print(f"Function name: {name}, Function object: {func}")


# LLM model placeholder to simulate choosing DSL functions
class LLM:
    def __init__(self):
        self.history = []  # Keep track of the functions used
        self.rewards = []  # Keep track of intermediate rewards

    def choose_function(self, state):
        # Sample one DSL function from the distribution of all DSL functions
        chosen_function = random.choice(dsl_functions)
        print(chosen_function)
        self.history.append(chosen_function.__ne__)
        return chosen_function

# Function to calculate intermediate rewards (e.g., difference from target state)
def calculate_reward(current_state, target_state):
    # Reward as absolute difference between current state and target state
    return abs(current_state - target_state)

# Function to check if target state is reached
def is_target_state(state, target_state):
    return state == target_state

# Main loop
def state_transition_with_rewards(initial_state, target_state, max_depth=3):
    current_state = initial_state
    llm = LLM()

    for _ in range(max_depth):
        print(f"Current State: {current_state}")
        chosen_function = llm.choose_function(current_state)
        print(f"LLM chose function: {chosen_function.__ne__}")

        # Apply the chosen function
        current_state = chosen_function(current_state)

        # Calculate intermediate reward
        reward = calculate_reward(current_state, target_state)
        llm.rewards.append(reward)

        # Check if the target state is reached
        if is_target_state(current_state, target_state):
            print(f"Success! Reached target state: {target_state}")
            return llm.history, llm.rewards  # Return history and rewards

        print(f"Intermediate Reward: {reward}")

    print("Reached max depth without reaching target state.")
    return llm.history, llm.rewards

# Example usage
initial_state = 5
target_state = 0
history_of_functions, rewards = state_transition_with_rewards(initial_state, target_state)

print(f"History of DSL functions used: {history_of_functions}")
print(f"Intermediate rewards: {rewards}")
