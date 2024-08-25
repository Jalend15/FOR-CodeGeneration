import importlib.util
import inspect
import os
import random
import input_data
import numpy as np

def extract_dsl_functions_from_file(file_path):
    # Get the module name by stripping the '.py' and extracting the base name
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get all the functions defined in the module
    dsl_functions_dict = {
        name: obj for name, obj in inspect.getmembers(module, inspect.isfunction)
    }
    print(dsl_functions_dict)
    dsl_functions = [obj for name, obj in inspect.getmembers(module, inspect.isfunction)]
    dsl_names = [name for name, obj in inspect.getmembers(module, inspect.isfunction)]
    print(dsl_names[150:-30])
    return dsl_functions[150:-30]

# Example usage
dsl_functions = extract_dsl_functions_from_file('solvers.py')


# # Print the function names and actual functions
# for name, func in dsl_functions.items():
#     print(f"Function name: {name}, Function object: {func}")


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


def pad_matrices(matrix1, matrix2):
    # Determine the maximum shape in each dimension
    print(matrix1.shape)
    max_rows = max(matrix1.shape[0], matrix2.shape[0])
    max_cols = max(matrix1.shape[1], matrix2.shape[1])

    # Pad both matrices to the same size
    padded_matrix1 = np.pad(matrix1, ((0, max_rows - matrix1.shape[0]), (0, max_cols - matrix1.shape[1])), mode='constant')
    padded_matrix2 = np.pad(matrix2, ((0, max_rows - matrix2.shape[0]), (0, max_cols - matrix2.shape[1])), mode='constant')

    return padded_matrix1, padded_matrix2

def hamming_distance(matrix1, matrix2):
    # Pad matrices to the same size if they differ in size
    if matrix1.shape != matrix2.shape:
        matrix1, matrix2 = pad_matrices(matrix1, matrix2)

    # Flatten the matrices
    flat_matrix1 = matrix1.flatten()
    flat_matrix2 = matrix2.flatten()

    # Compute the Hamming distance
    return np.sum(flat_matrix1 != flat_matrix2)
# Function to calculate intermediate rewards (e.g., difference from target state)
def calculate_reward(intermediate_state, target_state):
    print(np.array(intermediate_state).shape)
    return hamming_distance(np.array(intermediate_state), np.array(target_state))

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
        print(f"LLM chose function: {chosen_function.__name__}")

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

# # Example usage
# initial_state = 5
# target_state = 0
# history_of_functions, rewards = state_transition_with_rewards(initial_state, target_state)


train_data = input_data.get_data(train=True)
example_data_input = train_data['train']['00d62c1b'][0]['input']
example_data_output = train_data['train']['00d62c1b'][0]['output']
# print(example_data_input)
history_of_functions, rewards = state_transition_with_rewards(example_data_input, example_data_output)
print(f"History of DSL functions used: {history_of_functions}")
print(f"Intermediate rewards: {rewards}")
