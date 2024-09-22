import importlib.util
import inspect
import os

import numpy as np

# import OpenAI

import input_data
import tmp as llama3


def extract_dsl_functions_from_file(file_path):
    """
    Extracts DSL functions from a Python file dynamically.

    Parameters:
    file_path (str): Path to the Python file containing DSL functions.

    Returns:
    dict: A dictionary where keys are function names and values are the function objects.
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name: obj for name, obj in inspect.getmembers(module, inspect.isfunction)}


dsl_functions_dict = extract_dsl_functions_from_file("solvers.py")


class LLM:
    """
    A class representing an LLM model interface that predicts DSL functions and arguments.

    Attributes:
    history (list): Stores the history of functions used.
    rewards (list): Stores the intermediate rewards.
    """

    def __init__(self):
        self.history = []  # Track the functions used
        self.rewards = []  # Track intermediate rewards

    def get_llm_prediction(self, current_state_description, transformation_goal):
        """
        Uses an LLM to predict the next DSL function and its arguments based on the current state and goal.

        Parameters:
        current_state_description (str): Description of the current state.
        transformation_goal (str): Description of the target transformation goal.

        Returns:
        tuple: A tuple containing the predicted DSL function name and a list of arguments.
        """
        dsl_file_path = "data/prompts/dsl_prompt.txt"
        constants_file_path = "data/prompts/constant_prompt.txt"

        with open(dsl_file_path, "r") as dsl_file:
            dsl_prompt = dsl_file.read().strip()

        with open(constants_file_path, "r") as constants_file:
            constants_prompt = constants_file.read().strip()

        # client = OpenAI()

        try:
            prompt = f"""
            You are an LLM which predicts the correct function name and arguments. Your only output should be function name and arguments.
            Given the current state described as:
            {current_state_description}

            And the goal to:
            {transformation_goal}

            Predict the next DSL function and its arguments.
            Give only the DSL function and arguments in the format mentioned below and nothing else.
            Provide the output in this format:
            Function name: <function_name>
            Arguments: [<arg1>, <arg2>, ...]

            DSL functions available:
            {dsl_prompt}

            Arguments available:
            {constants_prompt}
            """
            # MODEL = "gpt-4o"
            # completion = client.chat.completions.create(
            #     model=MODEL,
            #     messages=[
            #         {
            #             "role": "system",
            #             "content": prompt,
            #         },
            #         {
            #             "role": "user",
            #             "content": "Choose functions and arguments such that target matrix is reached",
            #         },
            #     ],
            # )
            pipeline = llama3.load_text_generation_pipeline()

            # Generate a response
            response = llama3.generate_response(
                pipeline,
                system_prompt=prompt,
                user_query="Choose functions and arguments such that target matrix is reached",
            )
            response_text = response[-1]["content"]
            print("Response")
            print(response)
            print(response[-1]["content"])
            print("End of response")
            # response_text = completion.choices[0].message.content
            function_name = (
                response_text.split("Function name: ")[1].split("\n")[0].strip()
            )
            arguments_str = response_text.split("Arguments: ")[1].strip()
            arguments = eval(
                arguments_str
            )  # Convert string representation of list to actual list
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None, None

        self.history.append(function_name)
        return function_name, arguments


def pad_matrices(matrix1, matrix2):
    """
    Pads two matrices to the same size if they have different dimensions.

    Parameters:
    matrix1 (np.ndarray): The first matrix.
    matrix2 (np.ndarray): The second matrix.

    Returns:
    tuple: A tuple of padded matrices.
    """
    max_rows = max(matrix1.shape[0], matrix2.shape[0])
    max_cols = max(matrix1.shape[1], matrix2.shape[1])

    padded_matrix1 = np.pad(
        matrix1,
        ((0, max_rows - matrix1.shape[0]), (0, max_cols - matrix1.shape[1])),
        mode="constant",
    )
    padded_matrix2 = np.pad(
        matrix2,
        ((0, max_rows - matrix2.shape[0]), (0, max_cols - matrix2.shape[1])),
        mode="constant",
    )

    return padded_matrix1, padded_matrix2


def hamming_distance(matrix1, matrix2):
    """
    Computes the Hamming distance between two matrices.

    Parameters:
    matrix1 (np.ndarray): The first matrix.
    matrix2 (np.ndarray): The second matrix.

    Returns:
    int: The Hamming distance between the two matrices.
    """
    if matrix1.shape != matrix2.shape:
        matrix1, matrix2 = pad_matrices(matrix1, matrix2)

    return np.sum(matrix1.flatten() != matrix2.flatten())


def calculate_reward(intermediate_state, target_state):
    """
    Calculates the reward as the Hamming distance between the intermediate state and the target state.

    Parameters:
    intermediate_state (np.ndarray): The current state of the system.
    target_state (np.ndarray): The target state of the system.

    Returns:
    int: The computed reward (Hamming distance).
    """
    return hamming_distance(np.array(intermediate_state), np.array(target_state))


def is_target_state(state, target_state):
    """
    Checks if the current state matches the target state.

    Parameters:
    state (np.ndarray): The current state.
    target_state (np.ndarray): The target state.

    Returns:
    bool: True if the current state matches the target state, otherwise False.
    """
    return np.array_equal(state, target_state)


def state_transition_with_rewards(initial_state, target_state, max_depth=3):
    """
    Simulates state transitions using predicted DSL functions and calculates rewards until the target state is reached or the max depth is exceeded.

    Parameters:
    initial_state (np.ndarray): The initial state of the system.
    target_state (np.ndarray): The target state of the system.
    max_depth (int): The maximum number of transitions allowed.

    Returns:
    tuple: A tuple containing the history of DSL functions used and the rewards at each step.
    """
    current_state = initial_state
    llm = LLM()

    for _ in range(max_depth):
        print(f"Current State: {current_state}")
        predicted_dsl_fn, dsl_arguments = llm.get_llm_prediction(
            current_state, target_state
        )
        if not predicted_dsl_fn:
            print("Prediction failed.")
            break

        print(f"LLM chose function: {predicted_dsl_fn}")

        # Call the function dynamically with the provided arguments.
        try:
            dsl_function = dsl_functions_dict.get(predicted_dsl_fn)
            if dsl_function is None:
                print(f"Function {predicted_dsl_fn} not found in DSL.")
                break

            # Call the DSL function with arguments (use *args for positional arguments)
            current_state = dsl_function(*dsl_arguments)
        except Exception as e:
            print(f"Error calling function {predicted_dsl_fn}: {e}")
            break

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


# Example Usage
train_data = input_data.get_data(train=True)
example_data_input = train_data["train"]["00d62c1b"][0]["input"]
example_data_output = train_data["train"]["00d62c1b"][0]["output"]

history_of_functions, rewards = state_transition_with_rewards(
    example_data_input, example_data_output
)
print(f"History of DSL functions used: {history_of_functions}")
print(f"Intermediate rewards: {rewards}")
