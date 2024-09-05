import os
import json
import numpy as np
import pandas as pd
import re
import ast
import functools
import quantized_model as llama3
from Task import Task
from Utils import getPossibleOperations


class Candidate:
    def __init__(self, ops, tasks, score=1000, predictions=np.zeros((2, 2))):
        self.ops = ops
        self.score = score
        self.tasks = tasks
        self.t = None
        self.predictions = predictions
        self.checked = False

    def __lt__(self, other):
        return self.score < other.score

    def generateTask(self):
        self.t = Task(self.tasks[-1], "dummyIndex", submission=True)


class LLM:
    def __init__(self):
        self.history = []  # Track the functions used
        self.rewards = []  # Track intermediate rewards

    def get_llm_prediction(
        self, current_state_description, transformation_goal, func_list
    ):
        views = """
        1. Grid View: ['.', '.', '.', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', '.', '.', 'a', '.', '.', 'a', '.', '.', '.', '.', 'a', '.', '.', '.', '.', '.', '.', '.']
        2. Object View (Mono-Color):[{'start_index': 3, 'length': 10, 'cell_count': 10, 'shape': ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']},{'start_index': 16, 'length': 1, 'cell_count': 1, 'shape': ['x']},{'start_index': 19, 'length': 1, 'cell_count': 1, 'shape': ['x']},{'start_index': 23, 'length': 1, 'cell_count': 1, 'shape': ['x']}]
        3. Pixel View: {'a': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 19, 23]}
        """
        with open("data/prompts/example_functions.txt", "r") as f:
            assert_helpers = f.read()

        try:
            func_list_string = "\n".join([str(func) for func in func_list])

            prompt = f"""
            You are an LLM that predicts the correct DSL_function name given the Input State and the Goal State. Your output should **only** include the DSL_function name.

            ### Task Description:
            - **Input State**: A list described with different views:
            {current_state_description}

            - **Views**: 
            {views}

            - **Goal State**:
            {transformation_goal}

            ### Available DSL_function:
            {func_list_string}

            ### Instructions:
            - Predict the next DSL_function to transform the input into the goal state using the list of Available DSL_function. Just give the function output, no descriptions.
            """

            # Initialize LLaMA model pipeline
            pipeline = llama3.load_text_generation_pipeline()

            # Generate response from LLM
            response = llama3.generate_response(
                pipeline,
                system_prompt=prompt,
                user_query="Choose functions and arguments such that the target is reached.",
            )
            response_text = response[-1]["content"]
            print(f"LLM Response: {response_text}")

            function_name = response_text
            print("Predicted Function:", function_name)
        except Exception as e:
            print(f"Error during LLM inference: {e}")
            return None

        self.history.append(function_name)
        return function_name


def pad_matrices(matrix1, matrix2):
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
    if matrix1.shape != matrix2.shape:
        matrix1, matrix2 = pad_matrices(matrix1, matrix2)
    return np.sum(matrix1.flatten() != matrix2.flatten())


def calculate_reward(intermediate_state, target_state):
    return hamming_distance(np.array(intermediate_state), np.array(target_state))


def is_target_state(state, target_state):
    return np.array_equal(state, target_state)


def state_transition_with_rewards(initial_state, target_state, task, max_depth=1):
    current_state = initial_state
    llm = LLM()

    for _ in range(max_depth):
        print(f"Current State: {current_state}")

        cand = Candidate(ops=[], tasks=[], score=1000, predictions=np.zeros((2, 2)))
        cand.t = task
        func_list = getPossibleOperations(task, cand)

        llm_response = llm.get_llm_prediction(
            current_state_description=current_state,
            transformation_goal=target_state,
            func_list=func_list,
        )

        if not llm_response:
            print("Prediction failed.")
            break

        print(f"LLM chose function: {llm_response}")

        reward = calculate_reward(current_state, target_state)
        llm.rewards.append(reward)

        if is_target_state(current_state, target_state):
            print(f"Success! Reached target state: {target_state}")
            return llm.history, llm.rewards, 1

        print(f"Intermediate Reward: {reward}")

    return llm.history, llm.rewards, 0


# Load data from train.csv
train_csv = "/home/jalend/FOR-CodeGeneration/data/1D_ARC_train_data.csv"
train_data = pd.read_csv(train_csv)

# Load data from train.csv
test_csv = "/home/jalend/FOR-CodeGeneration/data/1D_ARC_test_data.csv"
test_data = pd.read_csv(train_csv)


# Helper function to convert CSV row to desired format
def convert_row_to_dict(row):
    return {
        "input": eval(row["input"]),  # Convert string to list
        "output": eval(row["output"]),  # Convert string to list
    }


# Convert train data
train_list = [convert_row_to_dict(row) for idx, row in train_data.iterrows()]

# Convert test data
test_list = [convert_row_to_dict(row) for idx, row in test_data.iterrows()]

# Create final data dictionary
data = {
    "test": test_list[:2],
    "train": train_list[:2],
    "uuid": "some_unique_identifier",  # Replace with actual UUID logic if necessary
}


def ensure_correct_format(data):
    for item in data["train"]:
        print(item)
        print(np.array(item["input"]).shape)
        item["input"] = (
            eval(item["input"]) if isinstance(item["input"], str) else item["input"]
        )
        item["output"] = (
            eval(item["output"]) if isinstance(item["output"], str) else item["output"]
        )

    for item in data["test"]:
        item["input"] = (
            eval(item["input"]) if isinstance(item["input"], str) else item["input"]
        )
        item["output"] = (
            eval(item["output"]) if isinstance(item["output"], str) else item["output"]
        )

    return data


# Example of using this before passing to Task
data = ensure_correct_format(data)
print(data)
accuracy = 0
for idx, row in train_data.iterrows():
    input_data = eval(row["input"])  # Convert string to list/matrix
    output_data = eval(row["output"])

    print(f"Processing example {idx + 1}")
    print(f"Input: {input_data}")
    print(f"Expected Output: {output_data}")

    # Apply the state transition logic
    history_of_functions, rewards, acc = state_transition_with_rewards(
        initial_state=input_data,
        target_state=output_data,
        task=Task(data, 0),
        max_depth=3,  # Adjust depth as needed
    )
    accuracy += acc / 100

    # Output the results for this example
    print(f"History of DSL functions used: {history_of_functions}")
    print(f"Intermediate rewards: {rewards}")
    break
