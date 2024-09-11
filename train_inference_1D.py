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
from Task import Matrix
import random


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

    def num_to_char(self, num):
        mapping = {
            0: ".",
            1: "a",
            2: "b",
            3: "c",
            4: "d",
            5: "e",
            6: "f",
            7: "g",
            8: "h",
            9: "i",
        }
        return mapping.get(num, ".")

    # Function to convert the list of numbers into a grid and generate the views
    def generate_views(self, numbers, grid_shape):
        # Reshape the flat list into a 2D grid
        grid = np.array(numbers).reshape(grid_shape)

        # Grid View: Replace numbers with characters
        grid_view = (
            f"1. Grid View: {[self.num_to_char(cell) for row in grid for cell in row]}"
        )

        # Object View (Mono-Color): Identify contiguous non-zero cells and create objects
        object_view = "2. Object View (Mono-Color):["
        objects = []

        visited = np.zeros_like(grid, dtype=bool)  # Keep track of visited cells

        # Function to perform DFS to find contiguous cells
        def dfs(i, j, grid, visited):
            stack = [(i, j)]
            obj_cells = []
            while stack:
                ci, cj = stack.pop()
                if visited[ci, cj] or grid[ci, cj] == 0:
                    continue
                visited[ci, cj] = True
                obj_cells.append((ci, cj))
                # Explore neighbors (left, right, up, down)
                for ni, nj in [(ci - 1, cj), (ci + 1, cj), (ci, cj - 1), (ci, cj + 1)]:
                    if (
                        0 <= ni < grid.shape[0]
                        and 0 <= nj < grid.shape[1]
                        and not visited[ni, nj]
                    ):
                        stack.append((ni, nj))
            return obj_cells

        # Find all objects in the grid
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0 and not visited[i, j]:
                    obj_cells = dfs(i, j, grid, visited)
                    if obj_cells:
                        # Get bounding box of the object
                        rows = [c[0] for c in obj_cells]
                        cols = [c[1] for c in obj_cells]
                        start_row, end_row = min(rows), max(rows)
                        start_col, end_col = min(cols), max(cols)
                        # Generate the object's shape
                        shape = [
                            [
                                "x" if (r, c) in obj_cells else "."
                                for c in range(start_col, end_col + 1)
                            ]
                            for r in range(start_row, end_row + 1)
                        ]
                        objects.append(
                            {
                                "start_index": (start_row, start_col),
                                "length": len(obj_cells),
                                "cell_count": len(obj_cells),
                                "shape": shape,
                            }
                        )

        # Convert object information into string format
        object_view_entries = []
        for obj in objects:
            shape_str = ["".join(row) for row in obj["shape"]]
            obj_str = f"{{'start_index': {obj['start_index']}, 'length': {obj['length']}, 'cell_count': {obj['cell_count']}, 'shape': {shape_str}}}"
            object_view_entries.append(obj_str)
        object_view += ",".join(object_view_entries) + "]"

        # Pixel View: Group the coordinates of each pixel value
        pixel_view = "3. Pixel View: {"
        pixel_dict = {}
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                char = self.num_to_char(grid[i, j])
                if char not in pixel_dict:
                    pixel_dict[char] = []
                pixel_dict[char].append((i, j))

        # Convert pixel dictionary to string format
        pixel_view_entries = []
        for key, coords in pixel_dict.items():
            pixel_view_entries.append(f"'{key}': {coords}")
        pixel_view += ", ".join(pixel_view_entries) + "}"

        # Combine all views into one string
        views = f"""
        {grid_view}
        {object_view}
        {pixel_view}
        """
        return views

    def get_llm_prediction(
        self, current_state_description, transformation_goal, func_list, func_dict
    ):
        # views = """
        # 1. Grid View: ['.', '.', '.', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', '.', '.', 'a', '.', '.', 'a', '.', '.', '.', '.', 'a', '.', '.', '.', '.', '.', '.', '.']
        # 2. Object View (Mono-Color):[{'start_index': 3, 'length': 10, 'cell_count': 10, 'shape': ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']},{'start_index': 16, 'length': 1, 'cell_count': 1, 'shape': ['x']},{'start_index': 19, 'length': 1, 'cell_count': 1, 'shape': ['x']},{'start_index': 23, 'length': 1, 'cell_count': 1, 'shape': ['x']}]
        # 3. Pixel View: {'a': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 19, 23]}
        # """
        print(current_state_description)
        print(len(current_state_description[0]))
        views = self.generate_views(
            current_state_description, (1, len(current_state_description[0]))
        )
        print("Views:", views)
        with open("data/prompts/example_functions.txt", "r") as f:
            assert_helpers = f.read()
        print("Length of function list,", len(func_list))
        try:
            func_list_string = "\n\n\n".join(
                ["<" + str(func) + ">" for func in func_dict.keys()]
            )
            print(func_list_string)

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
    hamming_dist = hamming_distance(
        np.array(intermediate_state), np.array(target_state)
    )
    max_distance = len(intermediate_state[0])  # Total number of elements in the matrix
    print(f"Max distance {max_distance}")
    print(f"Hamming distance {hamming_dist}")

    # Calculate normalized reward (1 means perfect match, 0 means completely different)
    normalized_reward = 10.0 * (1.0 - (hamming_dist / (max_distance + 0.0000001)))

    return normalized_reward


def is_target_state(state, target_state):
    return np.array_equal(state, target_state)


def state_transition_with_rewards(initial_state, target_state, task, max_depth=1):
    current_state = initial_state
    llm = LLM()

    for i in range(max_depth):
        print(max_depth)
        print(f"Iteration {i}")
        print(f"Current State: {current_state}")

        cand = Candidate(ops=[], tasks=[], score=1000, predictions=np.zeros((2, 2)))
        cand.t = task
        func_list = getPossibleOperations(task, cand)

        filtered_func_list = []
        other_func_list = []
        for func in func_list:
            try:
                # Apply the function to the intermediate state
                result = func(Matrix(current_state))
                # print(result)
                # print(target_state)
                # If the result matches the target state, keep the function
                if np.array_equal(result, target_state):
                    filtered_func_list.append(func)
                    print(f"Function {func} can transform the state successfully.")
                else:
                    other_func_list.append(func)
            except Exception as e:
                print(f"Error applying function {func}: {e}")
        print("Length of filtered function list,", len(filtered_func_list))

        # Choose all functions from filtered_func_list and a few from other_func_list
        func_list = filtered_func_list.copy()

        # Define how many functions you want to randomly pick from other_func_list
        num_random_from_other = 5  # For example, pick 3 random functions

        # Ensure that we don't pick more than what's available in other_func_list
        if len(other_func_list) > 0:
            num_random_from_other = min(num_random_from_other, len(other_func_list))
            func_list.extend(random.sample(other_func_list, num_random_from_other))

        function_dict = {}
        for func in func_list:
            func_name = func.func.__name__
            args = func.args
            kwargs = func.keywords

            # Create a unique key for each function including its arguments
            key = f"{func_name}(args={args}, kwargs={kwargs})"

            # Store the partial function in the dictionary
            function_dict[key] = func

        llm_response = llm.get_llm_prediction(
            current_state_description=current_state,
            transformation_goal=target_state,
            func_list=func_list,
            func_dict=function_dict,
        )

        if not llm_response:
            print("Prediction failed.")
            break

        match = re.search(r"<(.*?)>", llm_response)

        if match:
            extracted_text = match.group(1)
            print("Extracted text:", extracted_text)
        else:
            print("No valid text found inside the angle brackets.")
        # cleaned_text = re.search(r"\w+\(.*?\)\>", llm_response)

        # if cleaned_text:
        #     cleaned_text = cleaned_text.group(0)  # Extract the entire function call
        #     print("Cleaned text:", cleaned_text)
        # else:
        #     print("No valid function found in response.")
        #     return llm.history, llm.rewards, 0.0

        # print(f"LLM chose function: {cleaned_text}")

        # # match = re.search(
        # #     r"`functools\.partial\(<function (\w+) at 0x[\da-f]+>,\s*(.+)\)",
        # #     cleaned_text,
        # # )
        # match = re.search(r"(\w+)\(args=\(\), kwargs=\{(.*)\}\)", cleaned_text)

        # print("match1,", match)
        # text = cleaned_text
        # match = re.search(r"<function (\w+) at .*>,\s*(.*)\)$", text)
        # print(match)
        # Check if the match was successful
        if match:
            # func_name = match.group(1)
            # print(func_name)
            # args_string = match.group(2)
            # print(f"Function Name: {func_name}")
            # print(f"Arguments: {args_string}")
            # # Initialize an empty dictionary to store arguments
            # args_dict = {}

            # # Use a more sophisticated regex to parse key-value pairs safely
            # argument_pattern = re.findall(
            #     r"(\w+)=({[^}]*}|'[^']*'|None|True|False|[\w]+)", args_string
            # )

            # for key, value in argument_pattern:
            #     try:
            #         # Safely evaluate the argument value using ast.literal_eval
            #         args_dict[key] = ast.literal_eval(value)
            #     except (SyntaxError, ValueError):
            #         # If it can't be evaluated, keep it as a string (in case of unrecognized formats)
            #         args_dict[key] = value

            # # Construct the dictionary key as it was created
            # key = f"{func_name}(args=(), kwargs={args_dict})"
            key = match.group(1)
            print("Key from dict", key)
            print("\n\n\n\n\n\n")
            for x in function_dict.keys():
                print(x)

            # Retrieve the function from the dictionary
            from difflib import get_close_matches

            # Find closest matching key
            closest_matches = get_close_matches(
                key, function_dict.keys(), n=1, cutoff=0.8
            )

            if closest_matches:
                partial_func = function_dict.get(closest_matches[0])
                print("Closest Match Found:", closest_matches[0])
            else:
                print("No close match found in the dictionary.")
                llm.rewards.append(0)
                return llm.history, llm.rewards, 0.0
            print("Partial function from dictionary", partial_func)

            if partial_func:
                # Call the function with the current state
                current_state = partial_func(Matrix(current_state))
                print(f"Function result: {current_state}")
            else:
                print("No matching function found in the dictionary.")
                llm.rewards.append(0)
                return llm.history, llm.rewards, 0.0
        else:
            print("No match found.")
            llm.rewards.append(0)
            return llm.history, llm.rewards, 0.0
        reward = calculate_reward(current_state, target_state)
        llm.rewards.append(reward)

        if is_target_state(current_state, target_state):
            print(f"Success! Reached target state: {target_state}")
            return llm.history, llm.rewards, 1.0

        print(f"Intermediate Reward: {reward}")

    return llm.history, llm.rewards, 0.0


# Load data from train.csv
train_csv = "/home/jalend/FOR-CodeGeneration/data/1D_ARC_train_data.csv"
complete_train_data = pd.read_csv(train_csv)


tasks = ["1d_move_2p", "1d_padded_fill", "1d_denoising_1c"]
# tasks = complete_train_data["task"].unique().tolist()
# print(complete_train_data["task"].unique().tolist())
train_data_by_task = complete_train_data[complete_train_data["task"].isin(tasks)]


no_of_examples_per_task = 3
# Extract 10 examples for each task
train_data = train_data_by_task.groupby("task").head(no_of_examples_per_task)
# print(train_data)

# # Load data from test.csv
test_csv = "/home/jalend/FOR-CodeGeneration/data/1D_ARC_test_data.csv"
complete_test_data = pd.read_csv(test_csv)
test_data_by_task = complete_test_data[complete_test_data["task"].isin(tasks)]

# Extract 10 examples for each task
test_data = test_data_by_task.groupby("task").head(no_of_examples_per_task)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
print(test_data)


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
    "test": test_list,
    "train": train_list,
    "uuid": "some_unique_identifier",  # Replace with actual UUID logic if necessary
}

print(data)


def ensure_correct_format(data):
    for item in data["train"]:
        # Check if input is a string, safely parse it using ast.literal_eval, else use it as is
        if isinstance(item["input"], str):
            item["input"] = ast.literal_eval(item["input"])
        if isinstance(item["output"], str):
            item["output"] = ast.literal_eval(item["output"])

        # Convert inputs and outputs to NumPy arrays and ensure they are float32
        item["input"] = np.array(item["input"], dtype=np.float32)
        item["output"] = np.array(item["output"], dtype=np.float32)

        # Print shapes for debugging
        print(f"Train Input Shape: {item['input'].shape}")
        print(f"Train Output Shape: {item['output'].shape}")

    for item in data["test"]:
        # Check if input is a string, safely parse it using ast.literal_eval, else use it as is
        if isinstance(item["input"], str):
            item["input"] = ast.literal_eval(item["input"])
        if isinstance(item["output"], str):
            item["output"] = ast.literal_eval(item["output"])

        # Convert inputs and outputs to NumPy arrays and ensure they are float32
        item["input"] = np.array(item["input"], dtype=np.float32)
        item["output"] = np.array(item["output"], dtype=np.float32)

        # Print shapes for debugging
        print(f"Test Input Shape: {item['input'].shape}")
        print(f"Test Output Shape: {item['output'].shape}")

    return data


# Example of using this before passing to Task
data = ensure_correct_format(data)
print(data)

accuracy_by_class = {
    task: {"correct": 0, "total": 0} for task in tasks
}  # For each task (class)
overall_correct_predictions = 0.0
overall_total_predictions = 0.0
max_depth = 1
for idx, row in train_data.iterrows():
    print(f"Example {idx}")
    data1 = {
        "test": test_list[idx : idx + 1],
        "train": test_list[idx : idx + 1],
        "uuid": "some_unique_identifier",  # Replace with actual UUID logic if necessary
    }
    input_data = eval(row["input"])  # Convert string to list/matrix
    output_data = eval(row["output"])

    print(f"Processing example {idx + 1}")
    print(f"Input: {input_data}")
    print(f"Expected Output: {output_data}")

    # Apply the state transition logic
    history_of_functions, rewards, acc = state_transition_with_rewards(
        initial_state=input_data,
        target_state=output_data,
        task=Task(data1, 0),
        max_depth=max_depth,  # Adjust depth as needed
    )
    # Update correct predictions and accuracy tracking for the class
    if acc == 1.0:  # Assuming acc returns 1.0 for a correct prediction
        accuracy_by_class[row["task"]]["correct"] += 1
        overall_correct_predictions += 1

    accuracy_by_class[row["task"]]["total"] += 1
    overall_total_predictions += 1

# Calculate overall accuracy
overall_accuracy = overall_correct_predictions / overall_total_predictions
# Print accuracy by class and overall accuracy
print("Accuracy by Class:")
for task, values in accuracy_by_class.items():
    task_accuracy = values["correct"] / values["total"] if values["total"] > 0 else 0
    print(f"{task}: {task_accuracy:.2f}")

print(f"Overall Accuracy: {overall_accuracy:.2f}")
# Output the results for this example
print(f"History of DSL functions used: {history_of_functions}")
print(f"Intermediate rewards: {rewards}")
# print(f"Correct predictions: {correct_predictions}")
# print(f"Accuracy: {accuracy}")
