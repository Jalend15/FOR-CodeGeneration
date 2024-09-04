import importlib.util
import inspect
import os
import numpy as np
import tmp as llama3
import json
import Utils
from Task import Task
import re
import ast
import functools

class Candidate:
    """
    Objects of the class Candidate store the information about a possible
    candidate for the solution.

    ...
    Attributes
    ----------
    ops: list
        A list containing the operations to be performed to the input matrix
        in order to get to the solution. The elements of the list are partial
        functions (from functools.partial).
    score: int
        The score of the candidate. The score is defined as the sum of the
        number incorrect pixels when applying ops to the input matrices of the
        train samples of the task.
    tasks: list
        A list containing the tasks (in its original format) after performing
        each of the operations in ops, starting from the original inputs.
    t: Task.Task
        The Task.Task object corresponding to the current status of the task.
        This is, the status after applying all the operations of ops to the
        input matrices of the task.
    """

    def __init__(self, ops, tasks, score=1000, predictions=np.zeros((2, 2))):
        self.ops = ops
        self.score = score
        self.tasks = tasks
        self.t = None
        self.predictions = predictions
        self.checked = False

    def __lt__(self, other):
        """
        A candidate is better than another one if its score is lower.
        """
        if self.score == other.score:
            return len(self.ops) < len(other.ops)
        return self.score < other.score

    def generateTask(self):
        """
        Assign to the attribute t the Task.Task object corresponding to the
        current task status.
        """
        self.t = Task(self.tasks[-1], "dummyIndex", submission=True)


def extract_dsl_functions_from_file(file_path):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name: obj for name, obj in inspect.getmembers(module, inspect.isfunction)}


dsl_functions_dict = extract_dsl_functions_from_file("solvers.py")


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
            print(func_list_string)
            prompt = f"""
            You are an LLM which predicts the correct function name. Your only output should be function name.
            Given the current state list is described using the list and different views:
            Use helper tests to understand input transformations.
            Input:
            {current_state_description}
            
            The values from 'a' to 'j' represent different colors. '.' is a blank cell.
            For example, [['.','a','.'],['.','.','b']] represents a 2 row x 3 col grid with color a at position (1,0) and color b at position
            (2,1).
            Coordinates are 2D positions (row, col), row representing row number, col representing col number, with zero-indexing.
            Different views:
            {views}
            
            And the goal to:
            Goal:
            {transformation_goal}

            Helper Functions:
            {func_list_string} \n
            
            Use the assert tests as examples to understand input to output transformations. Input/output pairs may not reflect all possibilities, you are to infer the simplest possible relation.
            {assert_helpers}
            
            Predict the next DSL function.
            Give only the DSL function in the format mentioned below and nothing else.
            
            Provide the output in this format:
            Function name: <function_from_DSL_functions_available>
            
            Give the full name including arguments from the provided DSL functions. Give exactly the available function. Only output function name including arguments from the provided DSL functions. 
            """
            # print(prompt)
            pipeline = llama3.load_text_generation_pipeline()

            # Generate a response
            response = llama3.generate_response(
                pipeline,
                system_prompt=prompt,
                user_query="Choose functions and arguments such that target is reached",
            )
            response_text = response[-1]["content"]
            print("Response")
            # print(response)
            print(response[-1]["content"])
            print("End of response")
            # response_text = completion.choices[0].message.content
            function_name = response[-1]["content"]
            print("Functions", function_name)
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None, None

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
        # curr_task = Task(initial_state, 0)
        # tar_task = Task(target_state, 0)
        cand = Candidate(ops=[], tasks=[], score=1000, predictions=np.zeros((2, 2)))
        cand.t = task
        func_list = Utils.getPossibleOperations(task, cand)
        results = []
        for func in func_list:
            try:
                # Apply the function to the input and store the result
                result = func(task.trainSamples[0].inMatrix)
                results.append((func, result))  # Store function reference and its result
            except Exception as e:
                print(f"Error running {func}: {e}")
        # Print the results
        # for func, output in results:
        #     print(f"Function: {func}")
        #     print(f"Input: {task.trainSamples[0].inMatrix.m}\n")
        #     print(f"Output: {output}\n")
        # def format_function_call(func, *args, **kwargs):
        #     """
        #     Format the function call for writing into the assert statement.
        #     Extracts the function name, positional arguments, and keyword arguments.
        #     """
        #     func_name = func.func.__name__ if hasattr(func, 'func') else func.__name__
        #     arg_list = ', '.join([repr(arg) for arg in args])  # Positional arguments
        #     kwarg_list = ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])  # Keyword arguments

        #     # Combine positional and keyword arguments
        #     all_args = ', '.join(filter(None, [arg_list, kwarg_list]))
        #     return f"{func_name}({all_args})"

        # # Open a file for writing
        # with open("function_asserts.py", "w") as f:
        #     # Iterate through all functions in func_list and apply to the train_input
        #     for func in func_list:
        #         try:
        #             # If the function is functools.partial, extract arguments
        #             if isinstance(func, functools.partial):
        #                 args = (task.trainSamples[0].inMatrix.m,)
        #                 kwargs = func.keywords  # Get the keyword arguments passed to partial
        #             else:
        #                 args = (task.trainSamples[0].inMatrix.m,)
        #                 kwargs = {}

        #             # Apply the function to the input
        #             output = func(task.trainSamples[0].inMatrix)

        #             # Format the function call
        #             function_call = format_function_call(func, *args, **kwargs)

        #             # Write the assert statement in the required format
        #             f.write(f"assert {function_call} == {output}\n")

        #         except Exception as e:
        #             # Write error as comment if the function raises an exception
        #             f.write(f"# Error running {func}: {e}\n")
        # return
        function_dict = {}
        for func in func_list:
            func_name = func.func.__name__
            args = func.args
            kwargs = func.keywords

            # Create a unique key for each function including its arguments
            key = f"{func_name}(args={args}, kwargs={kwargs})"

            # Store the partial function in the dictionary
            function_dict[key] = func

        # # Printing the dictionary
        # for key, partial_func in function_dict.items():
        #     print(f"Function Identifier: {key}, Partial Function: {partial_func}")
        # return
        # print(func_list)
        llm_response = llm.get_llm_prediction(
            current_state_description=current_state,
            transformation_goal=target_state,
            func_list=func_list,
        )
        if not llm_response:
            print("Prediction failed.")
            break

        print(f"LLM chose function: {llm_response}")
        match = re.match(
            r"Function name: functools\.partial\(<function (\w+) at 0x[\da-f]+>, (.+)\)",
            llm_response,
        )

        print(match)
        # Check if the match was successful
        if match:
            func_name = match.group(1)
            args_string = match.group(2)
            print(f"Function Name: {func_name}")
            print(f"Arguments: {args_string}")
            args_dict = {}
            for arg in args_string.split(", "):
                key, value = arg.split("=")
                # Safely evaluate the argument value (e.g., convert strings to dict, set, etc.)
                args_dict[key] = ast.literal_eval(value)

            # Construct the dictionary key as it was created
            key = f"{func_name}(args=(), kwargs={args_dict})"

            # Retrieve the function from the dictionary
            partial_func = function_dict.get(key)
            print(partial_func)

            if partial_func:
                # Call the function with the current state
                result = partial_func(task.trainSamples[0].inMatrix)
                print(f"Function result: {result}")
            else:
                print("No matching function found in the dictionary.")
        else:
            print("No match found.")
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


json_file_path = "1d_denoising_1c_29.json"

with open(json_file_path, "r") as file:
    data = json.load(file)

test_data = data["test"]
for test_case in test_data:
    test_input = test_case["input"][0]
    test_output = test_case["output"][0]
    print("Test Input:", test_input)
    print("Test Output:", test_output)

# Extracting train data
train_data = data["train"]

tmp = Task(data, 0)
for train_case in train_data:
    train_input = train_case["input"][0]
    train_output = train_case["output"][0]
    print("Train Input:", train_input)
    print("Train Output:", train_output)
    state_transition_with_rewards(train_input, train_output, tmp)
    break

# example_data_input = train_data["train"]["00d62c1b"][0]["input"]
# example_data_output = train_data["train"]["00d62c1b"][0]["output"]

# history_of_functions, rewards = state_transition_with_rewards(
#     example_data_input, example_data_output
# )
# print(f"History of DSL functions used: {history_of_functions}")
# print(f"Intermediate rewards: {rewards}")
