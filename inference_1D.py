import importlib.util
import inspect
import os
import numpy as np
import tmp as llama3
import json
import Utils
from Task import Task


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

    def get_llm_prediction(self, current_state_description, transformation_goal):
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


def state_transition_with_rewards(initial_state, target_state, task, max_depth=3):
    current_state = initial_state
    llm = LLM()

    for _ in range(max_depth):
        print(f"Current State: {current_state}")
        # curr_task = Task(initial_state, 0)
        # tar_task = Task(target_state, 0)
        cand = Candidate(ops=[], tasks=[], score=1000, predictions=np.zeros((2, 2)))
        cand.t = task
        func_list = Utils.getPossibleOperations(task, cand)
        for x in func_list:
            print(x)
        # print(func_list)
        return
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
