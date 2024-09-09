import importlib.util
import inspect
import os
import numpy as np
import quantized_model as llama3
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


data = {
    "train": [
        {
            "input": [
                [
                    0,
                    0,
                    0,
                    0,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    0,
                    0,
                    0,
                    4,
                    0,
                    0,
                    0,
                    4,
                    0,
                    0,
                    0,
                    0,
                    4,
                    0,
                ]
            ],
            "output": [
                [
                    0,
                    0,
                    0,
                    0,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ],
        }
    ],
    "test": [
        {
            "input": [
                [
                    0,
                    0,
                    0,
                    0,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    0,
                    0,
                    0,
                    4,
                    0,
                    0,
                    0,
                    4,
                    0,
                    0,
                    0,
                    0,
                    4,
                    0,
                ]
            ],
            "output": [
                [
                    0,
                    0,
                    0,
                    0,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ],
        }
    ],
}
# a = [
#     [
#         0,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         0,
#         0,
#         0,
#         6,
#         0,
#         0,
#         0,
#         0,
#         6,
#         0,
#         0,
#         0,
#         0,
#         6,
#         0,
#         0,
#         6,
#         0,
#     ]
# ]
# b = [
#     [
#         0,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         6,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#     ]
# ]


tmp = Task(data, 0)
cand = Candidate(ops=[], tasks=[], score=1000, predictions=np.zeros((2, 2)))
cand.t = tmp
func_list = Utils.getPossibleOperations(tmp, cand)
# print(func_list)
# for x in func_list:
#     print(x)
for func in func_list:
    try:
        # Apply the function to the input and store the result
        # print(func)
        result = func(tmp.trainSamples[0].inMatrix)
        # print("input:", tmp.trainSamples[0].inMatrix.m)
        # print("ground truth", data["train"][0]["output"])
        # print("griund truth", result)
        input_matrix = np.array(tmp.trainSamples[0].inMatrix.m)
        ground_truth_1 = np.array(data["train"][0]["output"])
        ground_truth_2 = np.array(result)
        import pandas as pd

        df_comparison = pd.DataFrame(
            {
                "i": input_matrix.flatten(),
                "o": ground_truth_1.flatten(),
                "r": ground_truth_2.flatten(),
            }
        )
        if np.array_equal(ground_truth_1, ground_truth_2):
            print(func)
            print(df_comparison)

    except Exception as e:
        print(f"Error running {func}: {e}")
