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