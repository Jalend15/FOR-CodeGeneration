import re

# The LLM response string
llm_response = """
Based on the input state and goal state, I predict that the next DSL_function to transform the input into the goal state is:

`functools.partial(<function downsize at 0x7f4764dec040>, newShape=(1, 33))`

This function will downsize the input state to the goal state, reducing the number of cells in the grid while maintaining the same colors. The newShape argument specifies the desired shape of the output, which is (1, 33) in this case.
"""

# Remove backticks and newlines from the response to make the regex match more reliably
cleaned_response = llm_response.replace("`", "").replace("\n", " ")

# Regex to match function name and arguments from the cleaned response
match = re.search(
    r"functools\.partial\(<function (\w+) at 0x[\da-f]+>,\s*(.+)\)",
    cleaned_response,
)

if match:
    function_name = match.group(1)  # Extract the function name
    function_args = match.group(2)  # Extract the function arguments

    # Print the extracted values
    print(f"Function Name: {function_name}")
    print(f"Function Arguments: {function_args}")
else:
    print("No match found.")
