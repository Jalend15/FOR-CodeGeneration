import pandas as pd

# Load test data from CSV
test_data = pd.read_csv("/home/jalend/FOR-CodeGeneration/data/1D_ARC_test_data.csv")

# Inspect the data structure
print(test_data.head())

# Extract inputs and outputs from the test data
test_inputs = test_data["input"].tolist()
test_outputs = test_data["output"].tolist()

# Convert the string representations back into lists (if necessary)
formatted_inputs = [eval(input_string) for input_string in test_inputs]
formatted_outputs = [eval(output_string) for output_string in test_outputs]


# Placeholder for your inference engine
def inference_engine(input_data):
    # Dummy implementation for example purposes
    return input_data  # Replace with actual inference logic


# Perform inference and evaluate exact match accuracy
predictions = []
correct_matches = 0

for i, input_data in enumerate(formatted_inputs):
    prediction = inference_engine(input_data)  # Send input to the inference engine
    predictions.append(prediction)

    # Compare prediction with the actual output
    if prediction == formatted_outputs[i]:
        correct_matches += 1

# Calculate exact match accuracy
total_cases = len(formatted_inputs)
accuracy = (correct_matches / total_cases) * 100

print(f"Exact Match Accuracy: {accuracy:.2f}%")
