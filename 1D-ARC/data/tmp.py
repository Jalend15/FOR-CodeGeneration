import pandas as pd

# Load the original CSV file
df = pd.read_csv('1D_ARC_train_data.csv')

# Define the tasks to filter
tasks = ['1d_move_1p', '1d_fill', '1d_flip']

# Filter the dataframe to include only the rows with the specified tasks
filtered_df = df[df['task'].isin(tasks)]

# Create empty DataFrames for train and test data
train_df = pd.DataFrame(columns=filtered_df.columns)
test_df = pd.DataFrame(columns=filtered_df.columns)

# For each task, split 15 rows for training and the rest for testing
for task in tasks:
    task_data = filtered_df[filtered_df['task'] == task]
    train_data = task_data.iloc[:5]  # First 15 rows for training
    test_data = task_data.iloc[:45]  # Remaining rows for testing
    
    # Append to train and test DataFrames
    train_df = pd.concat([train_df, train_data], ignore_index=True)
    test_df = pd.concat([test_df, test_data], ignore_index=True)

# Save the training and testing data into CSV files
train_df.to_csv('train_3.csv', index=False)
test_df.to_csv('test_3.csv', index=False)

print("train_3.csv and test_3.csv have been created.")
