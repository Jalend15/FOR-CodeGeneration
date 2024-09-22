import os
import json
import pandas as pd

# Define the directory structure
base_dir = "data/1D-ARC/dataset"
folders = [
    "1d_denoising_1c",
    "1d_denoising_mc",
    "1d_fill",
    "1d_flip",
    "1d_hollow",
    "1d_mirror",
    "1d_move_1p",
    "1d_move_2p",
    "1d_move_2p_dp",
    "1d_move_3p",
    "1d_move_dp",
    "1d_padded_fill",
    "1d_pcopy_1c",
    "1d_pcopy_mc",
    "1d_recolor_cmp",
    "1d_recolor_cnt",
    "1d_recolor_oe",
    "1d_scale_dp",
]


# Function to load json data from a folder and append to a DataFrame
def load_and_split_data(base_dir, folders):
    train_data = []
    test_data = []

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".json"):
                    file_path = os.path.join(folder_path, file)
                    with open(file_path, "r") as json_file:
                        data = json.load(json_file)
                        for item in data.get("train", []):
                            train_data.append(
                                {
                                    "input": item["input"],
                                    "output": item["output"],
                                    "task": folder,
                                    "file": file,
                                }
                            )
                        for item in data.get("test", []):
                            test_data.append(
                                {
                                    "input": item["input"],
                                    "output": item["output"],
                                    "task": folder,
                                    "file": file,
                                }
                            )

    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Save as CSV
    train_csv_path = "1D_ARC_train_data.csv"
    test_csv_path = "1D_ARC_test_data.csv"

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    return train_csv_path, test_csv_path


# Load and split the data
train_csv, test_csv = load_and_split_data(base_dir, folders)
# # Display the combined DataFrame to the user
# tools.display_dataframe_to_user(name="Combined Data", dataframe=combined_df)
