import pandas as pd
import numpy as np
import configuration as cng
from collections import defaultdict
import json
import re
import os


### functions ###

def extract_stimulus_details(stimulus_col):
    set_col = []
    obj_col = []
    vis_level_col = []

    for index, value in stimulus_col.items():
        match_set = re.search(r"set_(\d+)", value)  # Extracts set_x
        match_obj = re.search(r"obj_(\d+)", value)  # Extracts obj_x
        match_vis = re.search(r"vis_(\d+)", value)  # Extracts vis_x

        set_num = int(match_set.group(1)) if match_set else None
        obj_num = int(match_obj.group(1)) if match_obj else None
        vis_level = int(match_vis.group(1)) if match_vis else None

        set_col.append(set_num)
        obj_col.append(obj_num)
        vis_level_col.append(vis_level)

    d = {'set': set_col, 'stim_id': obj_col, 'vis_level': vis_level_col}
    return pd.DataFrame(data=d)


def jason_extractor(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")
                continue
    return data


def assign_main_part_info(df):
    # create new column to characterize the main part:
    df["trial_within_block"] = 0
    df["block_num"] = 0
    df["congruency"] = None

    # find the trials:
    indexes = df.index[df["test_part"] == "main_test"].tolist()

    # divide in the number of trials within staircase:
    num_repeat = int(len(indexes) / cng.number_of_trials_within_block)
    trial_count_within_block = [x for _ in range(num_repeat) for x in range(1, cng.number_of_trials_within_block + 1)]

    # Initialize tracking variables
    block_count = 1

    for i, count in zip(indexes, trial_count_within_block):
        if count == 1:
            now_cond = df.loc[i, 'condition']

        df.loc[i - 1: i + 1, 'trial_within_block'] += count

        if count < cng.number_of_trials_within_block:
            df.loc[i - 1: i + 1, 'block_num'] += block_count
            df.loc[i - 1: i + 1, 'obj_code'] = cng.cond_dict[now_cond]['obj_code']
            df.loc[i - 1: i + 1, 'congruency'] = cng.cond_dict[now_cond]['cong']
            df.loc[i - 1: i + 1, 'condition'] = cng.cond_dict[now_cond]['cond']

        elif count == cng.number_of_trials_within_block:
            df.loc[i - 1: i + 3, 'block_num'] += block_count
            df.loc[i - 1: i + 3, 'obj_code'] = cng.cond_dict[now_cond]['obj_code']
            df.loc[i - 1: i + 3, 'congruency'] = cng.cond_dict[now_cond]['cong']
            df.loc[i - 1: i + 3, 'condition'] = cng.cond_dict[now_cond]['cond']

            block_count += 1

    # Drop unnecessary rows
    mask = df.index.isin(range(indexes[0], indexes[-1])) & (df['block_num'] == 0)
    df = df[~mask]  # Keep rows where mask is False

    return df


def assign_staircase_info(df):
    # create new column to characterize the staircase part:

    df["trial_within_staircase"] = 0
    df["staircase_number"] = 0

    # Convert vis_level values to integers, keeping NaN as is
    df['vis_level'] = pd.to_numeric(df['vis_level'], errors='coerce').fillna(np.nan).astype('Int64')

    # find the trials:
    indexes = df.index[df["test_part"] == "stair_test"].tolist()

    # divide in the number of trials within staircase:
    num_repeat = int(len(indexes) / cng.number_of_trials_within_stair)
    trial_count_within_stair = [x for _ in range(num_repeat) for x in range(1, cng.number_of_trials_within_stair + 1)]

    # Initialize tracking variables
    staircase_count = 1
    now_obj_code = None

    for i, count in zip(indexes, trial_count_within_stair):
        # assign object code:
        if count == 1:
            now_vis_level_ind = df.loc[i:]['vis_level']
            now_vis_level_ind = pd.to_numeric(now_vis_level_ind, errors='coerce')
            # Find the first valid integer (not NA)
            now_vis_level_ind = now_vis_level_ind[~now_vis_level_ind.isna()].apply(
                lambda x: isinstance(x, int)).idxmax()
            now_vis_level = df.loc[now_vis_level_ind, 'vis_level']
            if staircase_count == 1:
                now_set_and_obj = df.loc[i - 2, 'stimulus']
                match = re.search(cng.pattern_instr_stair, now_set_and_obj)
                if match:
                    x_value = int(match.group(1))  # First capturing group for set
                    y_value = int(match.group(2))  # Second capturing group for object
                    now_obj_code = [x_value, y_value]
                else:
                    now_obj_code = None

        df.loc[i - 1: i + 1, 'trial_within_staircase'] += count
        df.loc[i - 1: i + 1, 'staircase_number'] += staircase_count
        df.loc[i - 1: i + 1, 'obj_code'] = str(now_obj_code)
        df.loc[i - 1: i + 1, 'vis_level'] = int(now_vis_level)

        if count == cng.number_of_trials_within_stair:
            if isinstance(df.loc[i + 3, 'value'], str):
                if df.loc[i + 3, 'value'] == 'true':
                    staircase_count += 1
                elif df.loc[i + 3, 'value'] == 'false':
                    staircase_count = 1

    return df


def arrange_output_json(path, participant_id=None):
    # Load JSON data (assuming it's a string)
    with open(path, "r") as f:
        data_str = json.load(f)  # This should be a string

    # Split by rows
    rows = data_str.strip().split("\n")  # Remove extra whitespace and split by newline

    # Extract headers (first row)
    headers = rows[0].strip().split(",")  # First row contains column names

    # Extract data (remaining rows)
    data = [row.strip().split(",") for row in rows[1:]]  # Process each row

    # Group rows by their length
    grouped_data = defaultdict(list)

    for row in data:
        grouped_data[len(row)].append(row)

    # Convert each section into a DataFrame
    dfs = {f"Section_{length}": pd.DataFrame(rows) for length, rows in grouped_data.items()}

    # Print available sections
    print("Sections identified:", dfs.keys())

    # Define the sections that need fixing (exclude 'Section_18' and 'Section_57')
    sections_to_fix = [key for key in dfs.keys() if key not in ["Section_18", "Section_57", "Section_24"]]

    fixed_df = dfs["Section_18"]
    for section in sections_to_fix:
        df = dfs[section]  # Get the DataFrame

        # Process only the first row to find the stopping column
        first_row = df.iloc[0]  # First row

        # Initialize stop_col_idx as None (in case no stop condition is met)
        stop_col_idx = None
        key_value_column = None  # To keep track of the key column (32 or empty)

        # Find the column where value is "32" or empty string after column 7
        for col_idx in range(8, len(first_row)):  # Start checking from column 8 onward
            value = str(first_row[col_idx]).strip()

            if value == '"32"' or value == "":  # Stop when we hit "32" or empty string
                stop_col_idx = col_idx
                key_value_column = col_idx  # Store the key column
                break

        # If no stopping condition was found, use the last column as stop_col_idx
        if stop_col_idx is None:
            stop_col_idx = len(first_row)  # Use the last column

        # Merge columns from column 7 to the found stopping column (excluding the key column)
        for col_idx in range(8, stop_col_idx):  # Merge all columns between 7 and stop
            df.iloc[:, 7] = df.iloc[:, 7] + " " + df.iloc[:, col_idx]

            # Drop the merged column completely
        df.drop(df.columns[8:stop_col_idx], axis=1, inplace=True)
        # Save the fixed DataFrame back to the dictionary
        dfs[section] = df
        df.columns = range(df.shape[1])
        fixed_df = pd.concat([fixed_df, df], axis=0, ignore_index=True)

    # Load the stimulus data frame
    stimulus_df = dfs["Section_57"]

    fixed_df.columns = headers
    fixed_df.columns = fixed_df.columns.str.replace('"', '')
    fixed_df = fixed_df.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)
    fixed_df["trial_index"] = fixed_df["trial_index"].astype(int)
    fixed_df = fixed_df.sort_values(by="trial_index", ascending=True)

    more_info_stim_df = extract_stimulus_details(stimulus_df[50])

    clean_stimulus_df = pd.concat([stimulus_df[3], stimulus_df[50], more_info_stim_df], axis=1)

    # Convert 'Trial' column to integer for numerical operations
    clean_stimulus_df[3] = clean_stimulus_df[3].str.replace('"', '').astype(int)

    clean_stimulus_df.rename(columns={3: "stim_trial", 50: 'figure_name'}, inplace=True)

    # Shift 'trialNum' column and create the new 'stimTrial' column in response_df
    fixed_df['stim_trial'] = fixed_df['trial_index'] - 1

    # Merge the DataFrames on 'stim_trial' using a left join to preserve all rows from clean_stimulus_df
    comp_df = pd.merge(fixed_df, clean_stimulus_df, on='stim_trial', how='left')

    # arrange staircase part:
    comp_df = assign_staircase_info(comp_df)

    # arrange main part:
    comp_df = assign_main_part_info(comp_df)
    # extract the VVIQ2 answers:
    vviq_ans = dfs['Section_24']
    vviq_ans = vviq_ans.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)

    # Generate the new file name
    if participant_id:
        new_file_name_main = "sub_" + participant_id + "_cleaned_data.csv"
        new_file_name_vviq = f"sub_" + participant_id + "_VVIQ_data.csv"
    else:
        # If no participant ID is provided, try to extract it from the original file name
        match = re.search(r"sub_(\d{3})", path)  # Looks for "sub_XXX" where XXX is 3 digits
        if match:
            new_file_name_main = f"sub_{match.group(1)}_cleaned_data.csv"
            new_file_name_vviq = f"sub_{match.group(1)}_VVIQ_data.csv"
        else:
            # If no participant ID is found, give a generic name based on the original filename
            new_file_name_main = f"{os.path.splitext(os.path.basename(path))[0]}_cleaned_data.csv"
            new_file_name_vviq = f"sub_{os.path.splitext(os.path.basename(path))[0]}_VVIQ_data.csv"

    # Create the folder if it doesn't exist
    folder_name = 'clean_data'
    if not os.path.exists('clean_data'):
        os.makedirs(folder_name)

    # Define file paths
    fixed_df_path = os.path.join(folder_name, new_file_name_main)
    questionnaire_df_path = os.path.join(folder_name, new_file_name_vviq)

    # Save DataFrames as CSV
    comp_df.to_csv(fixed_df_path, index=False)
    vviq_ans.to_csv(questionnaire_df_path, index=False)

    print(f"Data saved in '{folder_name}' folder.")


def arrange_output_csv(path, participant_id=None):
    # Load csv data (assuming it's a string)
    with open(path, "r") as f:
        df = pd.read_csv(path)

    # Step 1: Find the section boundaries
    start_idx = df[df['stimulus'].astype(str).str.startswith("<br> This is block 1 out")].index[0]
    end_idx = df[df['stimulus'].astype(str).str.startswith("Thank you!. <br>In this final section")].index[0]

    # Step 2: Work only with this section
    section_df = df.loc[start_idx:end_idx].copy()

    # Step 3: Identify complete groups of 5 known test_part values
    required_parts = ['nan', 'fixation', 'main_test', 'conf_main', 'vividness_q']

    # Reset index for easier group tracking
    section_df = section_df.reset_index()

    # Container for final modified rows
    modified_rows = []

    i = 0
    while i <= len(section_df) - 5:
        window = section_df.iloc[i:i + 5]
        if sorted(window['test_part'].astype(str).tolist()) == sorted(required_parts):
            # store last main_row
            main_row = window[window['test_part'] == 'main_test'].iloc[0]
            prac_row = window[window['test_part'] == 'conf_main'].iloc[0]
            vivid_row = window[window['test_part'] == 'vividness_q'].iloc[0]

            for j in range(5):
                row = window.iloc[j].copy()

                # Copy from main_test row
                for col in ["stim_id", "set", "vis_level", "trial_within_block", "block_num",
                            "scene", "example", "congruency", "object_present", "imagine"]:
                    row[col] = main_row[col]

                # Copy from conf_main row
                for col in ["correct_response", "correct"]:
                    row[col] = prac_row[col]

                # Copy from vividness_q row
                row["slider_start"] = vivid_row["slider_start"]

                modified_rows.append(row)

            i += 5

        elif str(section_df.loc[i, 'test_part']) == 'ima_check':
            if 'main_row' not in locals():
                raise ValueError("'ima_check' row encountered before any 'main_test' row — can't copy values.")

            row = section_df.iloc[i].copy()
            for col in ["stim_id", "set", "vis_level", "trial_within_block", "block_num",
                        "scene", "example", "congruency", "object_present", "imagine"]:
                row[col] = main_row[col]

            modified_rows.append(row)
            i += 1

        else:
            modified_rows.append(section_df.iloc[i])
            i += 1

    # --- Trailing rows ---
    while i < len(section_df):
        row = section_df.iloc[i].copy()
        if str(row['test_part']) == 'ima_check' and 'main_row' in locals():
            for col in ["stim_id", "set", "vis_level", "trial_within_block", "block_num",
                        "scene", "example", "congruency", "object_present", "imagine"]:
                row[col] = main_row[col]
        modified_rows.append(row)
        i += 1

    # Convert back to DataFrame and restore original index
    modified_section_df = pd.DataFrame(modified_rows).set_index('index')

    # Step 4: Update original DataFrame with modified section
    df.update(modified_section_df)

    # Step 5: Extract object_y from stimulus and propagate it within blocks
    df["block_stim_id"] = None
    current_y = None
    block_count = 0

    for i, row in df.iterrows():
        stimulus = str(row.get("stimulus", ""))
        if "<br> This is block" in stimulus:
            match = re.search(r"obj_(\d+)", stimulus)
            if match:
                current_y = int(match.group(1))
                block_count += 1

        if block_count <= 16:
            df.at[i, "block_stim_id"] = current_y

        if block_count >= 16 and str(row.get("test_part")) == "ima_check":
            break

    df["block_stim_id"] = df["block_stim_id"].ffill()

    # Step 6: Also apply object_y extraction to the calibration block
    try:
        calib_start_idx = df[df["stimulus"].astype(str).str.startswith("We will now do a calibration block.")].index[0]
        calib_end_idx = \
        df[df["stimulus"].astype(str).str.startswith("During the experiment, we will sometimes ask you")].index[0]

        current_y = None
        for i in range(calib_start_idx, calib_end_idx + 1):
            stim = str(df.at[i, "stimulus"])
            if stim.startswith("You will be looking for the object below"):
                match = re.search(r"obj_(\d+)", stim)
                if match:
                    current_y = int(match.group(1))
            df.at[i, "block_stim_id"] = current_y

        # Forward-fill within the calibration section
        df.loc[calib_start_idx:calib_end_idx, "block_stim_id"] = df.loc[calib_start_idx:calib_end_idx,
                                                                  "block_stim_id"].ffill()

    except IndexError:
        print("Calibration section not found — skipping calibration object_y extraction.")


    # Generate the new file name
    if participant_id:
        new_file_name_main = "sub_" + participant_id + "_cleaned_data.csv"
    else:
        # If no participant ID is provided, try to extract it from the original file name
        match = re.search(r"sub_(\d{3})", path)  # Looks for "sub_XXX" where XXX is 3 digits
        if match:
            new_file_name_main = f"sub_{match.group(1)}_cleaned_data.csv"
        else:
            # If no participant ID is found, give a generic name based on the original filename
            new_file_name_main = f"{os.path.splitext(os.path.basename(path))[0]}_cleaned_data.csv"

    # Create the folder if it doesn't exist
    folder_name = 'clean_data'
    if not os.path.exists('clean_data'):
        os.makedirs(folder_name)

    # Define file paths
    fixed_df_path = os.path.join(folder_name, new_file_name_main)

    # Save DataFrames as CSV
    df.to_csv(fixed_df_path, index=False)

    print(f"Data saved in '{folder_name}' folder.")


def clean_txt_files_in_folder(folder_path):
    """
        Clean all .txt files in the given folder and save them with a modified name.
        The participant number will be included in the new file name.
        """
    # Get all the .txt files in the folder

    for folder_name in os.listdir(folder_path):
        f = os.path.join(folder_path, folder_name)
        # checking if it is a dir
        if os.path.isdir(f) and folder_name.lower().startswith('sub'):
            print(f)
            sub_num = folder_name[4:]
            for filename in os.listdir(f):
                if filename.endswith(".txt"):
                    file_path = os.path.join(f, filename)
                    # Use participant_counter to assign the participant ID
                    arrange_output_csv(file_path)


if __name__ == '__main__':
    path = r'' # put here the path for the folder of the raw data.
    # Should be a folder contain separated folders for each participant with 'data'.txt file.

    clean_txt_files_in_folder(path)
