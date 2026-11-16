import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os

os.environ["R_HOME"] = r"C:\PROGRA~1\R\R-45~1.2"  # your R path

import pandas as pd
import numpy as np
import seaborn as sns
import promptlib
import matplotlib.pyplot as plt
import configuration as cnfg
import scipy.stats as stats
from scipy.stats import ttest_rel
from scipy.stats import sem
import statsmodels.formula.api as smf
import itertools
from statsmodels.stats.anova import AnovaRM
import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
matplotlib.use("TkAgg")  # or "Qt5Agg"


### FUNCTIONS ###

def one_dataframe():
    """
    Load all preprocessed participant data files from a selected folder,
    combine them into a single DataFrame, and add a 'Subject' column
    to identify the participant associated with each row.

    Returns
    -------
    pandas.DataFrame
        A combined DataFrame containing the data from all participants,
        with an additional 'Subject' column indicating participant IDs.
    """

    prompter = promptlib.Files()
    dir = prompter.dir()
    dir = ''.join([dir, r'\\'])[:-1]

    file_list = [dir + f for f in os.listdir(dir) if f.startswith('sub') and 'cleaned_data' in f]
    csv_list = []

    for file in sorted(file_list):
        df = pd.read_csv(file)
        subj_id = os.path.basename(file).split('_')[:2]  # Gets "sub_XXX"
        df['Subject'] = subj_id[1]  # Add subject column
        csv_list.append(df)

    return pd.concat(csv_list, ignore_index=True)


def combined_dataframe(df1, df2, df3=[]):

    """
    Combine data from two or three experiments into a single DataFrame.
    Each dataset is assigned an experiment identifier ('Experiment 1',
    'Experiment 2', or 'Experiment 3') before being concatenated.

    Subject IDs are formatted consistently and updated to include their
    corresponding experiment identifier, ensuring that participants from
    different experiments can be distinguished.

    Parameters
    ----------
    df1 : pandas.DataFrame or str
        DataFrame or path to the data file for Experiment 1.
    df2 : pandas.DataFrame or str
        DataFrame or path to the data file for Experiment 2.
    df3 : pandas.DataFrame or optional
        Optional DataFrame containing data for Experiment 3.

    Returns
    -------
    pandas.DataFrame
        A combined DataFrame containing data from all provided experiments,
        with additional 'Experiment' and reformatted 'Subject' columns.
    """

    if isinstance(df1, pd.DataFrame):
        df1 = df1
    else:
        df1 = pd.read_csv(df1)

    if isinstance(df2, pd.DataFrame):
        df2 = df2
    else:
        df2 = pd.read_csv(df2)

    # Assign a unique experiment ID to each dataframe
    df1['Experiment'] = 'Experiment 1'
    df2['Experiment'] = 'Experiment 2'
    if isinstance(df3, pd.DataFrame):
        df3['Experiment'] = 'Experiment 3'
        third_df = True
    else:
        third_df = False

    # Concatenate the dataframes vertically
    if third_df:
        combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    else:
        combined_df = pd.concat([df1, df2], ignore_index=True)

    # Propagate subject values until a new subject is encountered
    combined_df['Subject'] = (
        combined_df.groupby('Experiment')['Subject']
            .ffill()
            .apply(lambda x: f'{x:03}')
            .astype(str)
            .radd('Subject ')
            .radd(combined_df['Experiment'] + ' ')
    )

    return combined_df


def calculate_accuracy(df):

    """
    Calculate the percentage of correct responses for each participant,
    object, and staircase (block).

    The data are grouped by 'Subject', 'block_stim_id', and 'block_num'.
    For each group, accuracy is calculated as the proportion of correct
    responses out of the total number of trials, expressed as a percentage.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing trial-level data, including subject IDs,
        object codes, staircase numbers, and correctness values.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per participant, object, and staircase,
        with the corresponding accuracy percentage.
    """

    accuracy_list = []

    for (subject, obj_code, staircase_number), group in df.groupby(['Subject', 'block_stim_id', 'block_num']):
        correct_responses = group['correct'].sum()
        total_trials = len(group)

        accuracy = (correct_responses / total_trials) * 100
        accuracy_list.append({
            'Subject': subject,
            'block_stim_id': obj_code,
            'block_num': staircase_number,
            'accuracy': accuracy
        })
    accuracy_df = pd.DataFrame(accuracy_list)
    return accuracy_df


def zscore_normalize_by_condition(df, condition_values):
    """
    Calculate z-score normalized response values separately for each
    participant and specified experimental condition.

    For each participant and condition, response values are standardized
    by subtracting the condition-specific mean and dividing by the
    condition-specific standard deviation. The normalized values are
    stored in a new column, 'normalized_response', without overwriting
    the original response values.

    If all response values within a participant-condition group are
    identical (i.e., the standard deviation is zero), the normalized
    values are set to 0.0.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing participant IDs, experimental conditions,
        and response values.
    condition_values : list
        List of experimental condition values in the 'imagine' column
        for which responses should be normalized.

    Returns
    -------
    pandas.DataFrame
        DataFrame with an additional 'normalized_response' column
        containing the condition-specific z-scores.
    """

    subjects = df['Subject'].unique()
    for sub in subjects:
        for condition in condition_values:
            mask = (df['imagine'] == condition) & (df['Subject'] == sub)
            x = df.loc[mask, 'response'].astype(float)

            if not x.empty and x.std(ddof=0) != 0:
                z = (x - x.mean()) / x.std(ddof=0)
                df.loc[mask, 'normalized_response'] = z
            elif not x.empty:
                df.loc[mask, 'normalized_response'] = 0.0  # if all values are the same


def analyze_staircase_acc_group(data):

    """
    Analyze and visualize staircase performance across participants
    and objects by plotting visibility levels and response accuracy.

    The function accepts either a file path or a pandas DataFrame,
    filters the data to include only staircase trials, and extracts
    the visibility level and accuracy for each participant, object,
    and staircase.

    For each object, two plots are generated:
        1. Visibility levels across staircases, showing individual
           participant trajectories and the group average.
        2. Response accuracy across staircases, showing individual
           participant trajectories and the group average, alongside
           a reference line indicating the 70% accuracy threshold.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to a data file or a DataFrame containing trial-level data,
        including staircase information, visibility levels, and
        response accuracy.

    Returns
    -------
    None
        Displays visibility and accuracy plots for each object.
    """

    # Handle file path or DataFrame input
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a file path or a DataFrame")

    # Filter for staircase trials
    staircase_df = df[df['test_part'] == "stair_test"]

    # Ensure object code is string
    staircase_df = staircase_df.copy()
    staircase_df['block_stim_id'] = staircase_df['block_stim_id'].astype(str)

    # Get visibility levels (first per staircase per subject)
    vis_df = staircase_df.groupby(['Subject', 'block_stim_id', 'block_num'])['vis_level'].first().reset_index()

    # Get accuracy (external function)
    accuracy_df = calculate_accuracy(staircase_df)

    unique_objects = vis_df['block_stim_id'].unique()
    fig, axes = plt.subplots(len(unique_objects), 2, figsize=(14, len(unique_objects) * 6))

    if len(unique_objects) == 1:
        axes = [axes]  # allow indexing when only one object

    for i, obj in enumerate(unique_objects):
        # Data for current object
        vis_obj = vis_df[vis_df['block_stim_id'] == obj]
        acc_obj = accuracy_df[accuracy_df['block_stim_id'] == obj]
        vis_avg = vis_df[vis_df['block_stim_id'] == obj]
        acc_avg = accuracy_df[accuracy_df['block_stim_id'] == obj]

        for i, obj in enumerate(unique_objects):
            # Data for current object
            vis_obj = vis_df[vis_df['block_stim_id'] == obj]
            acc_obj = accuracy_df[accuracy_df['block_stim_id'] == obj]
            vis_avg = vis_df[vis_df['block_stim_id'] == obj]
            acc_avg = accuracy_df[accuracy_df['block_stim_id'] == obj]

            # ========== Plot Visibility ==========
            ax1 = axes[i][0]
            sns.lineplot(data=vis_avg, x='block_num', y='vis_level',
                         marker='o', ax=ax1, color='black', label='Group Avg')
            # Individual subject lines
            sns.lineplot(
                data=vis_obj,
                x='block_num', y='vis_level',
                hue='Subject',  # 🎨 Automatically gives each subject a unique color
                ax=ax1,
                alpha=0.7,
                linewidth=1
            )

            ax1.set_title(f'Visibility Level - Object {obj}')
            ax1.set_xlabel('Staircase Number')
            ax1.set_ylabel('Vis Level')
            ax1.set_xticks(range(1, 13))
            # ax1.set_ylim(0, 110)

            # ========== Plot Accuracy ==========
            ax2 = axes[i][1]
            sns.lineplot(data=acc_avg, x='block_num', y='accuracy',
                         marker='o', ax=ax2, color='black', label='Group Avg')
            sns.lineplot(
                data=acc_obj,
                x='block_num', y='accuracy',
                hue='Subject',
                ax=ax2,
                alpha=0.7,
                linewidth=1
            )

            ax2.set_title(f'Accuracy - Object {obj}')
            ax2.set_xlabel('Staircase Number')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_xticks(range(1, 13))
            ax2.set_ylim(0, 110)
            ax2.axhline(y=70, color='r', linestyle='--', label='70% Threshold')
            ax2.legend()

        plt.tight_layout()
        plt.show()


def check_exclusions_acc_and_ima_check(data):
    """
    Identify participants who do not meet the predefined accuracy
    and imagery-check criteria.

    Participants are assessed based on their overall accuracy in
    perception-only main-task trials (acceptable range: 55–95%)
    and the number of incorrect imagery-check responses in each
    task condition (maximum of 2 incorrect responses per condition).

    The function generates a participant-level exclusion summary,
    including exclusion status and reasons, and saves it as a CSV file.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to a data file or a DataFrame containing participant
        responses, accuracy, and imagery-check information.

    Returns
    -------
    None
        Prints participant-level exclusion information and overall
        exclusion statistics, and saves the exclusion summary
        as 'subject_exclusion_summary.csv'.
    """

    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a file path or a DataFrame")

    # find invalid blocks within the main task
    check_df = df[df["stimulus"].str.startswith(cnfg.pattern_check_cond_block, na=False)]

    check_df = (
        check_df.groupby(['Subject', 'imagine'])['correct']
            .sum()
            .reset_index()
    )

    # Filtering by stair_test in the 'test_part' column
    main_df = df[(df['test_part'] == "main_test")]

    # Filter rows where 'imagine' contains False values
    perc_df = main_df[main_df['imagine'] == False]

    individual_all_general = (
        perc_df.groupby(['Subject'])['correct']
            .mean()
            .reset_index()
    )

    subject_summary = []

    for subject_id, subject_df in df.groupby('Subject'):
        reasons = []
        excluded = False

        # Criterion 1: Accuracy
        now_sub_acc = individual_all_general.loc[
            individual_all_general["Subject"] == subject_id, "correct"].reset_index(drop=True)
        msg1 = f"Accuracy for sub_{subject_id} : accuracy = {now_sub_acc[0]:.2f}"
        print(msg1)
        if now_sub_acc[0] < 0.55 or now_sub_acc[0] > 0.95:
            msg = f"Accuracy out of bounds for participant sub_{subject_id} (accuracy = {now_sub_acc[0]:.2f})"
            print(msg)
            reasons.append(f"accuracy out of bounds ({now_sub_acc[0]:.2f})")
            excluded = True

        # Criterion 2: ima_checks false count
        now_sub_ima_checks = check_df.loc[check_df["Subject"] == subject_id, "correct"].reset_index(drop=True)
        correct_ima_check_overall = now_sub_ima_checks[0] + now_sub_ima_checks[1]
        msg2 = f"false ima checks for sub_{subject_id} : n = {12 - correct_ima_check_overall}, perc only: n = {6 - now_sub_ima_checks[0]}, and imagine + perc: n = {6 - now_sub_ima_checks[1]}"
        print(msg2)
        if now_sub_ima_checks[0] < 4:
            msg = f"Too many false ima checks in perc only for sub_{subject_id} (n = {6 - now_sub_ima_checks[0]})"
            print(msg)
            reasons.append(f"too many false checks in perc only (n = {6 - now_sub_ima_checks[0]})")
            excluded = True
        if now_sub_ima_checks[1] < 4:
            msg = f"Too many false ima checks in detect + imagine for sub_{subject_id} (n = {6 - now_sub_ima_checks[1]})"
            print(msg)
            reasons.append(f"too many false checks in detect + imagine (n = {6 - now_sub_ima_checks[1]})")
            excluded = True

        subject_summary.append({
            "Subject": 'sub_' + subject_id,
            "excluded": excluded,
            "reason": "; ".join(reasons) if reasons else "passed"
        })

    result_df = pd.DataFrame(subject_summary)

    # Calculate stats
    total_participants = len(result_df)
    num_passed = (result_df["excluded"] == False).sum()
    num_excluded = total_participants - num_passed
    perc_excluded = (num_excluded / total_participants) * 100

    # Add stats to every row
    result_df["total_passed"] = num_passed
    result_df["perc_excluded"] = f"{perc_excluded:.2f}%"

    result_df.to_csv("subject_exclusion_summary.csv", index=False)

    print(f"\nSummary: {num_passed} participants passed, "
          f"{num_excluded} excluded ({perc_excluded:.2f}%).")

    result_df.to_csv("subject_exclusion_summary.csv", index=False)


def anova_positive_ans(data):

    """
    Perform a three-way repeated-measures ANOVA on the proportion of
    positive answers across imagery, congruency, and object presence
    conditions.

    The function accepts either a file path or a pandas DataFrame.
    It first excludes invalid blocks from the main task based on
    participants' performance on condition-check trials.

    It then filters the data to include only main-task trials and
    calculates the proportion of positive answers for each participant
    across the combinations of imagery, congruency, and object presence.

    A three-way repeated-measures ANOVA is then performed to examine
    the effects of imagery, congruency, and object presence, including
    their interactions.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to a data file or a DataFrame containing trial-level data,
        including participant IDs, condition information, and response
        accuracy.

    Returns
    -------
    None
        Prints the repeated-measures ANOVA results.
    """

    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a file path or a DataFrame")

    # Exclude invalid blocks within the main task
    check_df = df[df["stimulus"].str.startswith(cnfg.pattern_check_cond_block, na=False)]
    to_exclude_list = [(row.Subject, row.block_num) for row in check_df.loc[df["correct"] == False].itertuples()]

    # Use a boolean mask to filter out rows that match any of those tuples
    df = df[~df[["Subject", "block_num"]].apply(tuple, axis=1).isin(to_exclude_list)]

    # Filter for main test trials
    main_df = df[df['test_part'] == 'main_test'].copy()

    # All positive answers
    main_df["positive_answer"] = (
            ((main_df["object_present"] == 1) & (main_df["correct"] == 1)) |  # Hits
            ((main_df["object_present"] == 0) & (main_df["correct"] == 0))  # False alarms
    )

    # Compute per-subject hit rate (object present)
    prop_positive_answers = (
        main_df
            .groupby(["Subject", "congruency", "imagine", "object_present"])["positive_answer"]
            .mean()
            .reset_index()
    )

    anova_PA = AnovaRM(
        data=prop_positive_answers,
        depvar="positive_answer",  # DV
        subject="Subject",  # Subject ID
        within=["imagine", "congruency", "object_present"]
    ).fit()

    print(anova_PA)


def analyze_general_cong_Q_diff_bet_exp_collapsed(data, extract_df=False):
    """
    Calculate within-participant differences in congruency ratings
    (congruent minus incongruent), collapsed across object types,
    and compare these differences between experiments.

    The function filters congruency-question responses, classifies
    scenes as congruent or incongruent based on their filenames,
    and calculates participant-level mean ratings for each condition.

    If extract_df is True, returns a DataFrame containing participant-
    level congruency ratings, their difference, and the average rating
    across all scenes.

    Otherwise, calculates experiment-level mean differences, performs
    an independent-samples Welch's t-test between two experiments,
    and displays a plot of the results with 95% confidence intervals
    and individual participant data points.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to a data file or a DataFrame containing congruency-question
        responses and participant and experiment information.
    extract_df : bool, optional
        If True, returns the participant-level DataFrame instead of
        performing the between-experiment comparison. Default is False.

    Returns
    -------
    pandas.DataFrame or None
        Returns participant-level results if extract_df is True.
        Otherwise, prints the statistical results and displays the plot.
    """

    # Load data
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a file path or a DataFrame")

    # Filter relevant rows
    cong_Q_df = df[df["test_part"] == 'cong_questions'].copy()
    cong_Q_df["response"] = pd.to_numeric(cong_Q_df["response"], errors='coerce')

    # Extract filename parts
    cong_Q_df["img_filename"] = cong_Q_df["stimulus"].str.extract(r"src='img/([^']+)\.png'")
    cong_Q_df["set"] = cong_Q_df["img_filename"].str.extract(r'(set_\d+)')
    cong_Q_df["example"] = cong_Q_df["img_filename"].str.extract(r'example_(\d+)').astype(int)

    # Define congruency group (ignore object type)
    def assign_group(row):
        if row["set"] == "set_1":
            return "cong" if 1 <= row["example"] <= 6 else "incong"
        elif row["set"] == "set_2":
            return "cong" if 1 <= row["example"] <= 6 else "incong"
        else:
            return None

    cong_Q_df["congruency"] = cong_Q_df.apply(assign_group, axis=1)
    cong_Q_df = cong_Q_df[cong_Q_df["congruency"].notna()]

    # ---- Subject-level means ----
    if extract_df:
        subj_group_means = (
            cong_Q_df.groupby(["Subject", "congruency"], as_index=False)["response"]  #
                .mean()
        )
        subj_means = (
            cong_Q_df.groupby(["Subject"], as_index=False)["response"]  #
                .mean())
        subj_means = subj_means.rename(columns={'response': 'average_cong_rating'})

    else:
        subj_group_means = (
            cong_Q_df.groupby(["Subject", "congruency", "Experiment"], as_index=False)["response"]
                .mean()
        )

    # ---- Compute within-subject difference (cong - incong) ----
    if extract_df:
        pivot_df = subj_group_means.pivot(
            index=["Subject"],  # , "Experiment"
            columns="congruency",
            values="response"
        ).reset_index()
    else:
        pivot_df = subj_group_means.pivot(
            index=["Subject", "Experiment"],
            columns="congruency",
            values="response"
        ).reset_index()

    pivot_df["cong_diff"] = pivot_df["cong"] - pivot_df["incong"]

    if extract_df:
        pivot_df = pivot_df.merge(subj_means, on="Subject", how="left")
        return pivot_df

    # ---- Compute experiment-level means ----
    exp_means = pivot_df.groupby("Experiment", as_index=False)["cong_diff"].mean()

    print("\n=== Experiment-level mean congruency difference (collapsed) ===")
    print(exp_means)

    # ---- Independent-samples t-test ----
    exps = pivot_df["Experiment"].unique()
    if len(exps) == 2:
        exp1, exp2 = exps
        t_val, p_val = stats.ttest_ind(
            pivot_df.loc[pivot_df["Experiment"] == exp1, "cong_diff"].dropna(),
            pivot_df.loc[pivot_df["Experiment"] == exp2, "cong_diff"].dropna(),
            equal_var=False
        )
        print(f"\n=== Independent-samples t-test ===")
        print(f"{exp1} vs {exp2} → t = {t_val:.3f}, p = {p_val:.4f}")
    else:
        print("⚠️ Need exactly 2 experiments for between-experiment t-test.")

    # ---- Plot ----
    sns.set(style="white", font_scale=1.4)
    plt.figure(figsize=(5, 5))
    ax = sns.barplot(
        data=pivot_df,
        x="Experiment",
        y="cong_diff",
        errorbar=("ci", 95),
        capsize=0.2,
        errwidth=1.5,
        palette="Set2"
    )
    sns.stripplot(
        data=pivot_df,
        x="Experiment",
        y="cong_diff",
        color="black",
        alpha=0.4,
        jitter=0.1
    )

    ax.set_title("Congruency Difference (Cong - Incong, collapsed)")
    ax.set_ylabel("Mean Rating Difference")
    sns.despine()
    plt.tight_layout()
    plt.show()


def export_GLMM_with_scenes_and_diff_cong(data, output_path):

    """
    Prepare and export a dataset for generalized linear mixed-effects
    model (GLMM) analysis, incorporating scene-level information,
    participant-specific congruency ratings, and normalized vividness
    responses.

    The function accepts either a file path or a pandas DataFrame.
    It first excludes invalid participant blocks based on performance
    on condition-check trials.

    Vividness responses are extracted and z-score normalized separately
    for each participant and imagery condition. The normalized values
    are then merged with the main-task data.

    The dataset is cleaned by converting categorical and
    numerical variables to their appropriate data types, removing rows
    with missing values, and exporting the resulting dataset as a CSV
    file for use in R (lme4) GLMM analysis.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to the input data file or a DataFrame containing
        participant-level and trial-level experimental data.
    output_path : str
        Path where the prepared CSV file for GLMM analysis will be saved.

    Returns
    -------
    None
        Exports the prepared GLMM dataset to the specified output path
        and prints the location of the exported file.
    """


    # --- Load ---
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a file path or a DataFrame")

    # --- Exclusions ---
    check_df = df[df["stimulus"].str.startswith(cnfg.pattern_check_cond_block, na=False)]
    to_exclude_list = [(row.Subject, row.block_num)
                       for row in check_df.loc[df["correct"] == False].itertuples()]
    df = df[~df[["Subject", "block_num"]].apply(tuple, axis=1).isin(to_exclude_list)]

    # Extract vividness
    vividness_df = df[(df['test_part'] == 'vividness_q')].copy()

    # Positive answer
    vividness_df["positive_answer"] = (
            ((vividness_df["object_present"] == 1) & (vividness_df["correct"] == 1)) |
            ((vividness_df["object_present"] == 0) & (vividness_df["correct"] == 0))
    )

    # normalized_response column assumed to be computed
    vividness_df['normalized_response'] = np.nan

    # Conditions you want to normalize (others will remain NaN in normalized_response)
    conditions_to_normalize_ima = [True]
    conditions_to_normalize_perc = [False]

    # Apply normalization
    zscore_normalize_by_condition(vividness_df, conditions_to_normalize_ima)
    zscore_normalize_by_condition(vividness_df, conditions_to_normalize_perc)

    vividness_df = vividness_df.rename(columns={"normalized_response": "vividness_z_scored_response"})
    vividness_df = vividness_df[["Subject", "trial_within_block", "block_num", "vividness_z_scored_response"]]

    # --- Filter to main task ---
    main_df = df[df['test_part'] == 'main_test'].copy()

    # --- DV: positive answer (0/1) ---
    main_df["positive_answer"] = (
            ((main_df["object_present"] == 1) & (main_df["correct"] == 1)) |  # hits
            ((main_df["object_present"] == 0) & (main_df["correct"] == 0))  # correct rejections
    ).astype(int)

    # --- Make factors categorical with readable labels ---
    main_df["imagine"] = main_df["imagine"].map({True: "imag", False: "det"})
    main_df["congruency"] = main_df["congruency"].map({0: "cong", 1: "incong"})
    main_df["object_present"] = main_df["object_present"].map({0: "absent", 1: "present"})
    main_df["Subject"] = main_df["Subject"]
    main_df['example'] = main_df['example'].astype(int)
    main_df['block_stim_id'] = main_df['block_stim_id'].astype(int)
    main_df['scene'] = main_df['scene'].astype(int)
    main_df["scene_example"] = main_df[["scene", "example"]].astype(str).agg("_".join, axis=1)

    # --- Load and merge subject-level congruency difference ---
    cong_df = analyze_general_cong_Q_diff_bet_exp_collapsed(data, extract_df=True)[["Subject", "cong_diff", "average_cong_rating"]]
    main_df = main_df.merge(cong_df, on="Subject", how="left")
    main_df = main_df.merge(
        vividness_df,
        on=["Subject", "trial_within_block", "block_num"],
        how="left",
    )

    # Ensure numeric column
    main_df["cong_diff"] = pd.to_numeric(main_df["cong_diff"], errors='coerce')
    main_df["average_cong_rating"] = pd.to_numeric(main_df["average_cong_rating"], errors='coerce')

    # --- GLMM (pymer4 / R lme4) ---
    cat_cols = ["imagine", "congruency", "object_present", "scene_example", "Subject"]
    num_cols = ["cong_diff", "average_cong_rating", "positive_answer", "vividness_z_scored_response"]

    # Clean the data
    glmm_df = main_df[cat_cols + num_cols].copy()

    for col in cat_cols:
        glmm_df[col] = glmm_df[col].fillna("NA").astype(str)

    for col in num_cols:
        glmm_df[col] = pd.to_numeric(glmm_df[col], errors="coerce")

    glmm_df = glmm_df.dropna()
    glmm_df = glmm_df.reset_index(drop=True)

    # --- Export to CSV for R ---
    glmm_df.to_csv(output_path, index=False)
    print(f"Data exported to: {output_path}")


def lmm_visibility_vividness(data, run_ttest=True):

    """
    Examine the relationship between perceptual visibility, detection responses,
    scene congruency, and object presence using a linear
    mixed-effects model (LMM), restricted to perception trials.

    The function accepts either a file path or a pandas DataFrame.
    It first excludes invalid participant blocks based on performance
    on condition-check trials.

    Visibility responses are extracted from trials in the visibility
    rating part of the task and converted to numeric values. The
    responses are then z-score normalized separately for each
    participant within the perception condition.

    A linear mixed-effects model is fitted to examine the effects of
    response accuracy, scene congruency, and object presence on
    normalized visibility responses, including all interactions.
    Participant is included as a random intercept, and categorical
    predictors are coded using sum contrasts.

    If run_ttest is True, an additional paired-samples t-test is
    performed to compare mean visibility between trials with correct
    and incorrect responses, using participant-level averages.

    Finally, the function:
        - Calculates participant-level mean visibility for each
          combination of response accuracy, congruency, and object
          presence.
        - Computes group-level observed means and standard errors.
        - Generates model-predicted visibility values for all
          combinations of the experimental conditions.
        - Prints the model summary, optional t-test results,
          observed descriptive statistics, and predicted values.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to a data file or a DataFrame containing trial-level
        experimental data, including visibility responses, participant
        IDs, and experimental condition information.
    run_ttest : bool, optional
        If True, performs a paired-samples t-test comparing mean
        visibility between correct and incorrect responses.
        Default is True.

    Returns
    -------
    None
        Prints the mixed-effects model summary, optional paired-samples
        t-test results, observed descriptive statistics, and
        model-predicted values.
    """

    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a file path or a DataFrame")

    # Exclude invalid blocks
    check_df = df[df["stimulus"].str.startswith(cnfg.pattern_check_cond_block, na=False)]
    to_exclude_list = [(row.Subject, row.block_num) for row in check_df.loc[df["correct"] == False].itertuples()]
    df = df[~df[["Subject", "block_num"]].apply(tuple, axis=1).isin(to_exclude_list)]

    # Only perception trials
    vividness_df = df[(df['test_part'] == 'vividness_q') & (df['imagine'] == False)].copy()

    # Positive answer
    vividness_df["positive_answer"] = (
            ((vividness_df["object_present"] == 1) & (vividness_df["correct"] == 1)) |
            ((vividness_df["object_present"] == 0) & (vividness_df["correct"] == 0))
    )

    # normalized_response column assumed to be computed
    vividness_df['normalized_response'] = np.nan

    # Conditions you want to normalize (others will remain NaN in normalized_response)
    conditions_to_normalize_perc = [False]
    # Apply normalization
    zscore_normalize_by_condition(vividness_df, conditions_to_normalize_perc)

    vividness_perc_df = vividness_df[df['imagine'] == False]

    # Convert to categorical with sum coding
    # For sum coding in statsmodels, we can use C(var, Sum)
    # +0.5 / -0.5 coding is automatically done
    vividness_perc_df['positive_answer'] = pd.Categorical(
        vividness_perc_df['positive_answer'].map({False: "no", True: "yes"}), ordered=False)
    vividness_perc_df['congruency'] = pd.Categorical(vividness_perc_df['congruency'].map({0: "cong", 1: "incong"}),
                                                    ordered=False)
    vividness_perc_df['object_present'] = pd.Categorical(vividness_perc_df['object_present'].map({0: "absent", 1: "present"}),
                                                    ordered=False)

    # Fit LMM using sum coding
    model = smf.mixedlm(
        "normalized_response ~ C(congruency, Sum) * C(positive_answer, Sum) * C(object_present, Sum)",
        data=vividness_perc_df,
        groups=vividness_perc_df["Subject"],
        re_formula="1"
    )

    lmm_result = model.fit(method="lbfgs")
    print(lmm_result.summary())

    # --- (2) Paired-samples t-test ---
    if run_ttest:
        print("\n=== Paired-samples t-test (Visibility trials only) ===")

        # Compute subject-level mean vividness per category
        subject_means = vividness_perc_df.groupby(["Subject", "positive_answer"])["normalized_response"].mean().unstack()
        # Ensure both columns are present
        if set(["yes", "no"]).issubset(subject_means.columns):
            t_stat, p_val = ttest_rel(subject_means["yes"], subject_means["no"], nan_policy='omit')
            print(f"T({len(subject_means) - 1}) = {t_stat:.3f}, p = {p_val:.4f}")

    # Step 1: Compute subject-level means for each condition
    subject_means = (
        vividness_perc_df
            .groupby(["Subject", "positive_answer", "congruency", "object_present"])["normalized_response"]
            .mean()
            .reset_index()
    )

    # Step 2: Compute group-level means (across subjects)
    observed_summary = (
        subject_means
            .groupby(["positive_answer", "congruency", "object_present"])["normalized_response"]
            .agg(["mean", sem])
            .reset_index()
            .rename(columns={"mean": "ObservedMean", "sem": "SEM"})
    )

    print(observed_summary)

    # Define all combinations of fixed effects
    predict_df = pd.DataFrame(
        itertools.product(
            ["yes", "no"],
            ["cong", "incong"],
            ["present", "absent"]
        ),
        columns=["positive_answer", "congruency", "object_present"]
    )

    predict_df["Predicted"] = lmm_result.predict(predict_df)
    print(predict_df)


def lmm_imagine_vividness(data, run_ttest=True):

    """
    Examine the relationship between imagery vividness, detection response,
    scene congruency, and object presence using a linear
    mixed-effects model (LMM), restricted to imagery trials.

    The function accepts either a file path or a pandas DataFrame.
    It first excludes invalid participant blocks based on performance
    on condition-check trials.

    Vividness responses are extracted from imagery trials and converted
    to numeric values. The responses are then z-score normalized
    separately for each participant and imagery condition.

    A linear mixed-effects model is fitted to examine the effects of
    positive detection answers, scene congruency, and object presence on
    normalized vividness responses, including all interactions.
    Participant is included as a random intercept, and categorical
    predictors are coded using sum contrasts.

    If run_ttest is True, an additional paired-samples t-test is
    performed to compare mean vividness between trials with positive
    and negative answers, using participant-level averages.

    Finally, the function:
        - Calculates participant-level mean vividness for each
          combination of response, congruency, and object presence.
        - Computes group-level observed means and standard errors.
        - Generates model-predicted vividness values for all combinations
          of the experimental conditions.
        - Prints the model summary, optional t-test results,
          observed descriptive statistics, and predicted values.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to a data file or a DataFrame containing trial-level
        experimental data, including vividness responses, participant
        IDs, and experimental condition information.
    run_ttest : bool, optional
        If True, performs a paired-samples t-test comparing mean
        vividness between positive and negative answers. Default is True.

    Returns
    -------
    None
        Prints the mixed-effects model summary, optional paired-samples
        t-test results, observed descriptive statistics, and
        model-predicted values.
    """

    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a file path or a DataFrame")

    # Exclude invalid blocks
    check_df = df[df["stimulus"].str.startswith(cnfg.pattern_check_cond_block, na=False)]
    to_exclude_list = [(row.Subject, row.block_num) for row in check_df.loc[df["correct"] == False].itertuples()]
    df = df[~df[["Subject", "block_num"]].apply(tuple, axis=1).isin(to_exclude_list)]

    # Only imagine trials
    vividness_df = df[(df['test_part'] == 'vividness_q') & (df['imagine'] == True)].copy()

    vividness_df["response"] = pd.to_numeric(
        vividness_df["response"], errors="coerce"
    )

    # Positive answer
    vividness_df["positive_answer"] = (
            ((vividness_df["object_present"] == 1) & (vividness_df["correct"] == 1)) |
            ((vividness_df["object_present"] == 0) & (vividness_df["correct"] == 0))
    )

    # normalized_response column assumed to be computed
    vividness_df['normalized_response'] = np.nan

    # Conditions you want to normalize (others will remain NaN in normalized_response)
    conditions_to_normalize_ima = [True]

    # Apply normalization
    zscore_normalize_by_condition(vividness_df, conditions_to_normalize_ima)

    vividness_ima_df = vividness_df[df['imagine'] == True]

    # Convert to categorical with sum coding
    # For sum coding in statsmodels, we can use C(var, Sum)
    # +0.5 / -0.5 coding is automatically done
    vividness_ima_df['positive_answer'] = pd.Categorical(
        vividness_ima_df['positive_answer'].map({False: "no", True: "yes"}), ordered=False)
    vividness_ima_df['congruency'] = pd.Categorical(vividness_ima_df['congruency'].map({0: "cong", 1: "incong"}),
                                                    ordered=False)
    vividness_ima_df['object_present'] = pd.Categorical(vividness_ima_df['object_present'].map({0: "absent", 1: "present"}),
                                                    ordered=False)

    # Fit LMM using sum coding
    model = smf.mixedlm(
        "normalized_response ~ "
        "C(positive_answer, Sum) * C(congruency, Sum) * C(object_present, Sum)",
        data=vividness_ima_df,
        groups=vividness_ima_df["Subject"],
        re_formula="1"
    )

    lmm_result = model.fit(method="lbfgs")
    print(lmm_result.summary())
    # --- (2) Paired-samples t-test ---
    if run_ttest:
        print("\n=== Paired-samples t-test (Imagery trials only) ===")

        # Compute subject-level mean vividness per category
        subject_means = vividness_ima_df.groupby(["Subject", "positive_answer"])["normalized_response"].mean().unstack()
        # Ensure both columns are present
        if set(["yes", "no"]).issubset(subject_means.columns):
            t_stat, p_val = ttest_rel(subject_means["yes"], subject_means["no"], nan_policy='omit')
            print(f"T({len(subject_means) - 1}) = {t_stat:.3f}, p = {p_val:.4f}")

    # Step 1: Compute subject-level means for each condition
    subject_means = (
        vividness_ima_df
            .groupby(["Subject", "positive_answer", "congruency", "object_present"])["normalized_response"]
            .mean()
            .reset_index()
    )

    # Step 2: Compute group-level means (across subjects)
    observed_summary = (
        subject_means
            .groupby(["positive_answer", "congruency", "object_present"])["normalized_response"]
            .agg(["mean", sem])
            .reset_index()
            .rename(columns={"mean": "ObservedMean", "sem": "SEM"})
    )

    print(observed_summary)

    # Define all combinations of fixed effects
    predict_df = pd.DataFrame(
        itertools.product(
            ["yes", "no"],
            ["cong", "incong"],
            ["present", "absent"]
        ),
        columns=["positive_answer", "congruency", "object_present"]
    )

    predict_df["Predicted"] = lmm_result.predict(predict_df)
    print(predict_df)


def analyze_general_cong_Q_cong_incong_bet_exp(data):
    """
    Analyze congruency ratings across two experiments by comparing
    participant responses for congruent and incongruent scenes
    separately using independent-samples t-tests.

    The function accepts either a file path or a pandas DataFrame.
    It first filters the data to include only responses from the
    congruency-question part of the task and converts responses
    to numeric values.

    Scene filenames are used to extract the stimulus set and example
    number, which are then used to classify scenes as congruent or
    incongruent, irrespective of object type.

    Participant-level mean ratings are calculated separately for
    each experiment and congruency condition. Experiment-level means
    are then computed and printed.

    If exactly two experiments are present, independent-samples
    Welch's t-tests are performed to compare the experiments separately
    for congruent and incongruent scenes.

    Finally, the function generates and saves two visualizations:
        1. A bar plot showing mean congruency ratings for each
           experiment, with separate bars for congruent and
           incongruent scenes and 95% confidence intervals
           approximated using 1.96 × SEM.
        2. A grouped bar plot showing mean ratings for congruent
           and incongruent scenes across experiments, with 95%
           confidence intervals and individual participant data points.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to a data file or a DataFrame containing trial-level
        experimental data, including participant IDs, experiment
        labels, stimulus filenames, and congruency-question responses.

    Returns
    -------
    None
        Prints experiment-level mean ratings and independent-samples
        t-test results, and displays and saves the two plots.
    """

    # --- Load data ---
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a file path or a DataFrame")

    # --- Filter relevant rows ---
    cong_Q_df = df[df["test_part"] == 'cong_questions'].copy()
    cong_Q_df["response"] = pd.to_numeric(cong_Q_df["response"], errors='coerce')

    # --- Extract filename parts ---
    cong_Q_df["img_filename"] = cong_Q_df["stimulus"].str.extract(r"src='img/([^']+)\.png'")
    cong_Q_df["set"] = cong_Q_df["img_filename"].str.extract(r'(set_\d+)')
    cong_Q_df["example"] = cong_Q_df["img_filename"].str.extract(r'example_(\d+)').astype(int)

    # --- Define congruency (ignore object type) ---
    def assign_group(row):
        if row["set"] == "set_1":
            return "cong" if 1 <= row["example"] <= 6 else "incong"
        elif row["set"] == "set_2":
            return "cong" if 1 <= row["example"] <= 6 else "incong"
        else:
            return None

    cong_Q_df["congruency"] = cong_Q_df.apply(assign_group, axis=1)
    cong_Q_df = cong_Q_df[cong_Q_df["congruency"].notna()]

    # --- Subject-level means ---
    subj_means = (
        cong_Q_df.groupby(["Subject", "Experiment", "congruency"], as_index=False)["response"]
            .mean()
    )

    # --- Print experiment-level means ---
    exp_means = subj_means.groupby(["Experiment", "congruency"], as_index=False)["response"].mean()
    print("\n=== Experiment-level means (per congruency) ===")
    print(exp_means)

    # --- Run independent t-tests for cong and incong separately ---
    exps = subj_means["Experiment"].unique()
    if len(exps) == 2:
        exp1, exp2 = exps
        print(f"\n=== Independent-samples t-tests ({exp1} vs {exp2}) ===")

        for cong_type in ["cong", "incong"]:
            group1 = subj_means.loc[
                (subj_means["Experiment"] == exp1) & (subj_means["congruency"] == cong_type),
                "response"
            ].dropna()
            group2 = subj_means.loc[
                (subj_means["Experiment"] == exp2) & (subj_means["congruency"] == cong_type),
                "response"
            ].dropna()

            if len(group1) > 1 and len(group2) > 1:
                t_val, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                print(f"{cong_type.capitalize()} → t = {t_val:.3f}, p = {p_val:.4f}")
            else:
                print(f"{cong_type.capitalize()} → Not enough data for t-test")
    else:
        print("⚠️ Need exactly 2 experiments for between-experiment t-test.")

    # --- Plot 1 ---

    subj_means["plot_group"] = (
            subj_means["Experiment"] + "_" + subj_means["congruency"]
    )

    experiments = ["Experiment 1", "Experiment 2"]
    congruencies = ["cong", "incong"]

    colors = {
        "Experiment 1": "#aeaeae",
        "Experiment 2": "#8ed973"
    }


    bar_width = 0.35
    within_gap = 0.02
    between_gap = 0.35

    # Manually define experiment positions
    x_positions = [0, 1.2]

    sns.set(style="white", font_scale=1.4)
    fig, ax = plt.subplots(figsize=(4, 5))

    for i, exp in enumerate(experiments):

        for j, cong in enumerate(congruencies):
            data = subj_means[
                (subj_means["Experiment"] == exp) &
                (subj_means["congruency"] == cong)
                ]["response"]

            x = x_positions[i] + (j - 0.5) * (bar_width + within_gap)

            mean = data.mean()
            ci = 1.96 * data.sem()

            # Hatch incongruent bars
            hatch = None if cong == "cong" else "//"

            bar = ax.bar(
                x, mean,
                width=bar_width,
                facecolor=colors[exp],
                edgecolor="black",
                linewidth=0,
                hatch=hatch
            )

            ax.errorbar(
                x, mean,
                yerr=ci,
                fmt="none",
                ecolor="black",
                capsize=4,
                linewidth=2.5
            )

            # # Individual subject points
            # ax.scatter(
            #     np.random.normal(x, 0.025, size=len(data)),
            #     data,
            #     color="black",
            #     alpha=0.2,
            #     s=20,
            #     zorder=3
            # )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(experiments)

    ax.set_title("Congruency Ratings per Experiment")
    ax.set_ylabel("Mean Likelihood Response")

    sns.despine()
    plt.tight_layout()
    plt.show()

    # --- Plot 2 ---
    custom_colors = {
        "Experiment 1": "#aeaeae",  # Gray
        "Experiment 2": "#8ed973",  # Green
    }


    sns.set(style="white", font_scale=1.4)
    plt.figure(figsize=(4, 5))
    ax = sns.barplot(
        data=subj_means,
        x="congruency",
        y="response",
        hue="Experiment",
        errorbar=("ci", 95),
        capsize=0.2,
        errwidth=1.5,
        palette=custom_colors
    )
    sns.stripplot(
        data=subj_means,
        x="congruency",
        y="response",
        hue="Experiment",
        dodge=True,
        color="black",
        alpha=0.4,
        jitter=0.1,
        legend=False
    )

    ax.legend_.remove()
    #
    # ax.legend(
    #     title="",
    #     bbox_to_anchor=(1.02, 1),
    #     loc="upper left",
    #     frameon=False
    # )

    ax.set_title("Congruency Ratings per Experiment")
    ax.set_ylabel("Mean Likelihood Response")
    sns.despine()

    plt.tight_layout()
    plt.show()


def analyze_ans_per_cond(data):

    """
    Analyze and visualize hit and false alarm rates across imagery
    and scene congruency conditions.

    The function accepts either a file path or a pandas DataFrame.
    It first excludes invalid participant blocks based on performance
    on condition-check trials, then filters the data to include only
    main-task trials.

    Hit rates are calculated for object-present trials, and false
    alarm rates for object-absent trials, separately for each
    participant, imagery condition, and congruency condition.

    The function generates two plots showing group-level hit and
    false alarm rates, with 95% confidence intervals and individual
    participant data points.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to a data file or a DataFrame containing trial-level
        responses, participant IDs, and experimental condition information.

    Returns
    -------
    None
        Displays plots of hit and false alarm rates across conditions.
    """

    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a file path or a DataFrame")

    # Exclude invalid blocks within the main task
    check_df = df[df["stimulus"].str.startswith(cnfg.pattern_check_cond_block, na=False)]
    to_exclude_list = [(row.Subject, row.block_num) for row in check_df.loc[df["correct"] == False].itertuples()]
    df = df[~df[["Subject", "block_num"]].apply(tuple, axis=1).isin(to_exclude_list)]

    # Main test trials
    main_df = df[df['test_part'] == 'main_test'].copy()

    # Per-subject hit rate (object present)
    hits_individual = (
        main_df[main_df["object_present"] == 1]
            .groupby(["Subject", "congruency", "imagine"])["correct"]
            .mean()
            .reset_index()
    )
    hits_individual["hit_rate"] = pd.to_numeric(hits_individual["correct"], errors='coerce') * 100

    # Per-subject false alarm rate (object absent)
    fa_individual = (
        main_df[main_df["object_present"] == 0]
            .groupby(["Subject", "congruency", "imagine"])["correct"]
            .apply(lambda x: (1 - x.mean()) * 100)
            .reset_index()
            .rename(columns={"correct": "false_alarm_rate"})
    )
    fa_individual["false_alarm_rate"] = fa_individual["false_alarm_rate"].astype(float)

    # -------- Plotting --------
    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

    # Set font sizes
    plt.rcParams.update({
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'legend.title_fontsize': 18
    })

    # ---- HIT RATE ----
    sns.barplot(
        data=hits_individual,
        x="congruency", y="hit_rate",
        hue="imagine", hue_order=[False, True],  # True on left
        errorbar=('ci', 95), ax=axes[0],
        # palette={True: "#4c72b0", False: "#92c6ff"},
        # palette={True: "#00affa", False: "#aaddff"}, # light and dark blue
        palette={True: "#636363", False: "#c8c8c8"},  # exp 1
        # palette={True: "#507d40", False: "#b5e7a4"}, # exp 2
        capsize=0.1, err_kws={'linewidth': 1.5}
    )

    # Hatch bars for imagine == True
    n_x = hits_individual["congruency"].nunique()
    # for i, patch in enumerate(axes[0].patches):
    #     if i < n_x:            # first block = imagine==True (hue drawn first)
    #         patch.set_hatch("//")

    sns.stripplot(
        data=hits_individual,
        x="congruency", y="hit_rate",
        hue="imagine", hue_order=[False, True],
        dodge=True, jitter=True, marker='o', alpha=0.7, size=6,
        # palette={True: "#4c72b0", False: "#92c6ff"},
        # palette={True: "#00affa", False: "#aaddff"}, # light and dark blue
        palette={True: "#303030", False: "#777777"},  # exp 1
        # palette={True: "#365c2b", False: "#6b9560"}, # exp 2
        ax=axes[0], legend=False,  # edgecolor='black',linewidth=0.5,
        zorder=3
    )

    # ---- FALSE ALARM RATE ----
    sns.barplot(
        data=fa_individual,
        x="congruency", y="false_alarm_rate",
        hue="imagine", hue_order=[False, True],
        errorbar=('ci', 95), ax=axes[1],
        # palette={True: "#4c72b0", False: "#92c6ff"},
        # palette={True: "#00affa", False: "#aaddff"}, # light and dark blue
        palette={True: "#636363", False: "#c8c8c8"},  # exp 1
        # palette={True: "#507d40", False: "#b5e7a4"}, # exp 2
        capsize=0.1, err_kws={'linewidth': 1.5}
    )

    # Hatch bars for imagine == True (same logic)
    n_x_fa = fa_individual["congruency"].nunique()
    # for i, patch in enumerate(axes[1].patches):
    #     if i < n_x_fa:
    #         patch.set_hatch("//")

    sns.stripplot(
        data=fa_individual,
        x="congruency", y="false_alarm_rate",
        hue="imagine", hue_order=[False, True],
        dodge=True, jitter=True, marker='o', alpha=0.5, size=6,
        # palette={True: "#4c72b0", False: "#92c6ff"},
        # palette={True: "#00affa", False: "#aaddff"}, # light and dark blue
        palette={True: "#303030", False: "#777777"},  # exp 1
        # palette={True: "#365c2b", False: "#6b9560"}, # exp 2
        ax=axes[1], legend=False,  # edgecolor='black',linewidth=0.5,
        zorder=3
    )
    # axes[1].axhline(30, color="black", linestyle="dashed")
    axes[1].set_ylabel("Positive answers [%]")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("FA", fontsize=22)  # , fontweight="bold"
    axes[1].get_legend().remove()

    # X-axis formatting
    axes[1].set_xlabel("Congruency")
    axes[0].set_xlabel("")
    axes[1].set_xticklabels(["Cong", "Incong"])

    # Legend
    legend_handles = [
        Patch(facecolor="#4c72b0", hatch='//', label="Imagine=True"),
        Patch(facecolor="#92c6ff", label="Imagine=False"),
        Line2D([0], [0], marker='o', color='w', label='Participants',
               markerfacecolor='gray', markersize=10, alpha=0.5)
    ]
    axes[0].legend(title="Imagined", loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()


def analyze_ans_per_cond_collepsed_stimuli_presence_line_plot(data):

    """
    Analyze and visualize positive answer rates across imagery and
    scene congruency conditions, collapsed across object presence.

    The function accepts either a file path or a pandas DataFrame.
    It excludes invalid participant blocks based on condition-check
    performance and filters the data to include only main-task trials.

    Positive answer rates are calculated for each participant,
    imagery condition, and congruency condition. A line plot shows
    group-level rates across imagery conditions, with separate lines
    for congruent and incongruent scenes and bootstrapped 95%
    confidence intervals at each condition.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to a data file or a DataFrame containing trial-level
        responses, participant IDs, and experimental condition information.

    Returns
    -------
    None
        Displays a line plot of positive answer rates across conditions.
    """

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a file path or a DataFrame")

    # --------------------------------------------------
    # Exclude invalid blocks within the main task
    # --------------------------------------------------

    check_df = df[
        df["stimulus"].str.startswith(
            cnfg.pattern_check_cond_block,
            na=False
        )
    ]

    to_exclude_list = [
        (row.Subject, row.block_num)
        for row in check_df.loc[
            check_df["correct"] == False
        ].itertuples()
    ]

    df = df[
        ~df[["Subject", "block_num"]]
        .apply(tuple, axis=1)
        .isin(to_exclude_list)
    ]

    # --------------------------------------------------
    # Main test trials
    # --------------------------------------------------

    main_df = df[
        df["test_part"] == "main_test"
    ].copy()

    # --------------------------------------------------
    # Positive answers
    # Present: correct = hit = positive answer
    # Absent: incorrect = false alarm = positive answer
    # --------------------------------------------------

    main_df["positive_answer"] = np.where(
        main_df["object_present"] == 1,
        main_df["correct"].astype(float),
        1 - main_df["correct"].astype(float)
    )

    # --------------------------------------------------
    # Per-subject positive answer rate
    # Averaged across present and absent trials
    # --------------------------------------------------

    positive_individual = (
        main_df
        .groupby(
            ["Subject", "congruency", "imagine"]
        )["positive_answer"]
        .mean()
        .reset_index()
    )

    positive_individual["positive_rate"] = (
        positive_individual["positive_answer"] * 100
    )

    # --------------------------------------------------
    # Condition labels and ordering
    # --------------------------------------------------

    positive_individual["imagine_label"] = (
        positive_individual["imagine"].map({
            True: "Imagery",
            False: "No imagery"
        })
    )

    positive_individual["imagine_label"] = pd.Categorical(
        positive_individual["imagine_label"],
        categories=["No imagery", "Imagery"],
        ordered=True
    )

    positive_individual["congruency_label"] = (
        positive_individual["congruency"].map({
            0: "Congruent",
            1: "Incongruent"
        })
    )

    # --------------------------------------------------
    # Plot settings
    # --------------------------------------------------

    plt.rcParams.update({
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
        'legend.title_fontsize': 16
    })

    fig, ax = plt.subplots(figsize=(6, 5))

    # --------------------------------------------------
    # Line plot (without automatic error bars)
    # --------------------------------------------------

    sns.lineplot(
        data=positive_individual,
        x="imagine_label",
        y="positive_rate",
        hue="congruency_label",
        style="congruency_label",
        hue_order=["Congruent", "Incongruent"],
        style_order=["Congruent", "Incongruent"],
        palette={
            "Congruent": "#aeaeae",
            "Incongruent": "#aeaeae" # exp1
        },
        # palette={
        #     "Congruent": "#8ed973",
        #     "Incongruent": "#8ed973" # exp2
        # },
        dashes={
            "Congruent": "",
            "Incongruent": (4, 2)
        },
        markers={
            "Congruent": "o",
            "Incongruent": "o"
        },
        estimator="mean",
        errorbar=None,
        markersize=9,
        linewidth=3.5,
        ax=ax
    )

    # --------------------------------------------------
    # Calculate and plot 95% CI at endpoints only
    # --------------------------------------------------

    rng = np.random.default_rng(42)

    for (imagine_label, congruency_label), group in (
        positive_individual.groupby(
            ["imagine_label", "congruency_label"],
            observed=True
        )
    ):

        values = group["positive_rate"].dropna().to_numpy()

        if len(values) < 2:
            continue

        mean = np.mean(values)

        # Bootstrap 95% CI across participants
        boot_means = np.array([
            np.mean(rng.choice(values, size=len(values), replace=True))
            for _ in range(10000)
        ])

        ci_low, ci_high = np.percentile(
            boot_means, [2.5, 97.5]
        )

        x_position = {
            "No imagery": 0,
            "Imagery": 1
        }[imagine_label]

        ax.errorbar(
            x=x_position,
            y=mean,
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="none",
            # ecolor="#8ed973", # exp2
            ecolor="#aeaeae", # exp1
            elinewidth=2,
            capsize=5,
            zorder=4
        )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    ax.set_ylabel("Positive answers (%)")
    ax.set_xlabel("Imagery condition")
    ax.set_title("Positive Answer Rate", fontsize=22)

    ax.set_ylim(30, 50)

    # No imagery first, imagery second
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No imagery", "Imagery"])

    # Horizontal padding
    ax.set_xlim(-0.25, 1.25)

    ax.legend(
        title="Scene congruency",
        loc="upper left",
        bbox_to_anchor=(1, 1)
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(0.5)

    plt.tight_layout()
    plt.show()


def analyze_vividness_per_presenece_per_cond_per_response(data):

    """
    Analyze and visualize normalized vividness ratings across imagery,
    scene congruency, object presence, and detection response outcome conditions.

    The function accepts either a file path or a pandas DataFrame.
    It excludes invalid participant blocks based on condition-check
    performance and extracts vividness responses from the task.

    Vividness ratings are z-score normalized separately for each
    participant and imagery condition. Mean normalized vividness is
    then calculated for each combination of imagery, congruency,
    and response outcome (hits, misses, false alarms, and correct
    rejections).

    The function generates two bar plots, showing vividness ratings
    for imagined and perceived conditions, with standard error bars
    and individual participant data points.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to a data file or a DataFrame containing vividness
        responses, participant IDs, and experimental condition information.

    Returns
    -------
    None
        Displays plots of normalized vividness ratings across conditions.
    """

    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a file path or a DataFrame")

    # Exclude invalid blocks within the main task
    check_df = df[df["stimulus"].str.startswith(cnfg.pattern_check_cond_block, na=False)]
    to_exclude_list = [(row.Subject, row.block_num) for row in check_df.loc[df["correct"] == False].itertuples()]

    # Use a boolean mask to filter out rows that match any of those tuples
    df = df[~df[["Subject", "block_num"]].apply(tuple, axis=1).isin(to_exclude_list)]

    # Create a new column 'answer_state' based on conditions
    df['answer_state'] = np.where(
        (df['object_present'] == 0) & (df['correct'] == False), 'FA',  # False alarm: obj_num is NaN and correct is True
        np.where(
            (df['object_present'] == 0) & (df['correct'] == True), 'CR',
            # Correct Rejection: obj_num is NaN and correct
            # is True
            np.where(
                (df['object_present'] == 1) & (df['correct'] == False), 'miss',
                # miss: obj_num is there and correct is False
                np.where(
                    (df['object_present'] == 1) & (df['correct'] == True), 'hit',
                    # Correcet Rejection: obj_num is there and correct is True
                    np.nan  # In all other cases, set as NaN
                )
            )
        )
    )

    # Apply the filter to get only the rows matching the vividness question
    vividness_df = df[df['test_part'] == 'vividness_q']

    vividness_df['normalized_response'] = np.nan

    # Conditions you want to normalize (others will remain NaN in normalized_response)
    conditions_to_normalize_ima = [True]
    conditions_to_normalize_perc = [False]

    # Apply normalization
    zscore_normalize_by_condition(vividness_df, conditions_to_normalize_ima)
    zscore_normalize_by_condition(vividness_df, conditions_to_normalize_perc)

    # Now, calculate the mean of the normalized response for each condition
    vividness_df = vividness_df.groupby(["Subject", 'imagine', 'congruency', "answer_state"])[
        "normalized_response"].mean().reset_index()

    # --------------------------------------------------
    # LABELS
    # --------------------------------------------------

    vividness_df["congruency_label"] = vividness_df["congruency"].map(
        {0: "Cong", 1: "Incong"}
    )

    vividness_df["imagine_label"] = vividness_df["imagine"].map(
        {True: "Imagined", False: "Perceived"}
    )

    vividness_df["condition"] = (
            vividness_df["answer_state"]
            + "_"
            + vividness_df["congruency_label"]
    )

    # --------------------------------------------------
    # FIGURE SETTINGS
    # --------------------------------------------------

    plt.rcParams.update({
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    sns.set_style("white")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharey=True,
        facecolor="none"
    )

    fig.patch.set_alpha(0)

    # Order of bars:
    # Object Absent first -> Object Present second
    order = [
        # Object Absent
        "CR_Cong",
        "CR_Incong",
        "FA_Cong",
        "FA_Incong",

        # Object Present
        "miss_Cong",
        "miss_Incong",
        "hit_Cong",
        "hit_Incong",
    ]

    # subplot specification
    # subplot_info = [
    #     (True, "#636363", "Imagined"),
    #     (False, "#c8c8c8", "Perceived")
    # ] # exp1

    # subplot specification
    subplot_info = [
        (True, "#507d40", "Imagined"),
        (False, "#b5e7a4", "Perceived")
    ]  # exp2

    # --------------------------------------------------
    # LOOP OVER SUBPLOTS
    # --------------------------------------------------

    for ax, (imagery_cond, color, title) in zip(axes, subplot_info):

        subset = vividness_df[
            vividness_df["imagine"] == imagery_cond
            ].copy()

        # -----------------------
        # Barplot
        # -----------------------

        sns.barplot(
            data=subset,
            x="condition",
            y="normalized_response",
            order=order,
            color=color,
            errorbar="se",
            capsize=.1,
            err_kws={"linewidth": 1.5},
            linewidth=0,
            ax=ax
        )

        # -----------------------
        # Subject points
        # -----------------------

        sns.stripplot(
            data=subset,
            x="condition",
            y="normalized_response",
            order=order,
            color=color,
            alpha=.30,
            jitter=.15,
            size=3,
            edgecolor="black",
            linewidth=.3,
            ax=ax
        )

        # -----------------------
        # Hatch incongruent bars
        # -----------------------

        bars = ax.patches[:len(order)]

        for bar_idx, bar in enumerate(bars):

            if "Incong" in order[bar_idx]:
                bar.set_hatch("//")

            bar.set_edgecolor("black")
            bar.set_linewidth(0)

        # -----------------------
        # Formatting
        # -----------------------

        ax.set_title(title)

        ax.axhline(
            0,
            color="0.6",
            linewidth=1.2,
            zorder=0
        )

        # Optional separator between
        # Object Present and Object Absent
        ax.axvline(
            3.5,
            color="0.75",
            linestyle="--",
            linewidth=1
        )

        ax.set_facecolor("none")

        ax.grid(False)

        ax.set_ylim(-3.4, 3.4)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)

        ax.tick_params(
            axis='both',
            colors='black',
            width=1
        )

        ax.set_xlabel("")

        # remove legends
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    # --------------------------------------------------
    # Axis labels
    # --------------------------------------------------

    axes[0].set_ylabel("Z-scored Vividness")
    axes[1].set_ylabel("")

    # nicer x labels

    nice_labels = [
        "CR\nC",
        "CR\nI",
        "FA\nC",
        "FA\nI",
        "Hit\nC",
        "Hit\nI",
        "Miss\nC",
        "Miss\nI",
    ]

    for ax in axes:
        ax.set_xticklabels(nice_labels)

    # --------------------------------------------------
    # Save and show
    # --------------------------------------------------

    plt.tight_layout()

    # plt.savefig(
    #     r"YOUR_PATH\ng",
    #     dpi=500,
    #     transparent=True,
    #     bbox_inches="tight"
    # )

    plt.show()


if __name__ == '__main__':

    ### Data arrangement ###

    # For only one experiment folder
    file = one_dataframe()

    # For more than one experiment folders:

    # file1 = one_dataframe()
    # file2 = one_dataframe()
    # file = combined_dataframe(file1, file2)

    ### Data validation ###
    check_exclusions_acc_and_ima_check(file)
    analyze_staircase_acc_group(file)

    ### Analyses ###
    anova_positive_ans(file)
    lmm_imagine_vividness(file)
    lmm_visibility_vividness(file)
    # analyze_general_cong_Q_cong_incong_bet_exp(file)

    ### Visualization ###
    analyze_ans_per_cond(file)
    analyze_ans_per_cond_collepsed_stimuli_presence_line_plot(file)
    analyze_vividness_per_presenece_per_cond_per_response(file)

    ### Further analysis data extraction ###
    export_GLMM_with_scenes_and_diff_cong(file)
