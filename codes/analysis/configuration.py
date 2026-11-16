### This module contains all the keywords or squences used to find the differnet scetion within the SCEPRM experiment.


consent_form_name = 'MROS_6649-04_consent_v5.0_20210315'
info_form_name = 'MROS_6649-04_information_v5.0_20210315'

### Key sentences to recognize different part within the experiment:

staircasing_stars = "We will now do a calibration block."
staircasing_ends = "Well done! <br><br> The final part of the experiment is that during those trials"

### For LME power analysis:
RANDOM_INTERCEPT_SD = 0.5
RESIDUAL_SD = 0.8

### constants

obj_num = 4  # how many object we tested
number_of_trials_within_stair = 10
number_of_trials_within_block =24
staircase_code = 'stair_test'  # key code to recognize the staircase procedure part in the expriment

pattern_instr_stair = r"set_(\d+)_obj_(\d+)_example\.png"  # Define the regular expression to match 'set_x_obj_y_example.png' pattern, should be found before each staircase procedure for spesific object
pattern_general_cong_Q = "How likely is it to find"
pattern_check_cond_block = "Check! <br>"
cond_dict = {
    "0": {
        "obj_code": '[1,1]',
        "cong": True,
        "cond": 'cong_perc'
    },
    "1": {
        "obj_code": '[1,1]',
        "cong": True,
        "cond": 'cong_ima'
    },
    "2": {
        "obj_code": '[1,1]',
        "cong": False,
        "cond": 'incong_perc'
    },
    "3": {
        "obj_code": '[1,1]',
        "cong": False,
        "cond": 'incong_ima'
    },
    "4": {
        "obj_code": '[1,2]',
        "cong": True,
        "cond": 'cong_perc'
    },
    "5": {
        "obj_code": '[1,2]',
        "cong": True,
        "cond": 'cong_ima'
    },
    "6": {
        "obj_code": '[1,2]',
        "cong": False,
        "cond": 'incong_perc'
    },
    "7": {
        "obj_code": '[1,2]',
        "cong": False,
        "cond": 'incong_ima'
    },
    "8": {
        "obj_code": '[2,1]',
        "cong": True,
        "cond": 'cong_perc'
    },
    "9": {
        "obj_code": '[2,1]',
        "cong": True,
        "cond": 'cong_ima'
    },
    "10": {
        "obj_code": '[2,1]',
        "cong": False,
        "cond": 'incong_perc'
    },
    "11": {
        "obj_code": '[2,1]',
        "cong": False,
        "cond": 'incong_ima'
    },
    "12": {
        "obj_code": '[2,2]',
        "cong": True,
        "cond": 'cong_perc'
    },
    "13": {
        "obj_code": '[2,2]',
        "cong": True,
        "cond": 'cong_ima'
    },
    "14": {
        "obj_code": '[2,2]',
        "cong": False,
        "cond": 'incong_perc'
    },
    "15": {
        "obj_code": '[2,2]',
        "cong": False,
        "cond": 'incong_ima'
    },
}
