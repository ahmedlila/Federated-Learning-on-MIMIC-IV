import pandas as pd
from utils import get_data_path
from data_cleaning import (
    preprocess_data, clean_temperature_data, 
    clean_pain_data, fill_missing_values
)

def load_edstays_data(columns=None):
    """Load and process ED stays data."""
    if columns is None:
        columns = ["subject_id", 'stay_id', "gender", "race", "arrival_transport", "disposition"]
    
    df = pd.read_csv(get_data_path("edstays.csv"))
    df = df[df['disposition'].isin(["ADMITTED", "HOME"])] # use only these dispositions for training
    return preprocess_data(df, columns)

def load_triage_data(columns=None):
    """Load and process triage data."""
    if columns is None:
        columns = ['subject_id','stay_id', 'temperature', 'heartrate', 'resprate', 
                  'o2sat', 'sbp', 'dbp', 'pain', 'acuity', 'chiefcomplaint']
    
    df = pd.read_csv(get_data_path("triage.csv"))
    df = clean_pain_data(df)
    df = clean_temperature_data(df)
    return preprocess_data(df, columns)

def load_vitalsign_data(columns=None):
    """Load and process vitalsign data."""
    if columns is None:
        columns = ['subject_id', 'temperature', 'stay_id', 'heartrate', 
                  'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
    
    df = pd.read_csv(get_data_path("vitalsign.csv"))
    df = clean_pain_data(df)
    df = clean_temperature_data(df)
    df = fill_missing_values(df, ['temperature', 'heartrate', 'resprate', 
                                'o2sat', 'sbp', 'dbp', 'pain'])
    return preprocess_data(df, columns)