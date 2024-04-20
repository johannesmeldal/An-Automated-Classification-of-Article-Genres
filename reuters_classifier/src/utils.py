# src/utils.py

import pandas as pd
from ast import literal_eval

def load_data(file_path):
    """
    Loads the data from CSV and returns a DataFrame.
    """
    df = pd.read_csv(file_path)
    df.dropna(subset=['BODY', 'TOPICS'], inplace=True)
    df['TOPICS'] = df['TOPICS'].apply(literal_eval)  # Convert stringified lists back into actual lists
    return df
