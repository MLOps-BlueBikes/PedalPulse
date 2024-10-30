import pandas as pd

def remove_missing_values(df):
    df = df.dropna()
    return df

def remove_duplicates(df):
    return df.dropna()