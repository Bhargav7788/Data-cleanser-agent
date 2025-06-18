import pandas as pd
import numpy as np

def drop_high_null_columns(df, threshold):
    null_percent = df.isnull().mean()
    to_drop = null_percent[null_percent > threshold].index
    return df.drop(columns=to_drop)

def impute_missing_values(df, strategy='mean'):
    df_imputed = df.copy()
    for col in df_imputed.select_dtypes(include=[np.number]):
        if df_imputed[col].isnull().sum() > 0:
            if strategy == 'mean':
                df_imputed[col].fillna(df_imputed[col].mean(), inplace=True)
            elif strategy == 'median':
                df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
            elif strategy == 'mode':
                df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
    return df_imputed

def standardize_text_columns(df):
    df_std = df.copy()
    for col in df_std.select_dtypes(include='object'):
        df_std[col] = df_std[col].str.lower().str.strip()
    return df_std

def remove_duplicates(df):
    return df.drop_duplicates()
