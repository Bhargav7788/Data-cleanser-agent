import pandas as pd
import numpy as np
from scipy import stats

def impute_missing_values(df, strategy="mean"):
    df_cleaned = df.copy()
    for col in df_cleaned.select_dtypes(include=[np.number]).columns:
        if strategy == "mean":
            df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
        elif strategy == "median":
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        elif strategy == "mode":
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
    return df_cleaned

def drop_high_null_columns(df, threshold=0.5):
    return df.loc[:, df.isnull().mean() < threshold]

def standardize_text_columns(df):
    df_cleaned = df.copy()
    for col in df_cleaned.select_dtypes(include=['object', 'string']).columns:
        df_cleaned[col] = df_cleaned[col].str.lower().str.strip()
    return df_cleaned

def drop_duplicates(df):
    return df.drop_duplicates()

def remove_outliers_zscore(df, threshold=3.0):
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df_cleaned[numeric_cols]))
    mask = (z_scores < threshold).all(axis=1)
    return df_cleaned.loc[mask]

def remove_outliers_iqr(df, multiplier=1.5):
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    Q1 = df_cleaned[numeric_cols].quantile(0.25)
    Q3 = df_cleaned[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df_cleaned[numeric_cols] < (Q1 - multiplier * IQR)) | (df_cleaned[numeric_cols] > (Q3 + multiplier * IQR))).any(axis=1)
    return df_cleaned.loc[mask]

def standardize_text(df):
    df_cleaned = df.copy()
    for col in df_cleaned.select_dtypes(include=['object', 'string']).columns:
        df_cleaned[col] = df_cleaned[col].str.lower().str.strip()
    return df_cleaned
