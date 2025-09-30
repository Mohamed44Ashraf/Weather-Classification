import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

warnings.filterwarnings('ignore')


def load_data(path):
    df = pd.read_csv(path)
    return df


def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method (clip values)"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df


def preprocessing(df):
    # Drop unnecessary columns if they exist
    drop_cols = ['RISK_MM', 'Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Split numerical & categorical
    num_col = df.select_dtypes(include=['int64', 'float64']).columns
    cat_col = df.select_dtypes(include='object').columns

    # Handle missing values
    knn_imputer = KNNImputer(n_neighbors=5)
    df[num_col] = knn_imputer.fit_transform(df[num_col])

    simple_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_col] = df[cat_col].astype(str)
    df[cat_col] = simple_imputer.fit_transform(df[cat_col])

    # Feature engineering - month from Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df["month"] = df["Date"].dt.month.astype('int32')
        df = df.drop(columns=['Date'])

    # Handle outliers
    for col in num_col:
        df = remove_outliers_iqr(df, col)

    return df


if __name__ == "__main__":
    data_path = r"D:\Track projects\Weather_Classification\dataset\weatherAUS.csv"
    df = load_data(data_path)
    df = preprocessing(df)
    print(df.head())
