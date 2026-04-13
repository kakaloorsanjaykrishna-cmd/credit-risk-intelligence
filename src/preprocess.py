import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="data/credit_data.csv"):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Basic clean (your data is already clean)
    df = df.drop_duplicates()

    X = df.drop("Risk", axis=1)
    y = df["Risk"]

    return train_test_split(X, y, test_size=0.2, random_state=42)