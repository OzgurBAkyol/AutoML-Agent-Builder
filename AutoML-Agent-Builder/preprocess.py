# preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df: pd.DataFrame, target_column: str):
    df = df.copy()

    # Eksik verileri basit doldurma (numerik: ortalama, kategorik: mod)
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Hedef kolon ayrıştır
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Kategorik kolonları label encode et
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

    # Hedef kolon da kategorikse encode et (classification için)
    if y.dtype == "object" or y.dtype.name == "category":
        y = LabelEncoder().fit_transform(y)

    return X, y
