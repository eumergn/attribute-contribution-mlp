# src/preprocess.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_prepare_data(csv_path):
    # 1. Charger les données
    df = pd.read_csv(csv_path)

    # 2. Séparer les features et la cible
    X = df.drop(columns=["species"])
    y = df["species"]

    # 3. Encoder la cible (ex: setosa → 0, versicolor → 1, etc.)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 4. Détection des types de colonnes
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # 5. Prétraitement (scaling pour numérique, one-hot pour catégoriel)
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols)
    ])

    # 6. Transformation complète (fit + transform sur X)
    X_processed = preprocessor.fit_transform(X)

    # 7. Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor, label_encoder
