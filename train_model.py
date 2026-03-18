#!/usr/bin/env python3


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import numpy as np
from datetime import datetime

CSV_PATH = "pose_dataset.csv"
MODEL_OUT = "exercise_classifier.pkl"

def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run collect_data.py first.")

    df = pd.read_csv(csv_path)
    df = df.dropna()

    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values

    return X, y

def main():
    print("=== Train model ===")
    X, y = load_dataset(CSV_PATH)
    print("Loaded dataset:", X.shape, "samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train:", X_train.shape, "Test:", X_test.shape)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outname = f"exercise_classifier_{ts}.pkl"
    joblib.dump(clf, outname)
    joblib.dump(clf, MODEL_OUT)
    print("Model saved as:", outname, "and", MODEL_OUT)

if __name__ == "__main__":
    main()

