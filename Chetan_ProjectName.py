# ==============================================================================
# Project: ML Network Intrusion Detector
# File Name: YourName_ProjectName.py
# Author: YourName
# Description: End-to-End Scikit-Learn Pipeline to classify network traffic
#              anomalies (Normal, DoS, Probe, R2L, U2R) using the NSL-KDD dataset.
# ==============================================================================

import io
import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    f1_score
)

# 1. DEFINE NSL-KDD 41 FEATURE COLUMNS + TARGET LABEL
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

def generate_synthetic_nsl_kdd(n_samples=4000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "duration": rng.integers(0, 300, size=n_samples),
        "protocol_type": rng.choice(["tcp", "udp", "icmp"], size=n_samples, p=[0.8, 0.15, 0.05]),
        "service": rng.choice(["http", "private", "smtp", "ftp_data", "domain_u"], size=n_samples),
        "flag": rng.choice(["SF", "S0", "REJ", "RSTR"], size=n_samples, p=[0.65, 0.2, 0.1, 0.05]),
        "src_bytes": np.clip(rng.lognormal(mean=6.0, sigma=1.6, size=n_samples), 1, None).astype(int),
        "dst_bytes": np.clip(rng.lognormal(mean=5.8, sigma=1.8, size=n_samples), 1, None).astype(int),
        "land": rng.choice([0, 1], size=n_samples, p=[0.99, 0.01]),
        "wrong_fragment": rng.choice([0, 1, 2, 3], size=n_samples, p=[0.92, 0.04, 0.03, 0.01]),
        "urgent": rng.choice([0, 1, 2], size=n_samples, p=[0.98, 0.014, 0.006]),
        "hot": rng.integers(0, 100, size=n_samples),
        "num_failed_logins": rng.choice([0, 1, 2, 3], size=n_samples, p=[0.96, 0.02, 0.015, 0.005]),
        "logged_in": rng.choice([0, 1], size=n_samples, p=[0.45, 0.55]),
        "num_compromised": rng.integers(0, 30, size=n_samples),
        "root_shell": rng.choice([0, 1], size=n_samples, p=[0.94, 0.06]),
        "su_attempted": rng.choice([0, 1, 2], size=n_samples, p=[0.98, 0.015, 0.005]),
        "num_root": rng.integers(0, 25, size=n_samples),
        "num_file_creations": rng.integers(0, 20, size=n_samples),
        "num_shells": rng.integers(0, 15, size=n_samples),
        "num_access_files": rng.integers(0, 15, size=n_samples),
        "num_outbound_cmds": rng.integers(0, 5, size=n_samples),
        "is_host_login": rng.choice([0, 1], size=n_samples, p=[0.99, 0.01]),
        "is_guest_login": rng.choice([0, 1], size=n_samples, p=[0.97, 0.03]),
        "count": rng.integers(1, 511, size=n_samples),
        "srv_count": rng.integers(1, 511, size=n_samples),
        "serror_rate": np.round(rng.beta(0.5, 2.0, size=n_samples), 2),
        "srv_serror_rate": np.round(rng.beta(0.6, 2.0, size=n_samples), 2),
        "rerror_rate": np.round(rng.beta(1.0, 3.0, size=n_samples), 2),
        "srv_rerror_rate": np.round(rng.beta(1.0, 3.0, size=n_samples), 2),
        "same_srv_rate": np.round(rng.beta(2.0, 0.5, size=n_samples), 2),
        "diff_srv_rate": np.round(rng.beta(1.2, 2.0, size=n_samples), 2),
        "srv_diff_host_rate": np.round(rng.beta(1.0, 3.0, size=n_samples), 2),
        "dst_host_count": rng.integers(1, 255, size=n_samples),
        "dst_host_srv_count": rng.integers(1, 255, size=n_samples),
        "dst_host_same_srv_rate": np.round(rng.beta(2.5, 0.7, size=n_samples), 2),
        "dst_host_diff_srv_rate": np.round(rng.beta(1.0, 2.0, size=n_samples), 2),
        "dst_host_same_src_port_rate": np.round(rng.beta(2.0, 1.0, size=n_samples), 2),
        "dst_host_srv_diff_host_rate": np.round(rng.beta(1.0, 3.0, size=n_samples), 2),
        "dst_host_serror_rate": np.round(rng.beta(0.8, 2.0, size=n_samples), 2),
        "dst_host_srv_serror_rate": np.round(rng.beta(0.8, 2.0, size=n_samples), 2),
        "dst_host_rerror_rate": np.round(rng.beta(1.0, 3.0, size=n_samples), 2),
        "dst_host_srv_rerror_rate": np.round(rng.beta(1.0, 3.0, size=n_samples), 2),
        "label": np.full(n_samples, "normal"),
        "difficulty_level": np.zeros(n_samples, dtype=int),
    })

    attack_mask = (
        (df["flag"] == "S0") & (df["serror_rate"] > 0.5)
    ) | (
        (df["service"] == "private") & (df["count"] > 200)
    ) | (
        (df["num_failed_logins"] > 0)
    )

    attack_labels = np.where(
        attack_mask,
        rng.choice(["neptune", "portsweep", "guess_passwd"], size=n_samples),
        "normal"
    )
    df["label"] = attack_labels
    return df


def load_nsl_kdd_dataset():
    train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
    try:
        print("[INFO] Downloading NSL-KDD Training Dataset...")
        request = urllib.request.Request(train_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=15) as response:
            raw_data = response.read()
        df = pd.read_csv(io.BytesIO(raw_data), names=COLUMN_NAMES, header=None)
        df = df.drop(columns=["difficulty_level"], errors="ignore")
        print(f"[INFO] Successfully loaded {len(df):,} network records from NSL-KDD.")
    except Exception as e:
        print(f"[WARN] Network download failed ({e}). Generating a fast synthetic NSL-KDD sample instead...")
        df = generate_synthetic_nsl_kdd(n_samples=4000)
        print(f"[INFO] Synthetic dataset created with {len(df):,} records.")

    df["is_intrusion"] = (df["label"] != "normal").astype(int)
    return df

def build_and_evaluate_models(df):
    X = df.drop(columns=["label", "is_intrusion"])
    y = df["is_intrusion"]

    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    }

    best_pipeline = None
    best_f1 = 0.0

    for name, clf in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Intrusion (1)"]))

        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pipeline

    joblib.dump(best_pipeline, "network_intrusion_detector.pkl")
    print("\n[SUCCESS] Best model saved to 'network_intrusion_detector.pkl'")
    return best_pipeline

if __name__ == "__main__":
    dataset = load_nsl_kdd_dataset()
    trained_model = build_and_evaluate_models(dataset)
