import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import joblib

# === Load dataset ===
file_path = r"C:/Users/BIOLAB/Downloads/project/New folder/testing folder/alzheimers_disease_data.csv"
data = pd.read_csv(file_path)

# === Preprocessing ===
le = LabelEncoder()
y = le.fit_transform(data["Diagnosis"])
class_names = [str(c) for c in le.classes_]  # ensure string names

X = data.drop(columns=["Diagnosis"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Handle imbalance ratio ===
pos_weight = (sum(y == 0) / sum(y == 1))  # ~ 1389/760 in your dataset

# === Models with corrections ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "SVM": SVC(probability=True, kernel="rbf", random_state=42, class_weight="balanced"),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    "XGBoost": CalibratedClassifierCV(
        XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
            scale_pos_weight=pos_weight
        ),
        method="isotonic",
        cv=5
    )
}

results = []

for name, model in models.items():
    print(f"\n=== {name} Results ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probabilities
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # AUC handling
    if y_proba is not None:
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            auc = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
    else:
        auc = np.nan

    results.append([name, acc, prec, rec, f1, auc])

    # === Print in clean format ===
    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# === Results summary ===
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"])
print("\n✅ Model Performance Comparison:")
print(results_df)

# === Save the XGBoost model (calibrated) ===
joblib.dump(models["XGBoost"], "alzheimers_xgb_model.pkl")
print("Model saved as 'alzheimers_xgb_model.pkl'")