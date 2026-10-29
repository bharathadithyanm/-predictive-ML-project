# src/model.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


def run_model():
    print("🚀 Running Predictive Model...")

    # ==========================================
    # STEP 1: LOAD DATASET (CSV)
    # ==========================================
    try:
        data = pd.read_csv("data/dataset.csv")
        print("\n📄 Dataset Loaded Successfully!")
    except FileNotFoundError:
        print("❌ dataset.csv not found inside 'data/' folder")
        return

    print("\nDataset Preview:\n", data.head())

    # ==========================================
    # STEP 2: SPLIT FEATURES & TARGET
    # ==========================================
    X = data.drop("target", axis=1)
    y = data["target"]

    # ==========================================
    # STEP 3: TRAIN-TEST SPLIT
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ==========================================
    # STEP 4: MODELS
    # ==========================================
    lr = LinearRegression()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # ==========================================
    # STEP 5: TRAIN MODELS
    # ==========================================
    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # ==========================================
    # STEP 6: PREDICTIONS
    # ==========================================
    lr_pred = (lr.predict(X_test) > 0.5).astype(int)
    dt_pred = dt.predict(X_test)
    rf_pred = rf.predict(X_test)

    # ==========================================
    # STEP 7: ACCURACY
    # ==========================================
    print("\n📊 Accuracy:")
    print("Linear Regression:", accuracy_score(y_test, lr_pred))
    print("Decision Tree     :", accuracy_score(y_test, dt_pred))
    print("Random Forest     :", accuracy_score(y_test, rf_pred))

    # ==========================================
    # STEP 8: CREATE OUTPUT FOLDER
    # ==========================================
    os.makedirs("outputs", exist_ok=True)

    # ==========================================
    # STEP 9: CONFUSION MATRIX
    # ==========================================
    cm = confusion_matrix(y_test, rf_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # ==========================================
    # STEP 10: ROC CURVE
    # ==========================================
    y_prob = rf.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Random Forest")
    plt.legend()
    plt.savefig("outputs/roc_curve.png")
    plt.close()

    # ==========================================
    # FINAL MESSAGE
    # ==========================================
    print("\n✅ Outputs saved in 'outputs/' folder")
    print("📁 Check confusion_matrix.png and roc_curve.png")