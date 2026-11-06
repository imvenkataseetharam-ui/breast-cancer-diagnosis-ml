"""
Breast Cancer Diagnosis Using Machine Learning
Educational B.Tech AI & ML mini project.

Dataset: Wisconsin Breast Cancer Diagnostic dataset bundled with scikit-learn.
Classes: malignant and benign.
Disclaimer: This project is for education only and is NOT a medical diagnostic tool.
"""

import matplotlib
matplotlib.use("Agg")  # Allows plots to be saved in environments without a display.

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance
import joblib


def main():
    # 1. Load the public, built-in Wisconsin Breast Cancer Diagnostic dataset.
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data

    # scikit-learn encodes malignant as 0 and benign as 1.
    # Convert labels so malignant=1, benign=0 for clearer positive-class metrics.
    y = (dataset.target == 0).astype(int)
    class_names = {0: "Benign", 1: "Malignant"}

    print("=" * 72)
    print("BREAST CANCER DIAGNOSIS USING MACHINE LEARNING")
    print("EDUCATIONAL DEMONSTRATION ONLY — NOT FOR MEDICAL USE")
    print("=" * 72)
    print(f"\nDataset samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print("\nClass counts:")
    print(y.map(class_names).value_counts().to_string())

    # Save a copy of the dataset for project documentation.
    data_for_export = X.copy()
    data_for_export["diagnosis"] = y.map(class_names)
    data_for_export.to_csv("breast_cancer_dataset.csv", index=False)

    # 2. Split data. Keep a separate test set for final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Define models. Scaling is applied inside a pipeline for Logistic Regression.
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
    }

    # 4. Train and evaluate on the held-out test set.
    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        predictions = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
            auc = roc_auc_score(y_test, probabilities)
        else:
            auc = float("nan")

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision (malignant)": precision_score(y_test, predictions, zero_division=0),
            "Recall (malignant)": recall_score(y_test, predictions, zero_division=0),
            "F1-score (malignant)": f1_score(y_test, predictions, zero_division=0),
            "ROC-AUC": auc,
        })

        print(f"\n{'-' * 72}\n{name}")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            zero_division=0,
        ))

        # Save a confusion matrix image for the report.
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=[0, 1],
            display_labels=["Benign", "Malignant"],
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )
        ax.set_title(f"Confusion Matrix - {name}")
        fig.tight_layout()
        filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"
        fig.savefig(filename, dpi=160)
        plt.close(fig)

    results_df = pd.DataFrame(results).sort_values(
        by=["Recall (malignant)", "F1-score (malignant)"], ascending=False
    )
    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nMODEL COMPARISON (held-out test set)")
    print(results_df.round(4).to_string(index=False))

    # Cross-validation on training data gives a more stable estimate without
    # using the held-out test set for model selection.
    print("\n5-FOLD STRATIFIED CROSS-VALIDATION ON TRAINING DATA (accuracy)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

    # 5. Save the Random Forest model as a demonstration artifact.
    # It is not validated or approved for clinical use.
    final_model = fitted_models["Random Forest"]
    joblib.dump(final_model, "breast_cancer_random_forest.joblib")
    print("\nSaved files:")
    print("- breast_cancer_dataset.csv")
    print("- model_comparison_results.csv")
    print("- *_confusion_matrix.png")
    print("- breast_cancer_random_forest.joblib")

    # 6. Demonstrate prediction for one held-out sample (not a real-time diagnosis).
    sample = X_test.iloc[[0]]
    predicted_class = int(final_model.predict(sample)[0])
    probability = float(final_model.predict_proba(sample)[0, list(final_model.classes_).index(1)])
    print("\nDEMONSTRATION PREDICTION FOR ONE TEST-DATA SAMPLE")
    print(f"Predicted dataset class: {class_names[predicted_class]}")
    print(f"Model-estimated malignant-class score: {probability:.3f}")
    print("This score is not a clinical probability and must not guide medical decisions.")


if __name__ == "__main__":
    main()
