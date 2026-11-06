# Breast Cancer Diagnosis Using Machine Learning

## Project overview
This educational mini-project trains and compares Logistic Regression, Decision Tree, and Random Forest models using the Wisconsin Breast Cancer Diagnostic dataset included with scikit-learn.

**Important:** This is an academic demonstration only. It is not a medical device, has not been clinically validated, and must not be used to diagnose or rule out cancer.

## Requirements
- Python 3.10 or newer recommended
- Internet is not required for the dataset because it is bundled with scikit-learn

## Setup
Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Activate the environment:
- Windows: `.venv\\Scripts\\activate`
- macOS/Linux: `source .venv/bin/activate`

Install packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python breast_cancer_diagnosis.py
```

## Output files
The script creates:
- `breast_cancer_dataset.csv` — dataset with readable class labels
- `model_comparison_results.csv` — test-set metrics for each model
- `*_confusion_matrix.png` — confusion matrix plots
- `breast_cancer_random_forest.joblib` — saved Random Forest pipeline

The program prints classification reports, a model comparison table, cross-validation scores, and a demonstration prediction.

## Dataset notes
The scikit-learn dataset contains 569 samples and 30 numeric features. The original target is encoded by scikit-learn as malignant=0 and benign=1; the script remaps it for reporting so malignant=1. This project uses a stratified 80/20 train/test split with a fixed random seed.

## Metrics
- Accuracy: proportion of all predictions that are correct.
- Precision (malignant): among samples predicted malignant, the fraction labeled malignant in the dataset.
- Recall (malignant): among malignant samples in the dataset, the fraction detected as malignant by the model.
- F1-score: harmonic mean of precision and recall.
- ROC-AUC: ability to rank positive and negative classes across thresholds.

For a medical screening context, false negatives are especially important. Test-set metrics from this small, public dataset do not establish real-world clinical performance.

## Suggested report conclusion
This project demonstrates a machine-learning workflow for classifying breast tumor measurements in a public dataset. Three algorithms are trained and compared using held-out test metrics and cross-validation. The results illustrate the use of supervised learning in medical-data research, but the models are educational only and are not suitable for clinical decision-making.
