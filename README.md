# 🏥 Health Condition Classification Pipeline

An end-to-end, high-performance machine learning pipeline built with **LightGBM** and **Optuna** to predict multi-class health conditions (`fit`, `at-risk`, `unhealthy`) from tabular lifestyle and clinical health data.

By implementing stratified cross-validation and targeted class-weight balancing, this pipeline successfully elevated baseline **OOF (Out-Of-Fold) Balanced Accuracy from ~0.88 to over 0.94**.

---

## 📊 Dataset Overview

The dataset contains various demographic, clinical, and behavioral features of individuals:

*   **Target:** `health_condition` (Classes: `fit`, `at-risk`, `unhealthy`)
*   **Numerical Features:** Sleep duration, heart rate, BMI, calorie expenditure, step count, exercise duration, water intake.
*   **Categorical Features:** Diet type, stress level, sleep quality, physical activity level, smoking/alcohol habits, gender.

### ⚠️ Class Imbalance Challenge
The training dataset suffers from an intense class imbalance:
*   **at-risk**: ~85.9%
*   **unhealthy**: ~8.4%
*   **fit**: ~5.8%

To combat this, the pipeline relies on **Balanced Accuracy** as the core metric and utilizes LightGBM's native `class_weight='balanced'` parameter.

---

## 🛠️ Pipeline Architecture

This repository is organized into a clean, reproducible machine learning workflow:

├── data/
│   ├── train.csv
│   └── test.csv
├── baseline.ipynb         # Main prototyping, tuning, and evaluation notebook
├── submission.csv         # Generated test predictions
└── README.md              # Project documentation

### Key Workflow Stages:
1. **Target & Categorical Encoding:** Automatic type conversion to pass categorical data natively to LightGBM.
2. **Robust Cross-Validation:** 5-Fold Stratified Cross-Validation ensures fold splits maintain identical class ratios.
3. **Automated Hyperparameter Search:** Uses **Optuna** to optimize learning rates, tree depth, and regularization limits over 50 trials.
4. **Ensemble Inference:** Out-of-fold probability tracking and test prediction averaging across all trained fold-models to reduce variance.

---

## 🚀 Performance & Progress

| Model Setup | Mean CV (Balanced Accuracy) | OOF BA |
| :--- | :---: | :---: |
| **Unweighted Baseline** | 0.87939 | 0.87939 |
| **Balanced Class Weights** | 0.94728 | 0.94728 |
| **Optuna Hyper-Tuned Model** | **0.94764** | **0.94764** |

### Optimized Hyperparameters (via Optuna)
```python
{
    'learning_rate': 0.18596955222923703, 
    'num_leaves': 29, 
    'max_depth': 4, 
    'min_child_samples': 11, 
    'subsample': 0.8849762381207056, 
    'colsample_bytree': 0.6492126685477961, 
    'reg_alpha': 2.5742638697547786, 
    'reg_lambda': 0.0013010403739248985
}
```

📈 Visual Insights
1. Confusion Matrix
The normalized confusion matrix shows that class balancing allows the model to predict rare classes (unhealthy and fit) with excellent recall, avoiding the common pitfall of collapsing onto the majority at-risk class.

2. Feature Importance
The LightGBM feature importance maps which health metrics dictate the classification decisions the most (e.g., step counts, BMI, sleep quality).

Quick Start
1. Prerequisites
Install the required packages:
Bash
pip install numpy pandas scikit-learn lightgbm optuna matplotlib seaborn

3. How to Run
Place your data files in a directory named data/ at the root of your project.

Run the baseline.ipynb notebook cell-by-cell.

After execution, your final predictions will be exported as submission.csv in your root folder.

Future Enhancements
Feature Engineering: Create synthetic health ratios (e.g., calories_per_step or sleep_efficiency indicators).

Prior Correction: Test tuning raw predictions via threshold post-processing instead of class_weight='balanced'.

Model Ensembling: Train companion CatBoost and XGBoost models and soft-vote their predictions.  
