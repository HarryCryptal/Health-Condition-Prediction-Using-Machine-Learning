# Health Condition Classification Using Machine Learning

An end-to-end machine learning pipeline for multi-class health condition classification using **LightGBM**, **Optuna**, and **Stratified Cross-Validation**.

The objective of this project is to predict three health condition categories:

- `fit`
- `at-risk`
- `unhealthy`

using demographic, behavioral, and lifestyle-related features.

The project focuses on building a reliable classification pipeline while addressing challenges such as **class imbalance**, **model evaluation**, and **generalization**.

---

# Dataset Overview

The dataset contains demographic and lifestyle features describing individuals.

## Target Variable

`health_condition`

Classes:

- Fit
- At-risk
- Unhealthy


## Features

### Numerical Features

- Sleep duration
- Heart rate
- BMI
- Calorie expenditure
- Step count
- Exercise duration
- Water intake


### Categorical Features

- Diet type
- Stress level
- Sleep quality
- Physical activity level
- Smoking/alcohol habits
- Gender


---

# Class Imbalance Challenge

The dataset contains significant imbalance between classes:

| Class | Percentage |
|---|---:|
| At-risk | ~85.9% |
| Unhealthy | ~8.4% |
| Fit | ~5.8% |

Because accuracy can be misleading under heavy imbalance, this project uses:

- **Balanced Accuracy**
- **Stratified Cross-Validation**
- **LightGBM class weighting**

to ensure minority classes receive appropriate attention during training.


---

# Machine Learning Pipeline

```
health-condition-classification/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── lightGBM.ipynb
│
├── submission.csv
│
└── README.md
```

## Workflow

### 1. Data Preparation

- Loaded and inspected dataset distributions.
- Converted categorical variables into native LightGBM categorical features.
- Separated numerical and categorical feature groups.


### 2. Stratified Cross-Validation

Implemented **5-Fold Stratified Cross-Validation** to maintain consistent class distributions across validation folds.

Benefits:

- More reliable performance estimation.
- Reduced variance from a single train/validation split.


### 3. Handling Class Imbalance

Compared:

- Baseline LightGBM model.
- Class-weight balanced LightGBM model.
- Hyperparameter optimized model.


### 4. Hyperparameter Optimization

Used **Optuna** to optimize:

- Learning rate
- Number of leaves
- Tree depth
- Minimum child samples
- Feature sampling
- Regularization parameters


### 5. Ensemble Prediction

Generated predictions by:

- Training a model on each CV fold.
- Collecting out-of-fold predictions.
- Averaging test predictions across fold models.


---

# Model Performance

| Model | Balanced Accuracy |
|---|---:|
| Baseline LightGBM | 0.879 |
| Class Weighted LightGBM | 0.947 |
| Optuna Tuned LightGBM | **0.948** |


The results demonstrate that addressing class imbalance significantly improves minority-class recognition compared to an unweighted baseline model.


---

# Model Analysis

## Confusion Matrix

The normalized confusion matrix was used to analyze prediction behavior across all three classes.

Key observation:

- Class weighting reduced the tendency of the model to over-predict the majority `at-risk` class.
- Minority classes received improved representation during prediction.


## Feature Importance

LightGBM feature importance was analyzed to understand which variables contributed most to classification decisions.

Important features included:

- Step count
- BMI
- Sleep quality
- Exercise duration

---

# Responsible AI Considerations

Although this project focuses on predictive modeling, health-related machine learning systems require careful consideration.

Important limitations include:

- The dataset is synthetic and may not represent real-world populations.
- Feature importance does not imply causation.
- Predictions should not be interpreted as medical diagnoses.
- Models trained on imbalanced datasets may behave differently across demographic groups.

Future work could explore:

- Fairness evaluation across demographic groups.
- Model interpretability techniques such as SHAP.
- Robustness testing under distribution shifts.


---

# Installation

Install required dependencies:

```bash
pip install numpy pandas scikit-learn lightgbm optuna matplotlib seaborn
```


---

# Running the Project

1. Place dataset files inside:

```
data/
├── train.csv
└── test.csv
```

2. Open:

```
lightGBM.ipynb
```

3. Run all cells.

The generated predictions will be saved as:

```
submission.csv
```


---

# Future Improvements

Potential improvements include:

- Additional feature engineering:
  - Calories per step
  - Exercise consistency metrics
  - Sleep efficiency indicators

- Prediction calibration and post-processing:
  - Threshold tuning
  - Prior probability correction

- Model comparison:
  - CatBoost
  - XGBoost
  - Ensemble methods

- Explainability:
  - SHAP-based feature analysis


---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Optuna
- Matplotlib
- Seaborn
