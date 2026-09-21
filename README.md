# Credit Risk Modelling and Prediction

## Project Overview

This project develops a machine learning based credit risk classification system to predict whether a loan applicant is likely to be classified as **GOOD** or **BAD** credit risk.

The project covers the complete machine learning workflow, including data exploration, preprocessing, model development, hyperparameter tuning, threshold analysis, model evaluation, feature importance analysis, and deployment through a Streamlit application.

## Business Problem

Financial institutions need to assess the creditworthiness of loan applicants before approving credit.

Incorrectly approving a high-risk applicant can lead to financial losses, while incorrectly rejecting a low-risk applicant can result in lost business opportunities.

Therefore, this project focuses on developing a classification model that can help identify applicants with a higher probability of **BAD credit risk**.

## Objectives

* Explore and understand the credit risk dataset.
* Perform data cleaning and preprocessing.
* Build multiple classification models.
* Compare model performance using appropriate evaluation metrics.
* Tune the Random Forest model.
* Analyze the precision-recall trade-off for BAD-risk applicants.
* Identify the most important predictors of credit risk.
* Build a reusable preprocessing and prediction pipeline.
* Deploy the model using Streamlit.

## Dataset

The project uses the **German Credit dataset**.

The dataset contains information about loan applicants, including:

* Age
* Sex
* Job
* Housing
* Saving accounts
* Checking account
* Credit amount
* Duration
* Purpose
* Risk

The target variable is **Risk**, with two classes:

* `good`
* `bad`

## Machine Learning Workflow

The project follows this workflow:

1. Data loading
2. Exploratory Data Analysis
3. Missing-value analysis
4. Data preprocessing
5. Train-test split
6. Model development
7. Model comparison
8. Hyperparameter tuning
9. Threshold analysis
10. Precision-recall analysis
11. Feature importance analysis
12. Final model pipeline
13. Streamlit deployment

## Models

The following classification models were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Extra Trees

Random Forest was subsequently tuned using cross-validation.

## Model Evaluation

Model performance was evaluated using:

* Accuracy
* BAD-risk Precision
* BAD-risk Recall
* BAD-risk F1-score
* ROC-AUC
* Average Precision
* Confusion Matrix

Special attention was given to **BAD-risk recall**, because identifying applicants who may represent higher credit risk is important in a credit-risk setting.

## Final Model Results

The tuned Random Forest achieved the following results at the selected classification threshold:

| Metric            | Score |
| ----------------- | ----: |
| Accuracy          | 0.570 |
| BAD Precision     | 0.395 |
| BAD Recall        | 0.817 |
| BAD F1            | 0.533 |
| ROC-AUC           | 0.666 |
| Average Precision | 0.461 |

The classification threshold was adjusted from the default 0.50 to improve the detection of BAD-risk applicants.

At the selected threshold:

* BAD-risk recall increased substantially.
* BAD-risk precision decreased.
* This demonstrates the trade-off between identifying more potentially risky applicants and generating more false positives.

## Threshold Analysis

Different classification thresholds were evaluated to understand the precision-recall trade-off.

The analysis showed that lowering the threshold increases BAD-risk recall but also increases the number of applicants incorrectly classified as BAD.

This demonstrates that the classification threshold should depend on the business objective and the relative cost of false positives and false negatives.

## Feature Importance

The Random Forest feature importance analysis identified several influential variables.

The most important features in the final model included:

1. Credit amount
2. Duration
3. Age
4. Purpose
5. Checking account
6. Job
7. Housing
8. Saving accounts

The results suggest that **credit amount, loan duration, and age** were particularly influential in the model's predictions.

Feature importance indicates how much a feature contributes to the model's predictive decisions; it should not be interpreted as proof of a causal relationship.

## Machine Learning Pipeline

A Scikit-learn Pipeline was created to combine preprocessing and the final Random Forest model.

This allows new applicant data to pass through the same preprocessing steps used during model development before generating predictions.

The trained pipeline is saved using `joblib`.

## Streamlit Application

A Streamlit application was developed to provide an interactive interface for credit risk prediction.

The application allows the user to enter applicant information such as:

* Age
* Sex
* Job
* Housing
* Saving accounts
* Checking account
* Credit amount
* Loan duration

The application then produces a credit-risk classification.

## Project Structure

```text
CREDIT_RISK_PROJECT/
│
├── app/
│   └── app.py
│
├── Data/
│   └── german_credit_data.csv
│
├── models/
│   └── credit_risk_pipeline.pkl
│
├── notebooks/
│   └── credit_risk_analysis.ipynb
│
├── src/
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit
* Jupyter Notebook

## How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the project

```bash
cd CREDIT_RISK_PROJECT
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app/app.py
```

The application will open in the browser.

## Limitations

This project is an academic/portfolio prototype and should not be treated as a production credit approval system.

The dataset is relatively small, and model performance may not generalize to a real financial institution's customer population.

A production credit-risk system would require additional validation, monitoring, governance, explainability, fairness assessment, and domain-specific risk controls.

## Future Improvements

Possible future improvements include:

* Probability calibration
* More advanced models such as XGBoost
* SHAP-based model explainability
* Cost-sensitive learning
* Additional credit-related variables
* Model monitoring
* Fairness analysis
* External validation on another dataset
* Deployment through a cloud platform

## Disclaimer

This project is intended for educational and portfolio purposes. The predictions generated by the application should not be used as the sole basis for real-world lending decisions.
