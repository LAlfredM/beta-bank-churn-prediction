# Beta Bank — Customer Churn Prediction

**Project Type:** Supervised Machine Learning · Binary Classification  
**Primary Metric:** F1 Score (Target: $\ge 0.59$)  
**Secondary Metric:** AUC-ROC  

---

## Overview & Business Context

Customer retention is a critical driver of profitability in retail banking, as retaining existing clients is significantly more cost-effective than acquiring new ones. **Beta Bank** has experienced a gradual, month-over-month increase in customer attrition. 

The objective of this project is to develop a production-ready binary classification model capable of identifying customers at high risk of churning (`Exited = 1`) based on historical behavioral, demographic, and account data. Accurate early detection enables the retention team to deploy targeted interventions—such as personalized offers, proactive outreach, and product reviews—to maximize customer lifetime value.

---

## Dataset Summary

The dataset comprises **10,000 customer records** across **14 features**:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `RowNumber` | Identifier | Record index (dropped prior to modeling) |
| `CustomerId` | Identifier | Unique customer ID (dropped prior to modeling) |
| `Surname` | Text | Customer surname (dropped to prevent overfitting and demographic bias) |
| `CreditScore` | Numeric | Customer credit score (350–850) |
| `Geography` | Categorical | Country of residence (`France`, `Germany`, `Spain`)[cite: 1] |
| `Gender` | Categorical | Customer gender (`Female`, `Male`)[cite: 1] |
| `Age` | Numeric | Customer age in years (18–92)[cite: 1] |
| `Tenure` | Numeric | Years of fixed-term deposit maturity (0–10 years)[cite: 1] |
| `Balance` | Numeric | Current account balance[cite: 1] |
| `NumOfProducts` | Numeric | Number of bank products held (1–4)[cite: 1] |
| `HasCrCard` | Binary | Credit card ownership (`1` = Yes, `0` = No)[cite: 1] |
| `IsActiveMember` | Binary | Active engagement status (`1` = Yes, `0` = No)[cite: 1] |
| `EstimatedSalary` | Numeric | Estimated annual salary[cite: 1] |
| **`Exited`** | **Target** | **Churn status (`1` = Churned, `0` = Retained)**[cite: 1] |

---

## Key Exploratory & Data Quality Findings

* **Data Integrity:** Automated schema and value-range validation confirmed zero duplicate records and clean categorical/binary encodings[cite: 1].
* **Missing Value Analysis:** `Tenure` was the only feature with missing entries (909 records, ~9.1%)[cite: 1]. Comparative analysis showed virtually identical churn rates (~20.1% vs. ~20.4%) and feature distributions between missing and non-missing segments, indicating data is Missing Completely at Random (MCAR)[cite: 1]. Missing values were imputed using the **training set median** post-split to prevent data leakage[cite: 1].
* **Key Churn Drivers:**
  * **Age:** Churners skew significantly older (concentrated between ages 40–60) compared to retained customers (peaking in their 30s)[cite: 1].
  * **Account Balance:** Roughly 36.2% of clients hold a zero balance; however, retained clients represent a higher proportion of zero-balance accounts (39.1%) than churners (24.5%), indicating attrition is occurring among funded accounts[cite: 1].

---

## Methodology & Modeling Pipeline

1. **Data Preparation & Leakage Prevention:**
   * Removed non-predictive identifiers (`RowNumber`, `CustomerId`, `Surname`)[cite: 1].
   * Split data into stratified Training, Validation, and Test sets (`RANDOM_STATE = 12345`)[cite: 1].
   * Applied median imputation for `Tenure`, one-hot encoding for categorical variables, and feature scaling (`StandardScaler`) strictly fitted on the training partition[cite: 1].
2. **Baseline Evaluation:**
   * Quantified target class imbalance (~20.4% positive churn class)[cite: 1].
   * Trained unadjusted baseline classifiers (`DummyClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`) to establish reference performance[cite: 1].
3. **Class Imbalance Mitigation & Optimization:**
   * Evaluated three distinct imbalance-handling techniques: **Class Weighting** (`class_weight='balanced'`), **Random Upsampling**, and **Random Downsampling**[cite: 1].
   * Conducted hyperparameter tuning on the validation set alongside **decision-threshold optimization** to maximize the **F1 score** while monitoring **AUC-ROC**[cite: 1].
4. **Final Evaluation:**
   * Evaluated the top-performing model once on the unseen test set against the business threshold ($\text{F1} \ge 0.59$)[cite: 1].

---

## Repository Structure

```text
├── data/
│   └── Churn.csv                  # Input dataset
├── notebooks/
│   └── beta_bank_churn.ipynb      # End-to-end EDA, preprocessing, modeling & evaluation
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation