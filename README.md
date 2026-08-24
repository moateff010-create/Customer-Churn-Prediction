# 📊 End-to-End Customer Churn Prediction & Model Explainability Pipeline

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Business Problem Statement](#-business-problem-statement)
- [Dataset & Data Preprocessing](#-dataset--data-preprocessing)
- [Methodology & Model Architecture](#-methodology--model-architecture)
- [Experimental Results & Evaluation](#-experimental-results--evaluation)
- [Model Interpretability via SHAP](#-model-interpretability-via-shap)
- [Strategic Business Recommendations](#-strategic-business-recommendations)
- [Installation & How to Run](#-installation--how-to-run)

---

## 📌 Project Overview
This repository contains a complete, production-ready Machine Learning pipeline designed to forecast customer churn for a Telecommunications provider. Using **XGBoost Classifier** integrated with **SHAP (SHapley Additive exPlanations)**, the framework goes beyond raw predictions to deliver actionable insights on *why* customers are leaving, empowering retention teams to execute targeted campaigns with measurable ROI.

---

## 🎯 Business Problem Statement
Acquiring new customers in the telecom sector costs significantly more than retaining existing ones. Customer attrition directly impacts Recurring Revenue (ARR) and Customer Lifetime Value (CLTV). 

### Key Business Goals:
* **Early Detection:** Identify high-risk churners before they cancel their subscriptions.
* **Recall Prioritization:** Maximize the coverage of true churners (**Recall >= 80%**) to ensure high-risk clients receive retention offers.
* **Interpretability:** Explain individual risk factors using SHAP to enable personalized retention interventions (e.g., plan upgrades vs. technical support).

---

## 🧹 Dataset & Data Preprocessing

### 1. Data Cleaning & Feature Selection
* **Data Leakage Mitigation:** Features recorded *after* a churn event occurs—such as `Churn Label`, `Churn Reason`, and `Churn Score`—were strictly removed prior to modeling to prevent data leakage.
* **Categorical Encoding:** Applied One-Hot Encoding to categorical attributes (e.g., `Contract`, `Internet Service`, `Payment Method`).
* **Missing Values Handling:** Imputed numerical features and handled missing values cleanly across non-internet features.

### 2. Handling Class Imbalance
The dataset suffered from a heavy class imbalance (~26% Churn vs. ~74% Non-Churn). To fix this without artificially destroying or synthesizing data:
$$\text{scale\_pos_weight} = \frac{\text{Count of Non-Churners}}{\text{Count of Churners}}$$
This parameter was passed directly into XGBoost to heavily penalize false negatives on the minority class.

---

## 🏗️ Methodology & Model Architecture

### 1. Model Selection
* **Algorithm:** **XGBoost (Extreme Gradient Boosting)**
* **Hyperparameters:**
  * `n_estimators`: 100
  * `learning_rate`: 0.05 (prevents overfitting)
  * `max_depth`: 5 (balances tree complexity)
  * `scale_pos_weight`: Dynamically calculated ratio
  * `eval_metric`: Logloss

### 2. Custom Decision Threshold Tuning
The standard decision threshold of `0.50` yields sub-optimal recall for imbalanced business objectives. We adjusted the classification boundary to **`0.45`**:
$$\hat{y} = \begin{cases} 1 & \text{if } P(\text{Churn}) \ge 0.45 \\ 0 & \text{if } P(\text{Churn}) < 0.45 \end{cases}$$

---

## 📈 Experimental Results & Evaluation

The model was evaluated on a held-out test set ($20\%$ split) using standard classification metrics:

| Metric | Score | Business & Technical Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC Score** | **`85.5%`** | Excellent overall discriminative power between churners and non-churners. |
| **Recall (Churn = 1)** | **`83.0%`** | Captures **83 out of 100 actual churners**, fulfilling the primary business target. |
| **Precision (Churn = 1)**| **`51.0%`** | Over half of flagged customers churn, reducing wasted promotional spending. |
| **F1-Score (Churn = 1)** | **`0.63`** | Strong harmonic mean balance for an imbalanced classification problem. |
| **Overall Accuracy** | **`74.0%`** | Intentionally balanced to prioritize high recall over raw accuracy bias. |

---

## 🔍 Model Interpretability via SHAP

Rather than treating XGBoost as a black-box model, **TreeSHAP** was implemented to quantify the marginal feature attribution for every customer.

### Top Feature Drivers:
1. **Contract Type (`Contract_Two year`):** Strongest negative impact on churn (dramatically lowers churn probability).
2. **Tenure (`Tenure Months`):** Strong negative correlation; lower tenure significantly increases churn risk.
3. **Internet Infrastructure (`Internet Service_Fiber optic`):** Positive impact on churn (customers on Fiber Optic churn at a higher rate).

---

## 💡 Strategic Business Recommendations

Based on empirical model drivers, the retention strategy should focus on three core initiatives:

1. **Long-Term Contract Incentives:**
   * **Finding:** Month-to-month contracts are the leading predictor of churn.
   * **Action:** Offer a $10–15\%$ bill discount for customers who transition from month-to-month to 1- or 2-year commitments.

2. **First 90-Days Onboarding Program:**
   * **Finding:** Churn risk drops exponentially after the first year of tenure.
   * **Action:** Deploy automated onboarding touchpoints and proactive customer support check-ins during the first 3 months.

3. **Fiber Optic Service & Pricing Audit:**
   * **Finding:** Fiber subscribers churn more frequently than DSL users.
   * **Action:** Investigate potential service instability, network outages, or uncompetitive pricing in the Fiber Optic tier.

---

## 💻 Installation & How to Run

### Prerequisites
Ensure you have Python 3.9+ installed along with the required libraries.

```bash
# Clone the repository
git clone [https://github.com/moateff010-create/Customer-Churn-Prediction.git](https://github.com/moateff010-create/Customer-Churn-Prediction.git)
cd Customer-Churn-Prediction

# Install dependencies
pip install xgboost shap scikit-learn pandas matplotlib
