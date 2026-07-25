# Customer Churn Prediction using RFM Features & Logistic Regression

## 📌 Overview
This project extends RFM-based customer segmentation into predictive modeling — 
building a Logistic Regression classifier to predict customer churn based on 
purchasing behavior, enabling proactive retention strategies.

## 🎯 Business Problem
Identifying customers likely to churn *before* they go silent allows a business 
to intervene early with targeted retention campaigns, rather than reacting after 
revenue has already been lost.

## 🛠️ Tools & Skills Used
- **Python** (Pandas, NumPy) — feature engineering
- **Scikit-learn** — Logistic Regression, StandardScaler, train-test split, 
  classification evaluation metrics
- **Data Analysis** — class imbalance handling, precision-recall trade-off analysis

## 📊 Dataset
Built on the same Online Retail Dataset (UCI) used in the RFM segmentation project, 
extended with a churn label.

## 🔍 Approach
1. **Churn Definition:** Labeled customers as "Churned" if inactive for more than 
   180 days (Recency > 180), resulting in 860 churned vs. 3,478 active customers.
2. **Feature Selection:** Used Frequency and Monetary as predictors (Recency was 
   excluded to avoid data leakage, since it was used to define the target itself).
3. **Train-Test Split:** 80/20 stratified split to preserve class distribution.
4. **Model Iteration:**
   - Initial Logistic Regression (unscaled, unbalanced) achieved 80% accuracy but 
     0% recall — the model simply predicted "not churned" for every customer, 
     exposing how misleading accuracy can be on imbalanced data.
   - Applied StandardScaler and `class_weight='balanced'`, improving Recall to 
     **91%** (correctly identifying 156 of 172 churned customers), with a 
     deliberate trade-off of lower precision (35%) and overall accuracy (64%).

## 📈 Key Findings
| Metric | Naive Model | Improved Model |
|---|---|---|
| Accuracy | 80% (misleading) | 64% |
| Recall (Churn) | 0% | **91%** |
| Precision (Churn) | 0% | 35% |

**Feature Importance:** Frequency (coefficient: -5.18) is a far stronger predictor 
of churn than Monetary (coefficient: -0.08) — customers who purchase infrequently 
are significantly more likely to churn, regardless of how much they've spent 
historically.

## 💡 Business Recommendation
Retention efforts should prioritize **encouraging repeat purchases** over simply 
targeting high-spending customers, since purchase frequency is the dominant 
driver of churn risk. A high one-time spend does not protect against churn if 
the customer doesn't return.

## 📁 Files
- `churn_prediction_rfm.ipynb` — Full analysis notebook
