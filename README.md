# Customer Churn Prediction (RFM + Logistic Regression)

A follow-up to my RFM segmentation project. Here I try to predict which customers will churn, using how often they buy and how much they spend.

Dataset: Online Retail (UCI), the same one as the RFM project.

## What I did

- Marked a customer as churned if they hadn't bought in more than 180 days. That gave 860 churned and 3,478 active customers.
- Used Frequency and Monetary as features. I left Recency out because churn was defined from it, so the model would just be reading the answer.
- Split 80/20, stratified.
- First model (no scaling, no balancing) got 80% accuracy but 0% recall. It just predicted "not churned" for everyone.
- Then I added StandardScaler and class_weight='balanced'.

## Results

| | First model | Final model |
|---|---|---|
| Accuracy | 80% | 64% |
| Churn recall | 0% | 91% |
| Churn precision | 0% | 35% |

The final model catches 156 of 172 churned customers. Precision is low, so it flags a lot of customers who wouldn't have left. I chose recall because missing a churner seemed worse than a wasted retention offer.

Frequency mattered much more than Monetary (coefficients -5.18 and -0.08). Customers who buy rarely churn more, even if they spent a lot before.

## Files

- customer_segmentation_rfm_km
