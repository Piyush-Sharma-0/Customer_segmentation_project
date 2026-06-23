# Telco Customer Churn Prediction & Segmentation

## Executive Summary
This project analyzes telecom customer churn, predicts which customers are most likely to leave, and segments the customer base into actionable personas for retention planning. The final solution uses a Random Forest classifier for churn prediction and K-Means clustering for customer segmentation.

The project is designed to answer two business questions:
- Which customers are most likely to churn?
- How can the business group customers for more targeted retention actions?

## Dataset
- Source file: `Telco_customer_churn.xlsx`
- Records: 7,043 customers
- Raw features: 33 columns

The dataset includes customer demographics, contract details, subscribed services, monthly charges, total charges, and churn outcomes.

## Project Workflow
1. Data loading and cleaning
2. Exploratory data analysis
3. Feature engineering with one-hot encoding
4. Random Forest churn modeling
5. K-Means customer segmentation
6. Business interpretation and recommendations

## Final Selected Model
- Model: `RandomForestClassifier`
- Parameters:
  - `n_estimators=300`
  - `max_depth=10`
  - `random_state=42`
  - `class_weight='balanced'`

This final model was preserved from the original project logic and used as the production-ready churn model.

## Key Results
### Model Performance
| Metric | Value |
|---|---:|
| Accuracy | 78.28% |
| Precision | 59.01% |
| Recall | 74.75% |
| F1 Score | 65.95% |
| ROC-AUC | 0.8571 |

### Segment Breakdown
| Segment | Customers | Share |
|---|---:|---:|
| High Risk New Customer | 2,580 | 36.6% |
| Budget Loyal Customer | 2,312 | 32.8% |
| Loyal Premium Customer | 2,151 | 30.5% |

## Main Churn Drivers
The model and EDA consistently point to these factors as the strongest churn signals:
- Lower tenure
- Higher monthly charges
- Contract type, especially month-to-month plans
- Internet service mix
- Billing and payment behavior

## Business Recommendations
### High Risk New Customer
- Deploy early-life retention campaigns in the first 3 to 6 months.
- Strengthen onboarding, service education, and proactive support.
- Use pricing reassurance or targeted bundles for high-charge customers.

### Budget Loyal Customer
- Maintain strong value perception with light-touch loyalty rewards.
- Use renewal nudges and referral campaigns.
- Cross-sell carefully without weakening affordability.

### Loyal Premium Customer
- Prioritize premium support and service quality communication.
- Focus on high-value upsell bundles rather than broad discounting.
- Watch closely for service issues because this segment carries higher revenue value.

## Repository Files
- [final_cbsot_customer_segmentation_clean.ipynb](./final_cbsot_customer_segmentation_clean.ipynb): final notebook deliverable
- [final_cbsot_customer_segmentation.py](./final_cbsot_customer_segmentation.py): cleaned script version
- [Telco_customer_churn.xlsx](./Telco_customer_churn.xlsx): dataset used in the analysis

## How to Run
1. Open `final_cbsot_customer_segmentation_clean.ipynb` in Jupyter or VS Code.
2. Keep `Telco_customer_churn.xlsx` in the same project folder.
3. Run the notebook from top to bottom.

## Portfolio Value
This project demonstrates:
- End-to-end churn analytics
- Applied machine learning for business prediction
- Customer segmentation for strategy design
- Communication of model outputs through executive-ready reporting
