# Telco Customer Churn Prediction & Segmentation

## Executive Summary
This project predicts telecom customer churn and groups customers into actionable retention segments. The final solution uses a tuned XGBoost classifier for churn prediction and K-Means clustering for segmentation, with the churn decision threshold optimized on a validation set to improve recall without introducing leakage.

The final notebook implements the roadmap from `churn_model_analysis.md` and reaches the recall goal on a held-out test set:
- Recall improved from `74.75%` to `85.00%`
- ROC-AUC remained strong at `0.8574`
- Average Precision reached `0.6720`

## Business Questions
- Which customers are most likely to churn?
- How can the business group customers for more targeted retention actions?
- What actions should be prioritized to reduce churn risk early?

## Dataset
- Source file: `Telco_customer_churn.xlsx`
- Records: `7,043`
- Raw columns: `33`

The dataset includes customer demographics, services, contract details, billing behavior, tenure, charges, and churn outcomes.

## What Changed
Using the recommendations from `churn_model_analysis.md`, the project now applies:
1. Leakage-safe feature selection by continuing to drop `Churn Score`, `CLTV`, and `Churn Reason`
2. Domain-driven engineered features such as:
   - `Charge_per_Month_Ratio`
   - `HighRisk_Contract_Internet`
   - `Num_Services`
   - `Digital_Risk`
   - `Is_New_Customer`
3. A stratified `train / validation / test` split
4. Validation-based threshold tuning to maximize precision while enforcing recall `>= 80%`

## Final Modeling Workflow
1. Load and clean the dataset
2. Engineer targeted churn features
3. One-hot encode categorical variables
4. Split data into stratified train, validation, and test sets
5. Benchmark Random Forest against XGBoost
6. Tune the XGBoost decision threshold on the validation set
7. Evaluate the final selected model on the held-out test set
8. Use predicted churn probabilities for customer segmentation

## Final Selected Model
- Model: `XGBClassifier`
- Parameters:
  - `n_estimators=500`
  - `max_depth=3`
  - `learning_rate=0.05`
  - `subsample=0.8`
  - `colsample_bytree=1.0`
  - `scale_pos_weight=2.5`
  - `min_child_weight=1`
  - `gamma=0`
  - `random_state=42`
- Validation-selected threshold: `0.42`

## Final Test Metrics
| Metric | Value |
|---|---:|
| Accuracy | 74.55% |
| Precision | 51.18% |
| Recall | 85.00% |
| F1 Score | 63.89% |
| ROC-AUC | 0.8574 |
| Average Precision | 0.6720 |

## Benchmark Comparison
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline Dummy Classifier | 73.51% | 0.00% | 0.00% | 0.00% | 0.5000 |
| Random Forest Benchmark | 78.24% | 57.17% | 74.29% | 64.51% | 0.8580 |
| XGBoost Default Threshold | 77.11% | 56.15% | 75.00% | 64.52% | 0.8574 |
| Final XGBoost Tuned Threshold | 74.55% | 51.18% | 85.00% | 63.89% | 0.8574 |

The final model intentionally trades some precision and accuracy for a much stronger churn-capture rate, which is appropriate when the business goal is to identify more customers before they leave.

## Main Churn Drivers
Top drivers in the final XGBoost model include:
- `HighRisk_Contract_Internet`
- `Contract_Two year`
- `Internet Service_No`
- `Internet Service_Fiber optic`
- `Dependents_Yes`

At a business level, the strongest themes are:
- Early-tenure customers are more fragile
- Month-to-month contract structures increase churn risk
- Fiber optic and internet-service mix matter
- Billing behavior and digital payment patterns are meaningful churn signals

## Customer Segmentation
The final churn probabilities are used in K-Means segmentation to create three business personas.

| Segment | Customers | Share | Avg Tenure | Avg Monthly Charges | Avg Churn Probability |
|---|---:|---:|---:|---:|---:|
| High Risk New Customer | 2,539 | 36.0% | 10.99 | 72.99 | 0.72 |
| Budget Loyal Customer | 2,377 | 33.7% | 31.80 | 39.43 | 0.14 |
| Loyal Premium Customer | 2,127 | 30.2% | 58.54 | 96.43 | 0.23 |

## Business Recommendations
### High Risk New Customer
- Launch early-life retention offers within the first 3 to 6 months
- Use proactive onboarding and support check-ins
- Offer pricing reassurance or bundle incentives for higher-charge customers

### Budget Loyal Customer
- Protect value perception with low-cost loyalty rewards
- Use renewal nudges and referral campaigns
- Promote add-ons selectively without hurting affordability

### Loyal Premium Customer
- Prioritize premium support and service-quality communication
- Use targeted upsell bundles instead of broad discounting
- Monitor service issues closely because this segment carries higher revenue value

## Repository Files
- [Final_CBSOT_Customer_segmentation.ipynb](./Final_CBSOT_Customer_segmentation.ipynb): final notebook deliverable
- [final_cbsot_customer_segmentation.py](./final_cbsot_customer_segmentation.py): synced script version
- [churn_model_analysis.md](./churn_model_analysis.md): review and improvement roadmap
- [Telco_customer_churn.xlsx](./Telco_customer_churn.xlsx): source dataset

## How to Run
1. Open `Final_CBSOT_Customer_segmentation.ipynb` in Jupyter or VS Code.
2. Keep `Telco_customer_churn.xlsx` in the same project folder.
3. Ensure `xgboost`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn` are installed.
4. Run the notebook from top to bottom.

## Portfolio Value
This project demonstrates:
- End-to-end churn modeling with business framing
- Leakage-aware model improvement
- Threshold tuning for objective-driven classification
- Customer segmentation tied to operational recommendations
- Executive-ready reporting and visual storytelling
