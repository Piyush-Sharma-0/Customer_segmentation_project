
# Telco Customer Churn Prediction & Segmentation

## Overview
This project aims to analyze customer churn in a telecommunications company, build a predictive model to identify customers at risk of churning, and segment the customer base for targeted retention strategies. By understanding the factors driving churn and categorizing customers, the company can proactively reduce churn and optimize marketing efforts.

## Dataset
The analysis uses the `Telco_customer_churn.xlsx` dataset, which contains customer demographic information, services subscribed, monthly charges, total charges, tenure, and churn status. 

## Methodology
The project follows a comprehensive methodology:

1.  **Data Cleaning & Feature Engineering**:
    *   'Total Charges' were converted to numeric, with `coerce` handling non-numeric values and `fillna(0)` for new customers (tenure 0).
    *   Irrelevant identifier columns (`CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`, `Churn Label`, `Churn Score`, `Churn Reason`, `CLTV`) were dropped.
    *   Categorical features were one-hot encoded using `pd.get_dummies`.

2.  **Exploratory Data Analysis (EDA)**:
    *   Distributions of `Tenure Months`, `Monthly Charges`, `Contract`, `Internet Service`, and `Payment Method` were analyzed.
    *   Relationships between these features and `Churn Label` were visualized using box plots, KDE plots, and count plots.
    *   A correlation matrix for numerical features was generated to understand inter-variable relationships.
    *   **Key EDA Insights**:
        *   Customers with lower tenure and higher monthly charges show a higher propensity to churn.
        *   Customers on month-to-month contracts have a significantly higher churn rate compared to those with one-year or two-year contracts.
        *   `Fiber optic` internet service and `Electronic check` payment methods are associated with higher churn.

3.  **Machine Learning Pipeline (Churn Prediction)**:
    *   The target variable `Churn Value` was separated from features `X`.
    *   Data was split into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`) using `train_test_split`.
    *   **XGBoost Model with SMOTE and Hyperparameter Tuning**:
        *   Class imbalance was addressed using `SMOTE` (Synthetic Minority Over-sampling Technique) on the training data.
        *   An `XGBClassifier` was used with hyperparameter tuning via `GridSearchCV`.
        *   The `scoring` metric for `GridSearchCV` was set to `'recall'` to prioritize the identification of churners.
        *   Optimal parameters found were: `learning_rate=0.01`, `max_depth=4`, `n_estimators=300`, `scale_pos_weight=3`.
    *   The final model achieves strong performance in identifying churners.

4.  **Customer Segmentation (K-Means)**:
    *   A `segmentation_data` DataFrame was created using `Tenure Months`, `Monthly Charges`, `Total Charges`, and the `Churn Probability` derived from the trained XGBoost model.
    *   The data was scaled using `StandardScaler`.
    *   The Elbow Method was applied to determine the optimal number of clusters, which indicated `k=3`.
    *   K-Means clustering was performed to group customers into segments.
    *   Segments were mapped to business personas: 'Budget Loyal Customer', 'High Risk New Customer', and 'Loyal Premium Customer'.
    *   Cluster characteristics were summarized, and segments were visualized based on 'Tenure Months' vs 'Churn Probability' and 'Monthly Charges' vs 'Churn Probability'.

## Key Findings & Business Insights
*   **High Churn Risk for New Customers**: Customers in their early months of service (`Tenure Months` < 20) with high churn probability are a critical group to target.
*   **Impact of Contract Type and Internet Service**: Month-to-month contracts and Fiber Optic internet are strong indicators of churn.
*   **Predictive Power**: The optimized XGBoost model effectively identifies 91% of potential churners (recall score).
*   **Actionable Segments**:
    *   **High Risk New Customers**: Likely new customers with high churn probability and potentially high monthly charges. Requires early intervention and personalized onboarding.
    *   **Budget Loyal Customers**: Lower monthly charges, lower churn probability, and longer tenure. Focus on retention through value-for-money offerings.
    *   **Loyal Premium Customers**: Higher monthly charges, longer tenure, and moderate churn probability. Focus on premium service and upselling opportunities.

## Results Summary
*   **Model Recall**: 90.50% (Ability to correctly identify churners)
*   **Model Accuracy**: 69.55% (Overall correctness of predictions)

**Business Segmentation Breakdown**:
*   High Risk New Customer: 2580 customers (36.6%)
*   Budget Loyal Customer: 2312 customers (32.8%)
*   Loyal Premium Customer: 2151 customers (30.5%)

**Value Proposition**:
The model effectively identifies 91% of potential churners, allowing the marketing team to target interventions before revenue loss occurs. The segmentation further refines this by distinguishing between 'High Risk' and 'Loyal' profiles, enabling tailored strategies.

## How to Run the Notebook
1.  **Upload Data**: Ensure the `Telco_customer_churn.xlsx` file is uploaded to your Colab environment or the working directory.
2.  **Install Libraries**: The notebook uses standard libraries. If any are missing, install them using `pip install <library_name>`.
3.  **Run Cells Sequentially**: Execute all code cells in the notebook from top to bottom. The notebook is structured to flow logically through data loading, cleaning, EDA, model training, evaluation, and segmentation.
