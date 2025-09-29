# Loan Approval Prediction – ML Pipeline

## Overview
A machine learning pipeline to predict loan approval using applicant and loan attributes (income, credit score, loan amount, etc.).  
Target: **loan_status (1 = Approved, 0 = Rejected)**.  

##  Dataset
- **Size**: ~45,000 rows, 13 features + target  
- **Type**: Mixed numerical & categorical  
- **Imbalance**: ~78% rejected vs 22% approved  

## Preprocessing
- Median/mode imputation for missing values  
- Label encoding for categorical variables  
- Min-Max scaling for numerical features  
- Stratified train-test split (70/30)  

## Models
- Logistic Regression → Acc: **0.89**  
- KNN → Acc: **0.89**  
- Decision Tree → Acc: **0.90**  
- Neural Network (MLP) → **Best model**, Acc: **0.92**  
- KMeans (unsupervised clustering)  

## Results
- **MLP outperformed all** with best accuracy & AUC  
- Logistic Regression competitive & interpretable  
- Decision Tree close but risk of overfitting  
- KNN sensitive to scaling & imbalance  

## Conclusion
The pipeline effectively predicts loan approvals. Future improvements: class balancing, feature engineering, and hyperparameter tuning.  

