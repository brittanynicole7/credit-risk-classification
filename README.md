# Logistic Regression Challenge

# Project Description 

## Step 1: Split the Data into Training and Testing Sets
- Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.
- Create the labels set "y" from the loan_status column and create the features "X" DataFrame from the remaining columns.
- Split the data into training and testing datasets using train_test_split.

## Step 2: Create a Logistic Regression Model
- Fit a logistic regression model by using the training data (X_train and y_train).
- Save the predictions on the testing data labels by using X_test and the fitted model.
- Evaluate the model's performance by generating a confusion matrix, generating a classification report, and answering how well the logistic regression predicts the healthy and high-risk loan labels. 

## Step 3: Write a Credit Risk Analysis Report
- Provide an overview that explains the purpose of this analysis.
- Describe the accuracy, precision, and recall scores of the machine learning model, and summarize the results from the machine learning model and include any justification for recommending/not recommending the model to the company. 

# Software and Files
- import numpy as np
- import pandas as pd
- from pathlib import Path
- from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
- from sklearn.linear_model import LogisticRegression
- from imblearn.over_sampling import RandomOverSampler
- CSV: https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv

# Output/Analyses

## Step 1: Split the Data into Training and Testing Sets
- Read the lending_data.csv into a Pandas DataFrame.
<img width="733" alt="Screenshot 2023-05-24 at 6 37 12 PM" src="https://github.com/brittanynicole7/credit-risk-classification/assets/119909433/2090d35a-807a-4238-a516-ef370d12cc43">

- Created the labels set "y" from the loan_status column and created the features "X" DataFrame from the remaining columns. 
<img width="736" alt="Screenshot 2023-05-24 at 6 38 18 PM" src="https://github.com/brittanynicole7/credit-risk-classification/assets/119909433/fe233cb0-4ef7-49e8-bd2d-5c6fbd029901">
<img width="743" alt="Screenshot 2023-05-24 at 6 38 41 PM" src="https://github.com/brittanynicole7/credit-risk-classification/assets/119909433/35d6dd05-0380-4385-b7e0-8578b143d39c">

- Split the data into training and testing datasets using train_test_split.
<img width="747" alt="Screenshot 2023-05-24 at 6 39 32 PM" src="https://github.com/brittanynicole7/credit-risk-classification/assets/119909433/ec75897d-7c20-42e5-b3be-d262d1586590">

## Step 2: Create a Logistic Regression Model
- Fit a logistic regression model using the training data. 
<img width="731" alt="Screenshot 2023-05-24 at 7 40 47 PM" src="https://github.com/brittanynicole7/credit-risk-classification/assets/119909433/c10d11ba-7025-4a3a-badf-489793e45f97">

- Saved the predictions on the testing data labels by using the testing feature dataset and the fitted model. 
<img width="736" alt="Screenshot 2023-05-24 at 7 41 07 PM" src="https://github.com/brittanynicole7/credit-risk-classification/assets/119909433/698848c6-dbb5-48ae-9ab6-9410ee851c90">

- Evaluated the model's performance by generating a confusion matrix and generating a classification report. ***
<img width="739" alt="Screenshot 2023-05-24 at 7 41 24 PM" src="https://github.com/brittanynicole7/credit-risk-classification/assets/119909433/134c4b81-fa00-4ba7-a35b-1d9de8e7fa6f">


## Step 3: Write a Credit Risk Analysis Report

## Overview of the Analysis
 The purpose of this analysis was to create machine learning models to predict healthy and high risk loans. Independent variables in the dataset included loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogatory marks, total debt and the dependent variable in the model was loan status (i.e., healthy versus high risk loans via value_counts). For this analysis, I used a logistic regression initially and the accuracy was really high (95.2%) but there were too many false positives and negatives using this model and the recall, precision, and f1-scores for predicting high-risk loans were not high enough. Given these considerations, I used the RandomOverSampler module and the acccuracy, number of false positives/negatives, and precision, recall, and f1-scores all improved. 

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * For model 1, using just a logistic regression, the accuracy score was high at 95.2%, there were high precision (healthy loan- 1.00, high-risk loan-0.85), recall (healthy loan- 0.99, high-risk loan- 0.91), and f1-scores (healthy loan-1.00, high-risk loan-0.88).

* Machine Learning Model 2:
  * For model 2, using random oversampler and a logistic regression, the accuracy score was even higher at 99.4%, there were higher precision (healthy loan- 1.00, high-risk loan-0.84), higher recall (healthy loan- 0.99, high-risk loan- 0.99), and higher f1-scores (healthy loan-1.00, high-risk loan-0.91).

## Summary
 The model that seems to perform best is the model that used random oversampling because the accuracy, recall, and precision scores were all higher and the number of false positives/negatives decreased. In the case of this dataset, it is more important to accurately predict high-risk loans and to refrain from giving loans to individuals that are at high risk of defaulting on the loan so in this case, the second model is more efficient as the precision, recall, and accuracy scores all improve. 

# Author 
-Brittany Wright github:brittanynicole7
