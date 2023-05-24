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




## Step 3: Write a Credit Risk Analysis Report
## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.


# Author 
-Brittany Wright github:brittanynicole7
