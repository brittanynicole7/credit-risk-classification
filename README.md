# Logistic Regression Challenge

# Project Description 

## Step 1: Split the Data in Training and Testing Sets
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
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import StandardScaler
- import pandas as pd
- import tensorflow as tf
- import pandas as pd 
- CSV: https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv

# Output/Analyses

## Step 1: Preprocess the Data
- Created a DataFrame containing the charity_data.csv and identified the target and feature dataset.
<img width="1439" alt="Screenshot 2023-05-24 at 3 01 38 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/ad89cbcc-cdae-462e-b5eb-12f55112ae0e">
- Dropped the EIN and NAME columns.
<img width="1444" alt="Screenshot 2023-05-24 at 3 02 05 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/a6d93eb6-d1e4-44b2-8fbf-bfc20675443d">
- Determined the number of unique values in each column.
<img width="1447" alt="Screenshot 2023-05-24 at 3 02 29 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/e76a16d1-d4b0-49eb-8790-13f9ca722951">
- For columns with more than 10 unique values, determined the number of data points for each unique value. 
<img width="1436" alt="Screenshot 2023-05-24 at 3 02 52 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/41759676-edd8-44c3-a02e-edef93647073">
<img width="1441" alt="Screenshot 2023-05-24 at 3 03 40 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/fc2ad64c-88b8-40ce-b835-9cfddb766b50">
- Created a new value called Other that contains rare categorical variables. 
<img width="1442" alt="Screenshot 2023-05-24 at 3 03 14 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/490e27a5-6154-4e2d-abdb-dc68162540d5">
<img width="1442" alt="Screenshot 2023-05-24 at 3 03 55 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/c5ae8f44-f973-4f3e-981e-0cc2eba56d7d">
- Converted categorical data to numeric.
<img width="1447" alt="Screenshot 2023-05-24 at 3 04 41 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/cfb256cd-fd31-4f79-b383-648a550e01f9">
- Created a feature array, X and a target array y by using the preprocessed data.
<img width="1138" alt="Screenshot 2023-05-24 at 3 05 15 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/8c76b0ba-14b3-459b-9e9a-ddc110133526">
- Split the preprocessed data into training and testing datasets.
<img width="1100" alt="Screenshot 2023-05-24 at 3 05 42 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/e93797e4-5456-4017-b7fe-8a8327c0c87b">
- Scaled the data using a StandardScaler that has been fitted to the training data. 
<img width="1276" alt="Screenshot 2023-05-24 at 3 06 05 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/cf732903-42c7-4a3e-a58b-41ef16f9d37e">

## Step 2: Compile, Train, and Evaluate the Model
- Created a neural network model with a defined number of input features and nodes for each layer.
<img width="1113" alt="Screenshot 2023-05-24 at 3 11 29 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/dc040aa8-7b26-4eec-86b8-d5f5883397e5"
- Created hidden layers and an output layer with appropriate activation functions.
<img width="1111" alt="Screenshot 2023-05-24 at 3 12 35 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/251e152c-ec50-45df-ae87-335568603fd1">
- Checked the structure of the model.
<img width="1018" alt="Screenshot 2023-05-24 at 3 12 56 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/2bc47e60-1efc-4ede-8086-3c03d3eba754">
- Compiled and trained the model.
- <img width="1172" alt="Screenshot 2023-05-24 at 3 13 11 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/ad5a21c4-c45d-4c24-aaf5-460bdd1873fd">
- Evaluated the model using the test data to determine loss and accuracy. 
<img width="1179" alt="Screenshot 2023-05-24 at 3 13 30 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/5633cf2f-6103-42ca-8403-bff07a26da35">
- Exported the results to an HDF5 file named AlphabetSoupCharity.h5.
<img width="664" alt="Screenshot 2023-05-24 at 3 13 43 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/99af4b01-59f0-4a0e-aaf6-c1a8222c889a">

## Step 3: Optimize the Model
- Repeated the preprocessing steps in a new Jupyter notebook. 
<img width="1177" alt="Screenshot 2023-05-24 at 3 16 07 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/5ec8e56b-5da6-4c92-8acf-ac2154ee0c38">
- Created a new neural network model, implementing at least 3 model optimization methods.
- Optimization Attempt 1: Dropped an additional column (Organization) and created more bins for rare occurrences in columns by changing the threshold for others category <50 for both the application type and classification columns.
<img width="1358" alt="Screenshot 2023-05-24 at 3 18 46 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/17247f15-e193-4008-855c-10541846b7d3">
<img width="1375" alt="Screenshot 2023-05-24 at 3 19 02 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/c447bcfa-a6b5-43cf-8cad-cb73d847a42d">
<img width="1367" alt="Screenshot 2023-05-24 at 3 19 18 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/b343fb72-3b8b-4227-b6a7-cdfc0efa9f08">
- Optimization Attempt 2: Added more neurons to hidden layers (the first and second layer by 100 and 60, respectively) and added an additional hidden nodes layer.
<img width="1353" alt="Screenshot 2023-05-24 at 3 20 39 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/9f9af80c-6c0d-4aa3-9fdf-479d7c9c5d0a">
- Optimization Attempt 3: Used the sigmoid acitvation function for all the layers and increased the number of epochs to 200. 
<img width="1338" alt="Screenshot 2023-05-24 at 3 21 32 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/9c10dc7d-36ad-46a7-bbf0-ecb3bde793f0">
- Saved and exported the results to an HDF5 file named AlphabetSoupCharity_Optimization.h5.
<img width="1190" alt="Screenshot 2023-05-24 at 3 21 54 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/877f8046-d846-463c-97e5-e73dc06209d5">

## Step 4: Addressed questions regarding the purpose of the analysis, the results, and an overall summary of the process (see above). 

# Author 
-Brittany Wright github:brittanynicole7
