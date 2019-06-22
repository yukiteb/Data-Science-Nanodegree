# DSND-Capstone-Sparkify
This project is completed as the capstone project for Udacity Data Science Nanodegree Program. In this project, we will analyze the (realistic) data of (fictious) online music app called Sparklify and predict the churn using Spark.

## Installation
It requires the following packages.

- Pandas
- Spark
- PySpark
- Matplotlib

## Steps

### 1. Clean the data
There are about 550,000 rows in the data set, which corresponds to roughly about 450 unique users. Missing values are handled.

### 2. Feature Engineerig
Following features are used:

1. Gender: General demographic information about the user
2. Number of page not found (how many times successful or unsuccessful): This may indicate service level of the app
3. Number of thumbs up / down: This feature tells whether users like / dislike music content
4. Number of Add to Playlist: This feature also tells whether users like / dislike music content
5. Total number of page visit (exclude Cancel Confirmation): This features tells how users are engaged in the service
6. Average Hour of page visit in a day: This feature captures some behavioral pattern of users

### 3. Model
Following models are tried

- Logistic Regression
- Random Forest
- Gradient Boosting Trees

The models are optimized for their hyperparameters and we use F1 score as an evaluation metric.
The model with the highest F1 score on the 3-fold crossvalidation is chosen as the "best" model.

## Results
The Random Forest have the best cross validation F1 score of 0.718 and the test F1 score of 0.722

The summary of this analysis is posted on medium (https://medium.com/@yukiteb2/churn-prediction-using-pyspark-1948cfa2d00)

