# Practical_3

### Introduction

In this weeks lecture, we looked at the machine learning (ML) training and tuning aspects of the ML pipeline. We looked at the ease of which a model from scikit-learn can be trained and tasked with a prediction. Recall that the code can entail 4 steps (import library --> initialise model --> fit the model --> prediction/test. We then looked at some of the metrics used to evaluate a model's performance. To put this to practice, and to combine your knowledge from the previous Practicals, in this week's tutorial you will build a supervised model to predict whether a patient is diabetic or not using recorded medical data. The dataset can be found within this repository, labelled <ins>'Dataset for People for their Blood Glucose Level with their Superficial body feature readings.'</ins>. Once you have imported the data into your python, please proceed with building a model that predicts whether a patient is diabetic or not. You can either do it independently or use the code below as a template.

Once you have finished, upload your report to your own github repo. Please 'Example_report' for guidance on how to structure the report.


``` python
# import python and numpy
import pandas as pd
import numpy as np
import sys

#Check current python version
print(sys.version)

#read the data file
df = pd.read_excel()

#Check for missing data
df.isna().sum().sum()

#Check the Column names
df.columns

#Check feature statistics
df.describe()

# Exploratory Data Analysis
df['Age'].plot.box()

df[df.columns[-1]].value_counts(normalize = True).plot.pie()

# Model Training
from sklearn.ensemble import RandomForestClassifier # Import Random Forest
from sklearn.neural_network import MLPClassifier # Import MLP (Multi-Layer Perceptron)
from sklearn.linear_model import LogisticRegression # Import Logistic Regression
from sklearn.tree import DecisionTreeClassifier # Import Decision Trees
from sklearn.model_selection import train_test_split # Code for splitting the data into training and testing
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score #Code for Classification metrics

#Initialise the models
rf_clf = RandomForestClassifier()
ann_clf = MLPClassifier()
lr_clf = LogisticRegression()
dt_clf = DecisionTreeClassifier()

# Assign variables to your inputs and output
X = df.drop([df.columns[-1]], axis = 1)
y = df[df.columns[-1]]

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model fitting & prediction
rf_pred = rf_clf.fit(X_train, y_train).predict(X_test)
ann_pred = ann_clf.fit(X_train, y_train).predict(X_test)
lr_pred = lr_clf.fit(X_train, y_train).predict(X_test)

#check the accuracy score
accuracy_score(y_test, rf_pred), accuracy_score(y_test, ann_pred), accuracy_score(y_test, lr_pred), accuracy_score(y_test, dt_pred)
dt_pred = dt_clf.fit(X_train, y_train).predict(X_test)

#check the MCC
matthews_corrcoef(y_test, rf_pred), matthews_corrcoef(y_test, ann_pred), matthews_corrcoef(y_test, lr_pred), matthews_corrcoef(y_test, dt_pred)

#check the F1 score
f1_score(y_test, rf_pred), f1_score(y_test, ann_pred), f1_score(y_test, lr_pred), f1_score(y_test, dt_pred) ```
