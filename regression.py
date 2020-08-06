import sklearn
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


houses = pd.read_csv('data/housing.csv')
#drops all records with missing fields
houses = houses.dropna()
#provide some information about entire data
insight = houses.describe()
#identify unique values (ex: types of ocean proximity that are present)
proximities = houses['ocean_proximity'].unique()

#visualize some data
fix, ax = plt.subplots(figsize=(12,8))
plt.scatter(houses['total_rooms'], houses['median_house_value'])
plt.xlabel('Number of rooms')
plt.ylabel('Median house value')
#plt.show()

#show correlation between variables
correlation = houses.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation, annot=True)
#plt.show()

#clean data by removing values all at 500001
num500k = houses.loc[houses['median_house_value'] == 500001].count()
houses = houses.drop(houses.loc[houses['median_house_value'] == 500001].index)

#change string value categories to numeric
#one-hot encoding -> change categories to column headers
houses = pd.get_dummies(houses, columns=['ocean_proximity'])

#Transition data to independent (all columns aside from value) and dependent variables (value)
X = houses.drop('median_house_value', axis=1)
Y = houses['median_house_value']

#split data into training and testing using scikit built in function
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2)

#initialize model - normalize can improve efficiency and effectiveness
linearModel = LinearRegression(normalize=True).fit(xTrain,yTrain)
#evaluate model on training data
rSquaredTrain = linearModel.score(xTrain, yTrain)

predictors = xTrain.columns
#coefficients show the weight of that value on the predictions/model
coef = pd.Series(linearModel.coef_, predictors).sort_values()

#predict the results of test data and compare it to the actual test values
yPred = linearModel.predict(xTest)
dfPredActual = pd.DataFrame({'predicted': yPred, 'actual': yTest}) #visualize in pandas
rSquaredTest = r2_score(yTest, yPred) #evaluate model on test data -> good model if similar to value from training data

#view scatter plot to visualize model effectiveness, can also plot predicted and actual using plt.plot
fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(yTest, yPred)
plt.show()
