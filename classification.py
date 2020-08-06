import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


houses = pd.read_csv('data/housing.csv')

#clean data - see regression.py for more description
houses = houses.dropna()
houses = houses.drop(houses.loc[houses['median_house_value'] == 500001].index)
houses = pd.get_dummies(houses, columns=['ocean_proximity'])

#add column to dataframe that is whether or not it is above median
median = houses['median_house_value'].median()
houses['above_median'] = (houses['median_house_value'] - median) > 0

#set up test and training data
X = houses.drop(['median_house_value', 'above_median'], axis=1)
Y = houses['above_median']
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2)

#set up model, solver liblinear good for small datasets and binary classification
logisticModel = LogisticRegression(solver='liblinear').fit(xTrain, yTrain)
#evaluate how well it works on training data
evaluationTrain = logisticModel.score(xTrain, yTrain)

yPred = logisticModel.predict(xTest)
dfPredActual = pd.DataFrame({'predicted': yPred, 'actual': yTest})

#evaluate model on test data
evaluationTest = accuracy_score(yTest, yPred)
