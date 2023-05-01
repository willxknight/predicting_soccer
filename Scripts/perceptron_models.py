# Mute Warnings, Help From: https://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score


def model_assessment(df):
    xFeat = df.iloc[:,1:]
    y = df.iloc[:,0]
    X_train, X_test, Y_train, Y_test = train_test_split(xFeat, y, test_size = 0.3, random_state = 42)
    return X_train, X_test, Y_train, Y_test

def preprocess_data(trainDF, testDF):
    scaler = StandardScaler()
    scaler.fit(trainDF)
    scaledTrain = scaler.transform(trainDF)
    scaledTest = scaler.transform(testDF)
    trainDF = pd.DataFrame(scaledTrain)
    testDF = pd.DataFrame(scaledTest)
    return trainDF, testDF

def main():
    n_1_data = "data_n_of_1.CSV"
    data_1 = pd.read_csv(n_1_data)
    # Removing Ties: https://numpy.org/doc/stable/reference/generated/numpy.where.html
    data_1 = data_1[data_1.winner != 0.5]
    xTrain, xTest, yTrain, yTest = model_assessment(data_1)
    # Help From: https://towardsdatascience.com/predictive-modeling-picking-the-best-model-69ad407e1ee7
    percept1 = Perceptron(tol=1e-3, random_state=0)
    percept1.fit(xTrain, yTrain)
    y_hat1 = percept1.predict(xTrain)
    print('Train accuracy score on n = 1:', accuracy_score(yTrain, y_hat1))
    print('Test accuracy score on n = 1:', accuracy_score(yTest, percept1.predict(xTest)))
    f1 = f1_score(yTest, percept1.predict(xTest), average='micro')
    print('F1-score on n = 1', f1)
    print()

    n_2_data = "data_n_of_2.CSV"
    data_2 = pd.read_csv(n_2_data)
    # Removing Ties: https://numpy.org/doc/stable/reference/generated/numpy.where.html
    data_2 = data_2[data_2.winner != 0.5]
    xTrain, xTest, yTrain, yTest = model_assessment(data_2)
    # Help From: https://towardsdatascience.com/predictive-modeling-picking-the-best-model-69ad407e1ee7
    percept2 = Perceptron(tol=1e-3, random_state=0)
    percept2.fit(xTrain, yTrain)
    y_hat2 = percept2.predict(xTrain)
    print('Train accuracy score on n = 2:', accuracy_score(yTrain, y_hat2))
    print('Test accuracy score on n = 2:', accuracy_score(yTest, percept2.predict(xTest)))
    f2 = f1_score(yTest, percept2.predict(xTest), average='micro')
    print('F1-score on n = 2', f2)
    print()

    n_3_data = "data_n_of_3.CSV"
    data_3 = pd.read_csv(n_3_data)
    # Removing Ties: https://numpy.org/doc/stable/reference/generated/numpy.where.html
    data_3 = data_3[data_3.winner != 0.5]
    xTrain, xTest, yTrain, yTest = model_assessment(data_3)
    # Help From: https://towardsdatascience.com/predictive-modeling-picking-the-best-model-69ad407e1ee7
    percept = Perceptron(tol=1e-3, random_state=0)
    percept.fit(xTrain, yTrain)
    y_hat = percept.predict(xTrain)
    print('Train accuracy score on n = 3:', accuracy_score(yTrain, y_hat))
    print('Test accuracy score on n = 3:', accuracy_score(yTest, percept.predict(xTest)))
    f3 = f1_score(yTest, percept.predict(xTest), average='micro')
    print('F1-score on n = 3', f3)
    print()

    n_4_data = "data_n_of_4.CSV"
    data_4 = pd.read_csv(n_4_data)
    # Removing Ties: https://numpy.org/doc/stable/reference/generated/numpy.where.html
    data_4 = data_4[data_4.winner != 0.5]
    xTrain, xTest, yTrain, yTest = model_assessment(data_4)
    # Help From: https://towardsdatascience.com/predictive-modeling-picking-the-best-model-69ad407e1ee7
    percept4 = Perceptron(tol=1e-3, random_state=0)
    percept4.fit(xTrain, yTrain)
    y_hat4 = percept4.predict(xTrain)
    print('Train accuracy score on n = 4:', accuracy_score(yTrain, y_hat4))
    print('Test accuracy score on n = 4:', accuracy_score(yTest, percept4.predict(xTest)))
    f4 = f1_score(yTest, percept4.predict(xTest), average='micro')
    print('F1-score on n = 4', f4)
    print()

    n_5_data = "data_n_of_5.CSV"
    data_5 = pd.read_csv(n_5_data)
    # Removing Ties: https://numpy.org/doc/stable/reference/generated/numpy.where.html
    data_5 = data_5[data_5.winner != 0.5]
    xTrain, xTest, yTrain, yTest = model_assessment(data_5)
    # Help From: https://towardsdatascience.com/predictive-modeling-picking-the-best-model-69ad407e1ee7
    percept5 = Perceptron(tol=1e-3, random_state=0)
    percept5.fit(xTrain, yTrain)
    y_hat5 = percept5.predict(xTrain)
    print('Train accuracy score on n = 5:', accuracy_score(yTrain, y_hat5))
    print('Test accuracy score on n = 5:', accuracy_score(yTest, percept5.predict(xTest)))
    f5 = f1_score(yTest, percept5.predict(xTest), average='micro')
    print('F1-score on n = 5', f5)
    print()

    n_6_data = "data_n_of_6.CSV"
    data_6 = pd.read_csv(n_6_data)
    # Removing Ties: https://numpy.org/doc/stable/reference/generated/numpy.where.html
    data_6 = data_6[data_6.winner != 0.5]
    xTrain, xTest, yTrain, yTest = model_assessment(data_6)
    # Help From: https://towardsdatascience.com/predictive-modeling-picking-the-best-model-69ad407e1ee7
    percept6 = Perceptron(tol=1e-3, random_state=0)
    percept6.fit(xTrain, yTrain)
    y_hat6 = percept6.predict(xTrain)
    print('Train accuracy score on n = 6:', accuracy_score(yTrain, y_hat6))
    print('Test accuracy score on n = 6:', accuracy_score(yTest, percept6.predict(xTest)))
    f6 = f1_score(yTest, percept6.predict(xTest), average='micro')
    print('F1-score on n = 6', f6)
    print()

    n_7_data = "data_n_of_7.CSV"
    data_7 = pd.read_csv(n_7_data)
    # Removing Ties: https://numpy.org/doc/stable/reference/generated/numpy.where.html
    data_7 = data_7[data_7.winner != 0.5]
    xTrain, xTest, yTrain, yTest = model_assessment(data_7)
    # Help From: https://towardsdatascience.com/predictive-modeling-picking-the-best-model-69ad407e1ee7
    percept7 = Perceptron(tol=1e-3, random_state=0)
    percept7.fit(xTrain, yTrain)
    y_hat7 = percept7.predict(xTrain)
    print('Train accuracy score on n = 7:', accuracy_score(yTrain, y_hat7))
    print('Test accuracy score on n = 7:', accuracy_score(yTest, percept7.predict(xTest)))
    f7 = f1_score(yTest, percept7.predict(xTest), average='micro')
    print('F1-score on n = 7', f7)
    print()

if __name__ == "__main__":
    main()
