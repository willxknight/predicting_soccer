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
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import f1_score

def model_assessment(df):
    
    # separate samples from labels 
    xFeat = df.iloc[:,1:]
    y = df.iloc[:,0]
    
    # split data into random train and test; random_state controls for reproducible output
    X_train, X_test, Y_train, Y_test = train_test_split(xFeat, y, test_size = 0.3, random_state = 42)
    
    return X_train, X_test, Y_train, Y_test

def preprocess_data(trainDF, testDF):
    # preprocess data using standard scaling
    scaler = StandardScaler()
    scaler.fit(trainDF) # computes mean and std deviation for later scaling
    
    scaledTrain = scaler.transform(trainDF) # scale training data
    scaledTest = scaler.transform(testDF) # scale test data
    
    # convert back to pandas dataframe
    trainDF = pd.DataFrame(scaledTrain)
    testDF = pd.DataFrame(scaledTest)
    
    return trainDF, testDF

def main(): 
    performance_data = pd.read_csv('data_n_of_5.csv')
    
    performance_data = performance_data[performance_data.winner != 0.5]
    
    # result = performance_data.corr(method = 'pearson')
    # sns.heatmap(result)
    # plt.show()
    
    X_train, X_test, Y_train, Y_test = model_assessment(performance_data)
    
    # preprocess training data
    X_train_prep, X_test_prep = preprocess_data(X_train, X_test)
    
    # -------------- GET OPTIMAL PARAMETERS -------------- 
    # KNN
    knn_params = [{'n_neighbors': range(1, 30)}]
    knn_grid = GridSearchCV(KNeighborsClassifier(), 
                            param_grid = knn_params,
                            cv = 5)                         
    knn_grid.fit(np.array(X_train_prep), np.array(Y_train).ravel())
    best_knn_score = knn_grid.best_score_
    best_knn_p = knn_grid.best_params_

    print()
    print("---- Optimal hyperparameters for K-NN ----")
    print(best_knn_score, best_knn_p )
    print()

    # DECISION TREE 
    tree_params = [{'criterion': ['gini', 'entropy'],
                    'max_depth': range(1,31,2),
                    'min_samples_split': [2,3,4,5,6,7,8,9,10,11]
                    }]

    dt_grid = GridSearchCV(DecisionTreeClassifier(), 
                            param_grid = tree_params,
                            cv = 5) # again, uses 5-fold cross validation

    dt_grid.fit(np.array(X_train_prep), np.array(Y_train).ravel())

    print()
    print("---- Optimal hyperparameters for Decision Tree ----")
    print(dt_grid.best_score_, dt_grid.best_params_)
    print()

    # -------------- RUN MODELS -------------- 

    test3_result = pd.DataFrame(columns = ['method','accuracy', 'f1', 'AUC'])

    
    # use knn classifier, where n_neighbors equal to those that gave the optimal
    # score in part a 
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X_train, Y_train, test_size = 0.3, random_state = 42)
    X_train_prep2, X_test_prep2 = preprocess_data(X_train2, X_test2)
    
    
    knn3 = KNeighborsClassifier(n_neighbors = 29) 
    
    knn3.fit(X_train_prep2, Y_train2.values.ravel())

    yHat = knn3.predict(X_test_prep2) # class predictions
    
    accuracy = metrics.accuracy_score(Y_test2, yHat) # accuracy

    fpr, tpr, thresholds = metrics.roc_curve(Y_test2,
                                              yHat)
    auc = metrics.auc(fpr, tpr) # auc
    
    # f1 score
    f1 = f1_score(Y_test2, yHat, average='micro')
    
    test3_result = test3_result.append({'method': 'KNN',
                              'accuracy': accuracy,
                              'f1': f1,
                              'AUC': auc}, ignore_index = True)
    
    # repeat for the decision tree classifier, where parameters equal those that gave
    # the optimal score in part a
    dt3 = DecisionTreeClassifier(criterion = 'gini', 
                                  max_depth = 5, 
                                  min_samples_split = 7) 
    dt3.fit(X_train_prep2, Y_train2.values.ravel())
    yHat2 = dt3.predict(X_test_prep2) # class predictions
    accuracy2 = metrics.accuracy_score(Y_test2, yHat2) # accuracy
    fpr2, tpr2, thresholds = metrics.roc_curve(Y_test2,
                                              yHat2)
    auc2 = metrics.auc(fpr2, tpr2) # auc
    
    # f1 score
    f12 = f1_score(Y_test2, yHat2, average='micro')
    
    test3_result = test3_result.append({'method': 'Decision Tree', 
                              'accuracy': accuracy2,
                              'f1': f12,
                              'AUC': auc2}, ignore_index = True)
    
    print(test3_result)
    

    # plot ROC curves
    plt.plot(fpr2,tpr2)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.show()
    
    

if __name__ == "__main__":
    main()
