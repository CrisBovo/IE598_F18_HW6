# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:36:28 2018
IE-598 Machine Learning Assignment_6
Module 6 (Cross validation)
@author: Haichao Bo (hbo2)
"""

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target



scores_in_sample = []
scores_out_of_sample = []
for i in range(1,11):
    
    # Splitting data into 90% training and 10% test data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i, stratify=y)
    
    # Instantiate a DecisionTreeRegressor dt
    dt = DecisionTreeClassifier(max_depth=4, random_state=1)

    # Fit it to the data
    dt.fit(X_train, y_train)
    scores_in_sample.append(dt.score(X_train, y_train))
    scores_out_of_sample.append(dt.score(X_test, y_test))

    print('Train Accuracy: %.3f' % dt.score(X_train, y_train))
    print('Test Accuracy: %.3f' % dt.score(X_test, y_test))

print('In_sample accuracy: mean = %.3f' % np.mean(scores_in_sample))
print('In_sample accuracy: std = %.3f' % np.std(scores_in_sample))
print('Out_of_sample accuracy: mean = %.3f' % np.mean(scores_out_of_sample))
print('Out_of_sample accuracy: std = %.3f' % np.std(scores_out_of_sample))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10, stratify=y)
scores = cross_val_score(estimator=dt, X=X_train, y=y_train, cv=10, n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: mean = %.3f, std = %.3f' % (np.mean(scores),np.std(scores)))
scores = cross_val_score(estimator=dt, X=X, y=y, cv=10, n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: mean = %.3f, std = %.3f' % (np.mean(scores),np.std(scores)))

print("My name is Haichao Bo")
print("My NetID is: hbo2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
