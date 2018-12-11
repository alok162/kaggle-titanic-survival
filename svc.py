#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:23:22 2018

@author: alok
"""

# import libraries 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# read dataset csv file
dataset = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# lets preprocess the data
column_target = ['Survived']
column_train = ['Pclass', 'Sex', 'Age', 'Fare']
X = dataset[column_train]
Y = dataset[column_target]


X_test = test[column_train]

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
X['Sex'] = labelencoder_X_1.fit_transform(X['Sex'])
labelencoder_X_2 = LabelEncoder()
X_test['Sex'] = labelencoder_X_2.fit_transform(X_test['Sex'])

# fill nan value with median
X['Age'] = X['Age'].fillna(X['Age'].median())
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].median())
X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].median())

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, Y)
y_pred = classifier.predict(X_test)

submission = pd.DataFrame({
        "PassengerId" : test["PassengerId"],
        "Survived" : y_pred
        })

submission.to_csv('titanic.csv', index=False)









