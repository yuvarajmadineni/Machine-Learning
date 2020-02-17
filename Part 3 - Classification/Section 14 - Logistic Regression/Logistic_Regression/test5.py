# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
dataset = pd.read_csv("Social_Network_Ads.csv")

"""
iloc is for setting the certain columns of the dataset into the array or some other file for re-arranging the values
"""
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""
feature scaling is to tranform the different range of values to the minimal range so that the calucations may be easy to analyze

"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)