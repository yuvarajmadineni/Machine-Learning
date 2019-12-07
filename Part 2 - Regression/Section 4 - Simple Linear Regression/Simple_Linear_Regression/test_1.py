# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
dataset = pd.read_csv("Salary_Data.csv")

"""
iloc is for setting the certain columns of the dataset into the array or some other file for re-arranging the values
"""
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

"""
feature scaling is to tranform the different range of values to the minimal range so that the calucations may be easy to analyze


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

"""
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)