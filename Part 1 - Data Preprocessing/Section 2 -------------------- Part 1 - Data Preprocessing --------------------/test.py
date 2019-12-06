# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
dataset = pd.read_csv("Data.csv")

"""
iloc is for setting the certain columns of the dataset into the array or some other file for re-arranging the values
"""
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


"""
we use Imputer because if there are some missing values in the data set we can arrange them by keeping mean, median values in the missing column
"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


"""
Changing text to the integers we use LabelEncoder 
we use OneHotEncoder because if there are multiple values changing to numbers we categarious the values and divide it.

"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x = LabelEncoder()
X[:, 0] = le_x.fit_transform(X[:, 0])
o = OneHotEncoder(categorical_features = [0])
X = o.fit_transform(X).toarray()
le_y = LabelEncoder()
y = le_y.fit_transform(y)