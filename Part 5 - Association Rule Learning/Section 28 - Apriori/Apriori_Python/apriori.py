# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transcation = []
for i in range(0,7501):
    transcation.append([str(dataset.values[i,j]) for j in range(1, 20)])

from apyori import apriori
rules = apriori(transcation, min_support = 0.003, min_confidence =0.2, min_lift = 3, min_length = 2)

results = list(rules)