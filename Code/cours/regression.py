# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:21:33 2020

@author: Moi
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
data = pd.read_csv("./data/univariate_linear_regression_dataset.csv")
plt.scatter (data.col2, data.col1)
X = data.col2.values.reshape(-1, 1)
y = data.col1.values.reshape(-1, 1)
regr = linear_model.LinearRegression()
regr.fit(X, y)


regr.predict([[30]])


from joblib import dump, load
dump(regr, 'monpremiermodele.modele') 




regr2 = load('monpremiermodele.modele')
regr2.predict([[30]])


