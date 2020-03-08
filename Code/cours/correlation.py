# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:45:36 2020

@author: Moi
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot

k = pd.DataFrame()
k['X'] = np.arange(5)+3
k['Y'] = [1, 3, 4, 8, 12]
pyplot.scatter(k['X'], k['Y'], s = 150, c = 'red', marker = '*', edgecolors = 'blue')


k.corr(method='pearson')


k.corr(method='spearman')
k.corr(method='kendall')



titanic = pd.read_csv("../datasources/titanic/train.csv")
data = titanic.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
data.corr(method='spearman')




data.corr(method='spearman').style.format("{:.2}").background_gradient(cmap=pyplot.get_cmap('coolwarm'))
# Cf. https://matplotlib.org/examples/color/colormaps_reference.html pour les codes couleurs




