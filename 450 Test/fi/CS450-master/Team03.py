# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:22:09 2018

@author: Chad
"""

import pandas as pd

#pd.read_csv('pandas_dataframe_importing_csv/example.csv')
filename = 'C:/Users/Chad/Desktop/450/adult.data'
data = pd.read_csv(filename)
df = pd.DataFrame(data = data)
df.columns = ['age', 'workclass','fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'GDP']
print(df)
print(list(df.columns.values))
