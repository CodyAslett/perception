import pandas as pd
import numpy as np

# rank_mpg
#
#This function goes through and cleans up the mpg dataset
def rank_mpg():
    #column names
    headers = [
        'mpg',
        'cylinders',
        'displacement',
        'horsepower',
        'weight',
        'acceleration',
        'year',
        'origin',
        'name'
    ];

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data',
            header=None, names=headers, delim_whitespace=True)

    # the last column, names, is not required because it doesn't affect the mpgs. it's more of a convenience.
    df_names = df.pop('name')

    #replace all ? with NaN and deletes the rows
    df.replace('?', np.NaN, inplace=True)
    df.dropna(inplace=True)

    #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.as_matrix.html
    #convert the dataframe into numpy arrays.
    #the first return is for data while the second return is the targets.
    #first column is actually the target
    return df.as_matrix(columns=headers[1:8]).astype(float), df.as_matrix(columns=headers[0:1]).astype(float)
