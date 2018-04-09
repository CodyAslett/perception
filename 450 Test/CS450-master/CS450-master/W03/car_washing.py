import pandas as pd
import numpy as np

# car_washing_service
#
# This function cleans up the car dataset
def car_washing_service():
    #columns
    headers = [
        'buying',
        'maint',
        'doors',
        'persons',
        'lug_boot',
        'safety',
        'target'
    ];

    #building an object that will replace non-numericals with numerical values
    replaceObject = {
        'buying': {
            'vhigh': 0,
            'high': 1,
            'med': 2,
            'low': 3
        },
        'maint': {
            'vhigh': 0,
            'high': 1,
            'med': 2,
            'low': 3
        },
        'doors': {
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            '5more': 5
        },
        'persons': {
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            'more': 5
        },
        'lug_boot': {
            'small': 0,
            'med': 1,
            'big': 2
        },
        'safety': {
            'low': 0,
            'med': 1,
            'high': 2
        },
        'target': {
            'unacc': 0,
            'acc': 1,
            'good': 2,
            'vgood': 3
        }
    }

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
            header=None, names=headers)

    #replace non-numerical with numericals
    df.replace(replaceObject, inplace=True)

    #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.as_matrix.html
    #convert the dataframe into numpy arrays.
    #the first return is for data while the second return is the targets.
    return df.as_matrix(headers[0:6]), df.as_matrix(headers[6:7])
