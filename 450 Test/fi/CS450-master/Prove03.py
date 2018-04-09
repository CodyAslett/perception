"""
Author: Cutler Hollist
Purpose: This program adds to the "experiment shell" where the author
    will load training and test data, implement a hard-coded classifier
    as a placeholder, and report results with that classifier.
    This assignment is primarily meant to act as a framework for
    future assignments.
Attribution: The author used several sources to produce the code below:
    experiment_shell.py - by Brother Burton, for the replacement of much
        of the author's original main function. Burton's is much simpler,
        though the code is nearly the same. The real difference is in
        Burton's separating the code into multiple functions.
    http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
        - by Sebastian Raschka, primarily for the standardizing and
        normalizing features, though also for a baseline to read in
        datasets
    https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/ - by Kevin
        Zakka, for much of the implementation of the KNN algorithm and
        converting Pandas dataframes into NumPy arrays.
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#from slkearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import datasets as ds
#from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

class HardCodedClassifier:
    def fit(self, data, targets):
        return HardCodedModel()


class HardCodedModel:    
    def predict(self, test_data):
        predictions = []
        for i in range(len(test_data)):
            predictions.append(0)
        return predictions


class HollistKnnClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, data_train, targets_train):
        return HollistKnnModel(self.k, data_train, targets_train)


class HollistKnnModel:
    def __init__(self, k, data_train, targets_train):
        self.k = k
        self.data_train = data_train
        self.targets_train = targets_train

    def predict(self, data_test):
        predictions = []

        for i in range(len(data_test)):
            distances = []
            targets = []
            for j in range(len(self.data_train)):
                d = np.sqrt(np.sum(np.square(data_test[i, :] - self.data_train[j, :])))
                distances.append([d, i])

            distances = sorted(distances)
            for j in range(self.k):
                k = distances[i][1]
                targets.append(self.targets_train[k])

            predictions.append(Counter(targets).most_common(1)[0][0])

        return predictions


def run_test(data, targets, algorithm, test_size=0.3):
    """
    A function to build, train and test an algorithm model on the data
    set passed to it.
    :return: model
    """
    print("Running Experiment...")
    print("Dataset shape: {}".format(data.shape))

    # Randomizes the order, then breaks the data into training and testing sets
    data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=test_size)

    # Build a model using the provided algorithm
    model = algorithm.fit(data_train, targets_train)

    # Use the model to make a prediction
    targets_predicted = model.predict(data_test)

    # Compute the amount we got correct
    correct = (targets_test == targets_predicted).sum()
    total = len(targets_test)
    percent = correct / total * 100

    # Display result
    print("Correct: {}/{} or {:.2f}%".format(correct, total, percent))

    # Use k-fold cross validation to compare your results
    n = 3
    kfold = KFold(n_splits=n, random_state=7)
    
    #for k in xrange(n):
    #    training = [x for i, x in enumerate(X) if i % K != k]
    #    validation = [x for i, x in enumerate(X) if i % K == k]
    #    yield training, validation
    
    result = cross_val_score(algorithm, data_test, targets_test, cv=kfold, scoring='accuracy')
    print("{}-Fold Cross-Val Score: {}".format(str(n), result))

    # Send the model back
    return model


def get_algorithm():
    """
    A factory to create the algorithm we want.
    :return:
    """
    #algorithm = GaussianNB()
    #algorithm = HardCodedClassifier()
    algorithm = KNeighborsClassifier(n_neighbors=3)
    #algorithm = HollistKnnClassifier(3)

    return algorithm


def handle_non_num_cars1(data_array):
    # [0]: 0, 1, 2, 3     (low, med, high, vhigh)
    data_array["buying"] = data_array["buying"].astype('category')
    data_array["buying"] = data_array["buying"].cat.codes
    # [1]: 0, 1, 2, 3     (low, med, high, vhigh)
    data_array["maint"] = data_array["maint"].astype('category')
    data_array["maint"] = data_array["maint"].cat.codes
    # [2]: 0, 1, 2, 3     (2, 3, 4, 5more)
    data_array["doors"] = data_array["doors"].astype('category')
    data_array["doors"] = data_array["doors"].cat.codes
    # [3]: 0, 1, 2        (2, 4, more)
    data_array["persons"] = data_array["persons"].astype('category')
    data_array["persons"] = data_array["persons"].cat.codes
    # [4]: 0, 1, 2        (small, med, big)
    data_array["lug_boot"] = data_array["lug_boot"].astype('category')
    data_array["lug_boot"] = data_array["lug_boot"].cat.codes
    # [5]: 0, 1, 2        (low, med, high)
    data_array["safety"] = data_array["safety"].astype('category')
    data_array["safety"] = data_array["safety"].cat.codes
    
    return data_array


def handle_non_num_cars2(data_array):    
    cleanup_cols = {"buying"  : {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                    "maint"   : {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                    "doors"   : {"5more": 5, "4"   : 4, "3"  : 3, "2": 2},
                    "persons" : {"more": 5, "4": 4, "2": 2},
                    "lug_boot": {"big": 3, "med": 2, "small": 1},
                    "safety"  : {"high": 3, "med": 2, "low": 1}
                   }
    data_array.replace(cleanup_cols, inplace=True)
    return data_array


def load_cars():
    # Read in data set
    file_name = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
    df = pd.io.parsers.read_csv(file_name, header=None)
    df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df.head()

    # Split into data and targets
    data_array = df.ix[:, 0:6]	    # end index is exclusive
    targets_array = df['class'] 	# another way of indexing a pandas df

    # Handle non-numeric data
    #data_array = handle_non_num_cars1(data_array)
    data_array = handle_non_num_cars2(data_array)
    
    # Handle missing data
    # Unnecessary for this dataset

    # Standardize data
    std_scale = preprocessing.StandardScaler().fit(data_array)
    data_std = std_scale.transform(data_array)

    # OR Normalize (min-max) data
    #minmax_scale = preprocessing.MinMaxScaler().fit(df)
    #df_minmax = minmax_scale.transform(df)

    # convert from pandas dataframe to NumPy arrays
    data = np.array(data_std)
    targets = np.array(targets_array)

    return data, targets


def load_diabetes():
    # Read in data set
    file_name = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
    df = pd.io.parsers.read_csv(file_name, header=None)
    print(df.columns)
    df.columns = ['num_preg', 'gluc_conc', 'blood_press', 'skin_thick', 'insulin', 'bmi', 'pedigree', 'age', 'class']
    print(df.columns)
    df.head()

    # Split into data and targets
    data_array = df.ix[:, 0:8]	    # end index is exclusive
    targets_array = df['class'] 	# another way of indexing a pandas df

    # Handle non-numeric data
    # Unnecessary for this dataset

    # Handle missing data
    data_array[['gluc_conc', 'blood_press', 'skin_thick', 'insulin', 'bmi']] = data_array[['gluc_conc', 'blood_press', 'skin_thick', 'insulin', 'bmi']].replace(0, np.NaN)
    data_array.fillna(data_array.mean(), inplace=True)

    # Standardize data
    std_scale = preprocessing.StandardScaler().fit(data_array)
    data_std = std_scale.transform(data_array)

    # OR Normalize (min-max) data
    #minmax_scale = preprocessing.MinMaxScaler().fit(data_array)
    #df_minmax = minmax_scale.transform(data_array)

    # convert from pandas dataframe to NumPy arrays
    data = np.array(data_std)
    #data = np.array(df_minmax)
    targets = np.array(targets_array)

    return data, targets


def load_mpg():
    # Read in data set
    file_name = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    df = pd.io.parsers.read_csv(file_name, delim_whitespace=True, header=None)
    df.columns = ['class', 'cylinders', 'disp', 'hp', 'weight', 'acc', 'year', 'origin', 'name']
    df.head()    # cont     m-v dis      cont    cont  cont      cont   m-v dis m-v dis   string

    # Split into data and targets
    data_array = df.ix[:, 1:8]	    # end index is exclusive, ignore the column of names
    targets_array = df['class'] 	# another way of indexing a pandas df

    # Handle non-numeric data 
    ## This should be okay, I like leaving the discrete multi-values as they are
    # When commented out and data_array only goes from column 1 to column 8, ignore data in "name" column
    #data_array["name"] = data_array["name"].astype('category')
    #data_array["name"] = data_array["name"].cat.codes

    # Handle missing data ## This should be all that's necessary
    data_array[['hp']] = data_array[['hp']].replace('?', np.NaN)
    data_array['hp'].fillna(0, inplace=True)
    print(data_array)

    # Standardize data
    std_scale = preprocessing.StandardScaler().fit(data_array)
    data_std = std_scale.transform(data_array)

    # OR Normalize (min-max) data
    #minmax_scale = preprocessing.MinMaxScaler().fit(data_array)
    #df_minmax = minmax_scale.transform(data_array)

    # convert from pandas dataframe to NumPy arrays
    data = np.array(data_std)
    #data = np.array(df_minmax)
    targets = np.array(targets_array)

    return data, targets


def get_dataset():
    """
    A factory to load the dataset we want.
    :return:
    """
    #data, targets = ds.load_iris().data, ds.load_iris().target
    #data, targets = load_cars()
    #data, targets = load_diabetes()
    data, targets = load_mpg()

    return data, targets


def main():
    """
    Gets and algorithm, dataset, and then passes them to run_tests
    :return:
    """
    algorithm = get_algorithm()
    data, targets = get_dataset()

    model = run_test(data, targets, algorithm)


if __name__ == "__main__":
    main()
