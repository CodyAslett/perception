
# coding: utf-8

# In[43]:


from sklearn import datasets
from __future__ import division
import pandas as pd
import numpy as np
import urllib2
from knn import kNNClassifier



iris = datasets.load_iris()



#This is how your would load a cvs file

#url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/d546eaee765268bf2f487608c537c05e22e4b221/iris.csv"
#dataset = pd.read_csv(url)

#get the shape of the dataset
#dataset.shape


#dataset.head(5)
#dataset.groupby('species').size()


#feature_columns = ['sepal_length', 'sepal_width', 'petal_length','petal_width']
#X = dataset[feature_columns].values
#y = dataset['species'].values

#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#y = le.fit_transform(y)

#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Show the data (the attributes of each instance)
#print(iris.data)

# Show the target values (in numeric format) of each instance
#print(iris.target)

# Show the actual target names that correspond to each number
#print(iris.target_names)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)


def myAccurary(predicted, target):
    count = 0;
    for i in xrange(len(target)):
        if predicted[i] == target[i]:
            count += 1
    return count / len(target) * 100;




def main():
    
    print "*****************************"
    print "*     k      *    Accuracy  *"
    print "*****************************"

    for i in xrange(1,10):
        classifier = kNNClassifier(i)
        model = classifier.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = str(round(myAccurary(predictions, y_test), 2))
        print "*    ",   i, "     *   ", accuracy, "    *"
    print "*****************************"
    print " "
    print "Please enter in the following attributes of your iris to make a prediction"
    att_1 = raw_input(iris.feature_names[0])
    att_2 = raw_input(iris.feature_names[1])
    att_3 = raw_input(iris.feature_names[2])
    att_4 = raw_input(iris.feature_names[3])
    x = np.array([[att_1, att_2, att_3, att_4]], dtype=np.float64)
    _predictions = model.predict(x)
    print " "
    print "Based on your input, our prediction is", iris.target_names[_predictions[0]]




main()

