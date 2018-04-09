
# coding: utf-8

# In[21]:


from sklearn import datasets
from __future__ import division
iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)


# In[22]:


from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)


# In[23]:


from sklearn.naive_bayes import GaussianNB


# In[24]:


def myAccurary(predicted, target):
    count = 0;
    for i in xrange(len(target)):
        if predicted[i] == target[i]:
            count += 1
    return count / len(target) * 100;


# In[25]:


class HardCodedClassifier():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self;
    
    def predict(self, X_test):
        return [0 for i in xrange(len(X_test))]
        


# In[26]:


#classifier = HardCodedClassifier()
#model = classifier.fit(X_train, y_train)
#targets_predicted = model.predict(X_test)


# In[32]:


def main():
    dataSet = raw_input("Name of your dataset: ")
    testSize = raw_input("Enter in a number between 1 and 100 for your test size: ")
    
    
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=int(testSize) / 100, random_state=0)
    
    classifier = GaussianNB()
    model = classifier.fit(X_train, y_train)
    targets_predicted = model.predict(X_test)
    
    
    print "Your test accuracy is ", "%.2f" % myAccurary(targets_predicted, y_test), "%"
    


# In[33]:


main()

