
# coding: utf-8



import numpy as np
from sklearn.metrics import accuracy_score
class kNNClassifier():
    """
    This is my own implementation of knn.
    Use it as your own risk
    
    """
    def __init__(self,k):
        self.k = k
        
    def fit(self, train_data, train_target):
        self.train_data = train_data
        self.train_target = train_target
        return self;
    
    #Return the most common item in the list
    def most_common(self, some_list):
        return max(set(some_list), key=some_list.count)
    
    def knn(self, test_data):
        predict = []
        for i in test_data:
            distances = []
            for j in self.train_data:
                #calculate the distance between 2 points
                distances.append(np.linalg.norm(i-j))
            #get and slide the indices
            indices = sorted(range(len(distances)), key=lambda k: distances[k])[:self.k]
    
            predict.append(self.most_common(list(self.train_target[indices])))
        return predict
    
    def predict(self, test_data):
        return self.knn(test_data)

    def get_params(self, deep = False):
        return {'k': self.k}


