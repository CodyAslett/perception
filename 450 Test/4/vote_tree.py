
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from __future__ import division
head = ['class', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze','el-salvador-aid','religious-groups-in-schools','anti-satellite-test-ban','aid-to-nicaraguan-contras','mx-missile','immigration','synfuels-corporation-cutback','education-spending','superfund-right-to-sue','crime','duty-free-exports', 'export-administration-act-south-africa']
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data", header=None, names=head)


# In[45]:


featureNames = data.columns.values[1:17]


# In[6]:


data.replace("?","-",inplace=True)


# In[10]:


featureNames = head[1:17]


# In[4]:


_values = data.values


# In[5]:


#targets
Y = _values[:,0]
#train data
X = _values[:,1:17]


# In[114]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# In[ ]:


y_train.argmax()


# In[128]:


class treeClassifier(): 
    def fit(self, X_train, y_train, names):
        self.names = names
        self.X_train = X_train
        self.y_train = y_train
        tree = self.make_tree(X_train, y_train, names)
        return DecisionTreeModel(tree,names)
    
    
    def predict(self, X_test):
        pass
    
    def calc_entropy(self,p):
        if p!=0:
            return -p * np.log2(p)
        else:
            return 0
        
    
    def get_values(self,dataset, feature):
        values = []
        for datapoint in dataset:
            if datapoint[feature] not in values:
                values.append(datapoint[feature])
                
        return values
    
    def cal_info_gain(self, data, classes, feature):
        gain = 0
        nData = len(data)
        # List the values that feature can take
        #print feature;
        values = []
        print data
        for datapoint in data:
            if datapoint[feature] not in values:
                #print datapoint[feature]
                values.append(datapoint[feature])
                
                
        featureCounts = np.zeros(len(values))
        entropy = np.zeros(len(values))
        valueIndex = 0
        
        for value in values:
            dataIndex = 0
            newClasses = []
            for datapoint in data:
                if datapoint[feature] == value:
                    featureCounts[valueIndex]+=1
                    newClasses.append(classes[dataIndex])              
                dataIndex +=1
                
                
            classValues = []
            for aclass in newClasses:
                if classValues.count(aclass) == 0:
                    classValues.append(aclass)
            classCounts = np.zeros(len(classValues))
            classIndex = 0
            for classValue in classValues:
                for aclass in newClasses:
                    if aclass == classValue:
                        classCounts[classIndex]+=1
                classIndex +=1
                    
            for classIndex in range(len(classValues)):
                entropy[valueIndex] += self.calc_entropy(float(classCounts[classIndex]))/sum(classCounts)
                
            gain += float(featureCounts[valueIndex]) / nData * entropy[valueIndex]
            valueIndex += 1
        return gain
                
    def most_frequent(self):
        counts = np.zeros(len(self.y_train))
        list_target = list(y_train)
        for i in xrange(len(random)):
            counts[i] = list_target.count(y_train[i])
        return self.y_train[counts.argmax()]
        
    def make_tree(self, data, classes, featuresNames):    
        newData = np.array([])
        newClasses = np.array([])
        newNames = np.array([])     
        nData = len(data);
        nFeatures = len(featuresNames);
        
        if isinstance(classes,str):
            return classes
        
        if nData == 0 or nFeatures == 0 or len(np.unique(data)):
            if len(classes) != 0:
                target_set = set(classes)
                frequency = [0] * len(target_set)
                index = 0
                for value in target_set:
                    frequency[index] = np.count_nonzero(classes == value)
                    index += 1

                default = classes[np.argmax(frequency)]
            else:
                default = self.most_frequent()

            return default

           
        elif list(classes).count(classes[0]) == nData:
            return classes[0]
        else:
            
            values = []
            gain = np.zeros(nFeatures)
            for feature in range(nFeatures):
                g = self.cal_info_gain(data, classes,feature)
                values.extend(self.get_values(data, feature))
            bestFeature = np.argmin(gain)
            tree = {featuresNames[bestFeature]: {}}
            
            
            for value in values:
                index = 0;
                for datapoint in data:
                    if datapoint[bestFeature] == value:
                        if bestFeature == 0:
                            datapoint = datapoint[1:]
                            newNames = featuresNames[1:]
                        elif bestFeature == nFeatures:
                            datapoint = datapoint[:-1]
                            newNames = featuresNames[:-1]
                        else:
                            datapoint = datapoint[:bestFeature]
                            datapoint = np.append(datapoint, datapoint[bestFeature+1:])
                            newNames = featuresNames[:bestFeature]
                            newNames.extend(featuresNames[bestFeature+1:])
                        newData = np.append(newData, datapoint)
                        newClasses = np.append(newClasses,classes[index])
                    index +=1
                
            subtree = self.make_tree(newData,newClasses,newNames)
            
            tree[featuresNames[bestFeature][value]] = subtree
        return tree
                
                
        


# In[144]:


class DecisionTreeModel:
    def __init__(self, tree, feature_names):
        self.tree = tree
        self.model = []
        self.feature_names = feature_names
        
        

    def get_node(self, tree, row):
        if isinstance(tree, str):
            return tree

        key = next(iter(tree))
        key_index = np.where(self.feature_names == key)

        node_value = row[key_index][0]
        return self.get_node(tree[key][node_value], row)
    
    def predict(self, data):
        for row in data:
            self.model.append(self.get_node(self.tree, row))

        return self.model
    


# In[151]:


tree = treeClassifier()
tree_model = tree.fit(X_train, y_train, featureNames)


# In[147]:


target = tree_model.predict(X_test)


# In[148]:


count = 0;
for i in xrange(len(target)):
    if target[i] == y_train[i]:
        count += 1
        
accuracy = count / len(target) * 100
print "Accuracy: ", accuracy, "%"


# In[153]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz

