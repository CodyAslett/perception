import numpy as np
import pandas as pd
import math
import decision_tree as watdt
from node import Node
import operator

# Decision Tree
#
#This function helps keep the SamDecisionTree() away from the user
class DecisionTree():
    #nothing realy needs to be initiated here
    def __init__(self):
        pass

    #build the tee
    def fit(self, data, targets, headers):
        return SamDecisionTree(data, targets, headers)   

#SamDecisionTree
#
#This class serves as the heart of the ID3 algorithm.
class SamDecisionTree():
    def __init__(self, data, targets, headers):
        self.data = data
        self.targets = targets
        self.headers = headers

        #merge the data and targets into one dataframe so it can be used to 
        #build the tree.
        frames = [data, targets]
        final_data = pd.concat(frames, axis=1)
        
        self.tree = watdt.build_tree(final_data, headers[:-1])
    
    #overriding the print command to behave differently
    def __repr__(self):
        #the reason this is formatted like this is because Python doesn't like
        #printing with no returns. This basically calls the function that prints everything
        #then prints None. The way this is formatted helps the None from being printed.
        return '' if str(watdt.print_tree(self.tree)) == None else '' 

    #go through each item and begin traversal
    #tree traversal actually happens here
    def predict(self, data):
        predictions = []
        
        for (item, row) in data.iterrows():
            #replicating headers since .remove inside predict_class is destructive
            attr = self.headers[:]

            #append the results
            predictions.append(self.predict_class(self.tree, row, attr))

        return predictions

    #this function traverses the tree and determines the final prediction
    def predict_class(self, node, row, headers):
        for attribute in headers:
            if (attribute == node.name):
                value = row[attribute]
                if (value in node.children):
                    new_node = node.children[value]
                    if (new_node.isLeaf()):
                        return new_node.name
                    else:
                        headers.remove(attribute)
                        return self.predict_class(new_node, row, headers)
                else:
                    return 0
        return 1

    #this function actually goes through the tree recursively and prints
    #out the values.
    def print_tree(self, node):
        print(node.name),

        for (key, value) in node.children.items():
            print(key, "{"),
            self.print_tree(value),
            print("}"),
