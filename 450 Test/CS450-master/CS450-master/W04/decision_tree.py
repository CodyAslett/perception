import math
import operator
import numpy as np
from node import Node

# print_tree
#
#This function goes through and prints the tree in a nice, beautiful way to the user.
def print_tree(node):
    print(node.name, end="")

    if (not node.isLeaf()):
        for (key, value) in node.children.items():
            print(" ", key, "-> {", end="")
            print_tree(value),
            print("}", end="")

# entropy
#
# Takes a number and returns the entropy calculation for that p
def entropy(p):
    if p == 0:
        return 0
    else:
        return -p * math.log2(p)

# calculate_entropy
#
#This function goes through and calculates the weighted entropy of each category
def calculate_entropy(data, attribute): 
    #we only care about the attributes so here, we get a list of all unique values
    the_set = data[attribute].unique()
    no_bin = 0.0
    yes_bin = 0.0
    total_entropy = 0.0
        
    for attr in the_set:
        #determine whether the target is a 0 or a 1. This helps us incorporate
        #weighted entropy. 
        for index in range(0, len(data)):
            if attr == data[attribute][index]:
                if data.iloc[:,-1][index] == 0:
                    no_bin += 1
                else:
                    yes_bin += 1

        #calculations needed
        total = no_bin + yes_bin
        no_bin_entropy = entropy(no_bin/total)
        yes_bin_entropy = entropy(yes_bin/total)
        single_entropy = no_bin_entropy + yes_bin_entropy
        weighted_entropy = single_entropy * (total / len(data))
        total_entropy += weighted_entropy
        no_bin = yes_bin = 0.0

    return total_entropy

# most_common
#
# This function goes through and returns the most common feature
def most_common(data):
    count = {}
    
    unique, counts = np.unique(data, return_counts=True)
    count = dict(zip(unique, counts))
    most_common = max(count.items(), key=operator.itemgetter(1))[0]

    return most_common

# build_tree
#
#This function is a recursion function and is the heart of the ID3
#algorithm. 
def build_tree(data, attributes, removed = []):
    # make an empty node
    currentNode = Node()

    # remove any used attributes
    remaining = set(attributes) - set(removed)

    remaining_t = data.iloc[:, -1].unique()

    # if no more rows SOMETHING IS WRONG
    if len(data) == 0:
        currentNode.name = "INVALID DATA"
        return currentNode

    # else if no more options
    if len(remaining_t) == 1:
        currentNode.name = remaining_t[0]
        return currentNode
    elif len(remaining) == 0:
        leaf = Node(most_common(data.iloc[:, -1]))
        currentNode.appendChild(leaf, leaf)
        
        return currentNode
    else:        
        # calculate the best value and set it to the node
        entropies = {}
        for attribute in remaining:
            entropies[attribute] = calculate_entropy(data, attribute)
            
        # find the lowest value in our list of entropies
        bestVal = min(entropies, key = entropies.get)
        currentNode.name = bestVal
        # get all possible values of root
        poss_values = data[bestVal].unique()
            
        # build the tree
        childNodes = {}
        for poss_value in poss_values:
            data_subset = data[data[bestVal] == poss_value]
            data_subset.reset_index(inplace=True, drop=True)
            # remove this attribute
            removed.append(bestVal)

            #recursively call build_tree to build the next set of nodes
            node = build_tree(data_subset, attributes, removed)

            #creating the children
            childNodes[poss_value] = node
            currentNode.children = childNodes

            #reset the removed array for the next possible value
            removed = []

    return currentNode