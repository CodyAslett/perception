import sys
from math import log2


class Tree(object):
    """"The Tree"""
    def __init__(self, name, data):
        self.name = name
        self.children = dict()
        self.data = []

    def __str__(self):
        return self.name

    def add_layer(self):
        choices = dict()
        for attribute_number in range(len(self.data[0]) - 1):
            divided_data = dict()
            for row in self.data:
                if row[attribute_number] in divided_data:
                    divided_data[row[attribute_number]].append(row)
                else:
                    divided_data[row[attribute_number]] = []
                    divided_data[row[attribute_number]].append(row)

            # calculate entropy of dict
            entropy = 0
            for key, value in divided_data:
                weight = len(value) / float(len(self.data))
                entropy += self.calculate_entropy(value) * weight
            choices[attribute_number] = entropy

            # And so on...

    def calculate_entropy(self, data):
        entropy = 0
        targets = dict()
        for row in data:
            if row[-1] not in targets:
                targets[row[-1]] = 1
            else:
                targets[row[-1]] += 1
        for target in targets:
            count = targets[target]
            proportion = count / float(len(data))
            entropy -= proportion * log2(proportion)
        return entropy

    def add_child(self, child, choice):
        assert isinstance(child, Tree)
        self.children[choice] = child


class DTClassifier:
    """Decision Tree Classifier"""

    def __init__(self):
        self.tree = Tree("root", [])

    def fit(self, data_train, target_train):
        pass

    def predict(self, data_test):
        predictions = []
        return predictions

def main(argv):
    print("DT")

# Main shell function call
if __name__ == "__main__":
    main(sys.argv)