import sys
from sklearn import datasets
from math import log2
import pandas


class Tree(object):
    """"The Tree"""
    def __init__(self):
        self.name = ""
        self.children = dict()

    def set_name(self, name):
        self.name = name

    def add_child(self, key):
        self.children[key] = Tree()

    def display(self):
        print(self.name, self.children)
        print("down")
        for key, value in self.children:
            value.display()
            print("over")
        print("up")


class DTClassifier:
    """Decision Tree Classifier"""

    def __init__(self):
        """Constructor"""
        self.tree = Tree()

    def fit(self, data_train, target_train):
        """Training Function"""
        self.tree = self.build(self.tree, data_train, target_train,
                               data_train.columns.values.tolist())

    def build(self, tree, data, target, remaining_columns):
        """Recursive Tree Building"""

        # Recursion break condition
        if len(target.target.unique()) == 1 or len(remaining_columns) == 0:
            tree.set_name(target.target.value_counts().index[0])
            return tree

        else:
            # Identify column with lowest entropy
            lowest_entropy = 1
            lowest_entropy_column = ""
            for column in remaining_columns:
                entropy = self.calculate_entropy(data, target, column)
                if entropy < lowest_entropy:
                    lowest_entropy = entropy
                    lowest_entropy_column = column

            # Fill information for the nameless node
            tree.set_name(lowest_entropy_column)
            remaining_columns.remove(lowest_entropy_column)

            # Recurse for each child
            for choice in data[lowest_entropy_column].unique():
                tree.add_child(choice)
                self.build(tree.children[choice],
                           data[data[lowest_entropy_column] == choice],
                           target[data[lowest_entropy_column] == choice],
                           remaining_columns)

            # Return the tree to add to its parent
            return tree

    def calculate_entropy(self, data, target, column):
        """Calculates Entropy of a Data Table Split on Column"""
        total_entropy = 0
        total_rows = len(data.index)

        for choice in data[column].unique():
            entropy = 0
            choice_target = target[data[column] == choice]
            choice_rows = len(choice_target.index)
            weight = choice_rows / float(total_rows)

            for result in choice_target.target.unique():
                result_target = choice_target[choice_target.target == result]
                result_rows = len(result_target.index)
                proportion = result_rows / float(choice_rows)
                entropy -= proportion * log2(proportion)

            entropy *= weight
            total_entropy += entropy

        return total_entropy

    def predict(self, data_test):
        """Testing Function"""
        prediction = []
        for index, row in data_test.iterrows():
            prediction.append(self.find(self.tree, row))
        return prediction

    def find(self, tree, row):
        """Recursive Tree Navigation"""
        if len(tree.children) == 0:
            return tree.name
        else:
            choice = row[tree.name]
            if tree.children.get(choice) is None:
                return 1
            return self.find(tree.children[choice], row)

# A seed to preserve order in shuffling
SEED = 135


def main(argv):
    """Function to test the classes."""

    iris = datasets.load_iris()

    bins = int(input("Input number of bins: "))

    # Decoupling iris data for future flexibility
    data = pandas.DataFrame(iris.data)
    for column in data:
        data[column] = pandas.cut(data[column], bins)
    target = pandas.DataFrame(iris.target, columns=["target"])

    data = data.sample(frac=1, random_state=123).reset_index(drop=True)
    target = target.sample(frac=1, random_state=123).reset_index(drop=True)

    rows = len(data.index)
    top = round(rows * .7)
    data_train = data.head(top)
    target_train = target.head(top)
    data_test = data.tail(rows - top)
    target_test = target.tail(rows - top)

    classifier = DTClassifier()
    classifier.fit(data_train, target_train)

    prediction = classifier.predict(data_test)

    num_correct = 0
    i = 0
    for index, row in target_test.iterrows():
        if prediction[i] == row.target:
            num_correct += 1
        i += 1
    accuracy = num_correct / len(prediction)
    percent = accuracy * 100
    print("Accuracy: " + "{0:.2f}".format(percent) + "%")

# Main shell function call
if __name__ == "__main__":
    main(sys.argv)
