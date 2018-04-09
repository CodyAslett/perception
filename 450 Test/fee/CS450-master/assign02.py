from sklearn import datasets
from collections import Counter
import random
import sys


class KnnClassifier:
    """A classifier that doesn't learn and always predicts 0"""

    def __init__(self, k):
        self.data = []
        self.k = k

    def fit(self, data_train, target_train):
        for i in range(len(data_train)):
            data_train[i] = data_train[i].tolist()
            data_train[i].append(target_train[i])
            self.data.append(data_train[i])

    def predict(self, data_test):
        predictions = []

        for point in data_test:
            distances = []

            for row in self.data:
                distance = 0

                for i in range(len(row) - 1):
                    distance += abs(row[i] - point[i])
                pair = [distance, row[4]]
                distances.append(pair)
            distances.sort(key=lambda item: item[0])
            neighbors = []

            for count in range(self.k):
                neighbors.append(distances[count][1])
            counter = Counter(neighbors)
            predictions.append(counter.most_common()[0][0])

        return predictions


# A seed to preserve order in shuffling
SEED = 135


def main(argv):
    """Main shell function"""
    # Loading iris data
    iris = datasets.load_iris()

    # Decoupling iris data for future flexibility
    data = iris.data
    target = iris.target

    # User input number of slices
    slices = int(input("Input number of slices: "))
    slice_size = int(len(data) / slices)

    k = int(input("Input k value: "))

    # Shuffling data and target
    random.seed(SEED)
    random.shuffle(data)
    random.seed(SEED)
    random.shuffle(target)

    # Slicing data and target
    data_sliced = []
    target_sliced = []
    for i in range(slices):
        start = i * slice_size
        end = (i + 1) * slice_size
        data_sliced.append(data[start: end])
        target_sliced.append(target[start: end])

    # Training and Testing
    accuracy_sum = 0
    for i in range(slices):
        # Decoupling Classifier for future flexibility
        classifier = KnnClassifier(k)

        # Defining Training data and target
        data_train = []
        target_train = []
        for j in range(slices):
            if j != i:
                data_train.extend(data_sliced[j])
                target_train.extend(target_sliced[j])

        # Defining Testing data and target
        data_test = data_sliced[i]
        target_test = target_sliced[i]

        # Training
        classifier.fit(data_train, target_train)

        # Testing
        prediction = classifier.predict(data_test)

        # Calculating Accuracy
        num_correct = 0
        for j in range(len(prediction)):
            if prediction[j] == target_test[j]:
                num_correct += 1
        accuracy = num_correct / len(prediction)

        # Summing Accuracy for calculation of average
        accuracy_sum += accuracy

    # Calculating and Displaying Average Accuracy
    accuracy_average = accuracy_sum / slices
    percent = accuracy_average * 100
    print("Accuracy: " + "{0:.2f}".format(percent) + "%")


# Main shell function call
if __name__ == "__main__":
    main(sys.argv)
