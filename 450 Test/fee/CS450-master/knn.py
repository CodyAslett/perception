import sys
from collections import Counter


class KnnClassifier:
    """A classifier that doesn't learn and always predicts 0"""

    data = []
    k = 1

    def fit(self, data_train, target_train):
        for i in range(len(data_train)):
            data_train[i].append(target_train[i])
        self.data = data_train

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


def main(argv):
    print("knn")

# Main shell function call
if __name__ == "__main__":
    main(sys.argv)
