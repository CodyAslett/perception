from knn import SamKNNClassifier
from sklearn.neighbors import KNeighborsClassifier

#I don't like some of the warnings. They are not that applicable...
import warnings
warnings.filterwarnings("ignore")

# battle
#
# This function applies both my kNN and sklearn kNN algorithm and displays
# the results to the users
def initiate_battle(train_data, test_data, train_target, test_target):

    print()
    print("#---------------------------------------#")
    print("|-------------- Results ----------------|")
    print("#---------------------------------------#")
    print("|   K    |     Sam      |       SK      |")

    for i in range(1, 10):
        sk_count = 0
        sam_count = 0

        sam_classifier = SamKNNClassifier(i)
        sk_classifier = KNeighborsClassifier(n_neighbors = i)

        model_sam = sam_classifier.fit(train_data, train_target)
        model_sk = sk_classifier.fit(train_data, train_target)

        sam_target_predicted = model_sam.predict(test_data)
        sk_target_predicted = model_sk.predict(test_data)

        for index in range(len(test_data)):
            if sk_target_predicted[index] == test_target[index]:
                sk_count += 1

            if sam_target_predicted[index] == test_target[index]:
                sam_count += 1

        sam_accuracy = get_accuracy_battle(sam_count, len(test_data))
        sk_accuracy = get_accuracy_battle(sk_count, len(test_data))

        print("|   {: ^1}    |    {: ^2}%    |     {: ^1}%    |"\
        .format(i, "{:.2f}".format(sam_accuracy), "{:.2f}".format(sk_accuracy)))

    print("#---------------------------------------#")

# get_accuracy()
#
# This function calculates and returns the accuracy
def get_accuracy_battle(count, length):
    return (count / length) * 100
