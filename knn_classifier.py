import numpy as np
from collections import Counter
from data_loader import build_dataset
import matplotlib.pyplot as plt


def k_nearest_neighbours(train_set, predict, k=3):
    """
    :param train_set: A list of training data subsets
        each subset is a dictionary with two keys - 5 and 6
        and value is a list of all data-points having that label
    :param predict: a single data-point whose k nearest neighbors must be determined
    :param k: number of neighbors
    :return: a list of size of of 3-tuples (distance, neighbour data-point, label)
    """
    distances = []
    for subset in train_set:
        for label in subset:
            for x in subset[label]:
                euclidean_distance = np.linalg.norm(np.array(x) - np.array(predict))
                distances.append((euclidean_distance, x, label))

    neighbours = sorted(distances)[:k]
    return neighbours


def knn_classify(train_set, predict, k=3):
    """
    :param train_set: A list of training data subsets
        each subset is a dictionary with two keys - 5 and 6
        and value is a list of all data-points having that label
    :param predict: a single data-point whose k nearest neighbors must be determined
    :param k: number of neighbors
    :return: the predicted label of the single target data-point
    """

    neighbours = k_nearest_neighbours(train_set, predict, k)
    neighbour_classes = [i[2] for i in neighbours]
    selected_class = Counter(neighbour_classes).most_common(1)[0][0]
    return selected_class


def knn_classify_with_cross_validation(train_set, test_set, k=3):
    """
    :param train_set: A list of training data subsets
        each subset is a dictionary with two keys - 5 and 6
        and value is a list of all data-points having that label
    :param test_set: A single subset of the training data used for cross validation
    :param k: number of neighbors
    :return: the predicted labels of all the data-points in the test set and the accuracy
    """

    predicted_labels = []
    correct = 0
    incorrect = 0
    for true_label in test_set:
        for x in test_set[true_label]:
            predicted_label = knn_classify(train_set, x, k)
            predicted_labels.append(predicted_label)
            if predicted_label == true_label:
                correct += 1
            else:
                incorrect += 1

    accuracy = correct / (correct + incorrect) * 100

    return predicted_labels, accuracy


def ten_fold_cross_validation(full_dataset, num_neighbours=3):

    avg_accuracy = 0
    for i in range(10):
        test_set = full_dataset.pop(0)
        accuracy = knn_classify_with_cross_validation(full_dataset, test_set, k=num_neighbours)[1]
        #print('Experiment %d accuracy = %.2f %%' % (i+1, accuracy))
        full_dataset.append(test_set)
        avg_accuracy += accuracy

    return avg_accuracy / 10


k_values = []
accuracy_values = []


def get_best_k(full_dataset):

    for k in range(1, 31):
        print(len(full_dataset))
        accuracy = ten_fold_cross_validation(full_dataset, num_neighbours=k)
        print('At k = %d Accuracy = %.2f' % (k, accuracy))
        k_values.append(k)
        accuracy_values.append(accuracy)

    best_acc = max(accuracy_values)
    best_k = k_values[accuracy_values.index(best_acc)]

    return best_k, best_acc


def plot_accuracy():
    plt.plot(k_values, accuracy_values, '.-')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig("output.png")


if __name__ == '__main__':
    dataset = build_dataset('./knn-dataset')
    predictions, acc = knn_classify_with_cross_validation(dataset[:-1], dataset[-1])
    print(predictions)
    print(acc)
    print(ten_fold_cross_validation(dataset))

    print(get_best_k(dataset))
    plot_accuracy()
