import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p=2):
    return np.power(np.sum(np.abs(x1 - x2) ** p), 1/p)

def chebyshev_distance(x1, x2):
    return np.max(np.abs(x1 - x2))

def compute_distances(X_train, x_test, distance_func=euclidean_distance):
    distances = []

    for x_train in X_train:
        distance = distance_func(x_train, x_test)
        distances.append(distance)

    return distances

def majority_vote(labels):
    label_count = {}

    for label in labels:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1

    max_count = -1
    majority_label = None
    for label, count in label_count.items():
        if count > max_count:
            max_count = count
            majority_label = label

    return majority_label


def k_nearest_neighbor_predict(X_train, y_train, X_test, k=3, regression=False, distance_func=euclidean_distance):
    y_pred = []

    for x_test in X_test:
        dists = compute_distances(X_train, x_test, distance_func)
        distance_dict = {idx: dist for idx, dist in enumerate(dists)}
        sorted_distance_dict = dict(sorted(distance_dict.items(), key=lambda item: item[1]))
        nearest_indices = list(sorted_distance_dict.keys())[:k]
        nearest_values = y_train[np.array(nearest_indices)]

        if regression:
            pred = np.mean(nearest_values)
        else:
            pred = majority_vote(nearest_values)

        y_pred.append(pred)

    return np.array(y_pred)