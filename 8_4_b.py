import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Generate random labels with c equi-distributed classes
def generate_random_points_and_labels(n_points, n_features, c_classes):
    np.random.seed(2)
    X = np.random.rand(n_points, n_features)
    labels = np.random.choice(range(c_classes), n_points)
    return X, labels

def get_number_of_thresholds(data, labels):
    table = [(np.sum(row), label) for row, label in zip(data, labels)]
    sorted_table = sorted(table, key=lambda x: x[0])
    thresholds = 0
    previous_label = None
    for data_sum, label in sorted_table:
        if label != previous_label:
            previous_label = label
            thresholds += 1
    return thresholds

def train_nearest_neighbors(X_train, y_train):
    clf = KNeighborsClassifier(n_neighbors=1) 
    clf.fit(X_train, y_train)
    return clf

def get_number_of_instances_memorized(X_test, y_test, clf):
    y_pred = clf.predict(X_test)
    memorized_instances = np.sum(y_pred == y_test)
    return memorized_instances

def main(n_points=10000, n_features=2, c_classes=3):
    X, labels = generate_random_points_and_labels(n_points, n_features, c_classes)
    clf = train_nearest_neighbors(X, labels)
    thresholds = get_number_of_thresholds(X, labels)
    memorized_instances = get_number_of_instances_memorized(X, labels, clf)
    info_capacity = memorized_instances / thresholds
    print(f"Information Capacity: {info_capacity:.3f} bits per parameter for  {c_classes}-class classification problem.")

if __name__ == "__main__":
    main(n_points=100000, n_features=10, c_classes=2)
    main(n_points=100000, n_features=10, c_classes=3)
    main(n_points=100000, n_features=10, c_classes=4)
    main(n_points=100000, n_features=10, c_classes=5)
    main(n_points=100000, n_features=10, c_classes=6)

# Output:
# Information Capacity: 1.995 bits per parameter for  2-class classification problem.
# Information Capacity: 1.502 bits per parameter for  3-class classification problem.
# Information Capacity: 1.330 bits per parameter for  4-class classification problem.
# Information Capacity: 1.254 bits per parameter for  5-class classification problem.
# Information Capacity: 1.200 bits per parameter for  6-class classification problem.