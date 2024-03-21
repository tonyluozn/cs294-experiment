import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def generate_random_points_and_labels(n_points, n_features, c_classes):
    # Generate random points with n_features
    np.random.seed(2)
    X = np.random.rand(n_points, n_features)
    # Generate random labels with c equi-distributed classes
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
    # Train a nearest neighbors classifier
    clf = KNeighborsClassifier(n_neighbors=1)  # Using 1-NN for simplicity
    clf.fit(X_train, y_train)
    return clf

def get_number_of_instances_memorized(X_test, y_test, clf):
    # Predict the labels for the test set
    y_pred = clf.predict(X_test)
    # For 1-NN, we'll consider the number of correctly classified instances as memorized instances
    memorized_instances = np.sum(y_pred == y_test)
    return memorized_instances

def main(n_points=10000, n_features=2, c_classes=3):
    # Step 1: Generate random points and labels
    X, labels = generate_random_points_and_labels(n_points, n_features, c_classes)

    # Step 2: Train the nearest neighbors classifier
    clf = train_nearest_neighbors(X, labels)

    # Step 3: Information Capacity
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