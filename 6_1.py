import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

def generate_random_points(n_points, n_features):
    # X = np.random.rand(n_points, n_features)
    X = np.random.randint(0, 2, (n_points, n_features))
    return X

def generate_data(n_points, n_features):
    data = np.zeros((n_points, n_features))
    for i in range(n_points):
        for j in range(n_features):
            data[i, j] = i >> j & 1
    return data

def generate_random_labels(n_points, c_classes):
    labels = np.random.choice(range(c_classes), n_points)
    return labels

def generate_unique_random_labels(n_points, c_classes, n_labelings):
    unique_labelings = set()
    while len(unique_labelings) < n_labelings:
        # Generate a random labeling
        labels = tuple(np.random.choice(range(c_classes), n_points))
        unique_labelings.add(labels)

    # Convert each tuple back to a numpy array
    unique_labelings = [np.array(labels) for labels in unique_labelings]
    return unique_labelings

def condensed_nearest_neighbor(X, y):
    # Initialize S with one example from each class (optional)
    # unique_labels = np.unique(y)
    # S_indices = [np.where(y == label)[0][0] for label in unique_labels]
    S_indices = [0]
    changed = True
    while changed:
        changed = False
        for i in range(len(X)):
            if i in S_indices:
                continue  # Skip if already in S
            S = X[S_indices]
            S_labels = y[S_indices]
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(S, S_labels)
            pred = knn.predict([X[i]])[0]
            if pred != y[i]:
                S_indices.append(i)
                changed = True
    return len(S_indices)

def main(d, num_function, c_classes):
    n = 2**d
    avg_mem_size = 0
    X = generate_data(n, d)
    # Y = generate_unique_random_labels(n, c_classes, num_function)
    for i in range(num_function):
        labels = generate_random_labels(n, c_classes)
        req_points = condensed_nearest_neighbor(X, labels)
        avg_mem_size += req_points
    avg_mem_size /= num_function
    print(f"d={d}: n_full={2**d}, \
          Avg. req. points for memorization n_avg={avg_mem_size:.2f}, \
          n_full/n_avg={(2**d)/avg_mem_size}")

if __name__ == "__main__":
    main(d=2, num_function=16, c_classes=2)
    main(d=4, num_function=2**4, c_classes=2)
    main(d=6, num_function=2**6, c_classes=2)
    main(d=8, num_function=2**8, c_classes=2)

