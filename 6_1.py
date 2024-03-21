import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

def generate_data(n_points, n_features):
    data = np.zeros((n_points, n_features))
    for i in range(n_points):
        for j in range(n_features):
            data[i, j] = i >> j & 1
    return data

def generate_random_labels(n_points, c_classes):
    labels = np.random.choice(range(c_classes), n_points)
    return labels

def req_points_for_memorization(X, y):
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
    for i in range(num_function):
        labels = generate_random_labels(n, c_classes)
        req_points = req_points_for_memorization(X, labels)
        avg_mem_size += req_points
    avg_mem_size /= num_function
    print(f"d={d}: n_full={2**d}, \
          Avg. req. points for memorization n_avg={avg_mem_size:.2f}, \
          n_full/n_avg={(2**d)/avg_mem_size}")

if __name__ == "__main__":
    main(d=2, num_function=16, c_classes=2)
    main(d=4, num_function=2**4, c_classes=2)
    main(d=6, num_function=2**6, c_classes=2)

# d=2: n_full=4,           Avg. req. points for memorization n_avg=2.69,           n_full/n_avg=1.4883720930232558
# d=4: n_full=16,           Avg. req. points for memorization n_avg=8.94,           n_full/n_avg=1.7902097902097902
# d=6: n_full=64,           Avg. req. points for memorization n_avg=33.47,           n_full/n_avg=1.912231559290383
