import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Load breast cancer binary dataset from sklearn
print(f"----- Dataset -----")
def generate_random_binary_dataset(n_samples=1000, n_features=2):
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=n_samples)
    return X, y
X, y = generate_random_binary_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Number of features: {len(data.feature_names)}")
print(f"Number of training instances: {len(X_train)}")
print(f"Number of testing instances: {len(X_test)}")


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
num_if_then_clauses = count_if_then_clauses(clf.tree_)
print(f"----- Default Scikit-learn decision tree -----")
print(f"Number of if-then clauses: {num_if_then_clauses}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Train Accuracy: {clf.score(X_train, y_train):.2f}")


# Strategy 1: Limit Tree Depth
depths = range(1, 35)
accuracies_depth = []
clauses_depth = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    num_if_then_clauses = count_if_then_clauses(clf.tree_)
    accuracies_depth.append(clf.score(X_train, y_train))
    clauses_depth.append(num_if_then_clauses)


# Plot for Strategy 1
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.plot(depths, accuracies_depth, linestyle='-', color='b', label='Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Train Accuracy')
plt.title('Accuracy vs Max Depth')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(depths, clauses_depth, linestyle='-', color='r', label='If-Then Clauses')
plt.xlabel('Max Depth')
plt.ylabel('Number of If-Then Clauses')
plt.title('If-Then Clauses vs Max Depth')
plt.grid(True)

plt.tight_layout()
plt.show()

# Strategy 2: Limit Number of Leaf Nodes
leaf_nodes = range(2, 300)
accuracies_leaf = []
clauses_leaf = []

for max_leaf_nodes in leaf_nodes:
    clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    num_if_then_clauses = count_if_then_clauses(clf.tree_)
    accuracies_leaf.append(clf.score(X_train, y_train))
    clauses_leaf.append(num_if_then_clauses)

# Plot for Strategy 2
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.plot(leaf_nodes, accuracies_leaf, linestyle='-', color='b', label='Accuracy')
plt.xlabel('Max Leaf Nodes')
plt.ylabel('Train Accuracy')
plt.title('Accuracy vs Max Leaf Nodes')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(leaf_nodes, clauses_leaf, linestyle='-', color='r', label='If-Then Clauses')
plt.xlabel('Max Leaf Nodes')
plt.ylabel('Number of If-Then Clauses')
plt.title('If-Then Clauses vs Max Leaf Nodes')
plt.grid(True)

plt.tight_layout()
plt.show()