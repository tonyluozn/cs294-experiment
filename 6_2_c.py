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


# Train a decision tree classifier and evaluate its accuracy
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Function to count the number of if-then clauses (non-leaf nodes)
def count_if_then_clauses(tree):
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    if_then_clauses = 0
    for i in range(n_nodes):
        if children_left[i] != -1 or children_right[i] != -1:
            if_then_clauses += 1
    return if_then_clauses

# Count the non-leaf nodes in the decision tree
num_if_then_clauses = count_if_then_clauses(clf.tree_)
print(f"----- Default Scikit-learn decision tree -----")
print(f"Number of if-then clauses: {num_if_then_clauses}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Training score: {clf.score(X_train, y_train):.2f}")



# Strategy 1: Limit Tree Depth
for depth in range(15,20):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    num_if_then_clauses = count_if_then_clauses(clf.tree_)
    print(f"----- Decision tree with max_depth={depth} -----")
    print(f"Number of if-then clauses: {num_if_then_clauses}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Training score: {clf.score(X_train, y_train):.2f}")

# Strategy 2: Limit Number of Leaf Nodes
for max_leaf_nodes in range(2, 15):
    clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    num_if_then_clauses = count_if_then_clauses(clf.tree_)
    print(f"----- Decision tree with max_leaf_nodes={max_leaf_nodes} -----")
    print(f"Number of if-then clauses: {num_if_then_clauses}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Training score: {clf.score(X_train, y_train):.2f}")