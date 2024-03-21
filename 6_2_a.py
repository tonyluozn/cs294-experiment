from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Load breast cancer binary dataset from sklearn
print(f"----- Dataset -----")
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Strategy 1: Limit Tree Depth
depths = range(1, 10)
accuracies_depth = []
clauses_depth = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    num_if_then_clauses = count_if_then_clauses(clf.tree_)
    accuracies_depth.append(accuracy_score(y_test, y_pred))
    clauses_depth.append(num_if_then_clauses)

# Strategy 2: Limit Number of Leaf Nodes
leaf_nodes = range(2, 20)
accuracies_leaf = []
clauses_leaf = []

for max_leaf_nodes in leaf_nodes:
    clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    num_if_then_clauses = count_if_then_clauses(clf.tree_)
    accuracies_leaf.append(accuracy_score(y_test, y_pred))
    clauses_leaf.append(num_if_then_clauses)


# Plot for Strategy 1
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.plot(depths, accuracies_depth, marker='o', linestyle='-', color='b', label='Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Max Depth')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(depths, clauses_depth, marker='s', linestyle='-', color='r', label='If-Then Clauses')
plt.xlabel('Max Depth')
plt.ylabel('Number of If-Then Clauses')
plt.title('If-Then Clauses vs Max Depth')
plt.grid(True)

plt.tight_layout()
plt.show()


# Plot for Strategy 2
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.plot(leaf_nodes, accuracies_leaf, marker='o', linestyle='-', color='b', label='Accuracy')
plt.xlabel('Max Leaf Nodes')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Max Leaf Nodes')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(leaf_nodes, clauses_leaf, marker='s', linestyle='-', color='r', label='If-Then Clauses')
plt.xlabel('Max Leaf Nodes')
plt.ylabel('Number of If-Then Clauses')
plt.title('If-Then Clauses vs Max Leaf Nodes')
plt.grid(True)

plt.tight_layout()
plt.show()