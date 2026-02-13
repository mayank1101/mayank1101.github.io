---
layout: post
title: "Decision Trees: Understanding Tree-Based Learning"
date: 2026-02-05
series: "Machine Learning Series"
series_author: "Mayank Sharma"
series_image: "/assets/images/2026-02-05-decision-trees/decision-trees.png"
excerpt: "Learn how decision trees work, including splitting criteria, pruning, and visualization techniques for interpretable models."
---

Continuning in our journey through machine learning, today we turn to **decision trees**, a powerful and intuitive algorithm that mimics human decision-making. Imagine you're playing a game of "20 Questions" where you try to guess what animal someone is thinking of. You might ask: "Does it live in water?" If yes, you've ruled out all land animals. Then: "Does it have scales?" This narrows it down further. Each question splits the possibilities into smaller groups until you identify the answer. This is exactly how decision trees work, they ask a series of yes/no questions about data features, splitting the dataset at each step until they can make accurate predictions.

## Table of Contents

1. [Introduction: Why Decision Trees Matter](#introduction-why-decision-trees-matter)
2. [The Anatomy of a Decision Tree](#the-anatomy-of-a-decision-tree)
3. [How Decision Trees Make Decisions](#how-decision-trees-make-decisions)
4. [Splitting Criteria: Measuring Impurity](#splitting-criteria-measuring-impurity)
5. [Information Gain: Choosing the Best Split](#information-gain-choosing-the-best-split)
6. [Tree Construction Algorithms](#tree-construction-algorithms)
7. [Decision Trees for Regression](#decision-trees-for-regression)
8. [Implementation from Scratch](#implementation-from-scratch)
9. [Advantages and Limitations](#advantages-and-limitations)
10. [Conclusion](#conclusion)
11. [Jupyter Notebook](#jupyter-notebook)

## Introduction: Why Decision Trees Matter

### The Power of Interpretability

In an era where "black box" models like deep neural networks dominate headlines, decision trees stand out as beautifully transparent. When a decision tree makes a prediction, you can trace the exact path it took, which questions it asked and which answers led to the conclusion. This interpretability makes decision trees invaluable in scenarios where understanding the reasoning behind predictions is crucial, for example:

- **Healthcare**: Doctors need to understand *why* a model recommends a treatment
- **Finance**: Loan officers must explain *why* someone was denied credit
- **Legal systems**: Decisions affecting people's lives require justification
- **Business**: Executives want to understand the logic behind recommendations

### What Makes Decision Trees Special?

Decision trees offer unique advantages:

1. **No feature scaling required**: Unlike neural networks or SVMs
2. **Handle both categorical and numerical features**: Naturally
3. **Non-linear decision boundaries**: Can model complex relationships
4. **Automatic feature selection**: Irrelevant features are simply not used
5. **Fast prediction**: Just follow a path down the tree
6. **Visual interpretation**: Can be drawn and understood visually

## The Anatomy of a Decision Tree

### Tree Structure Components

A decision tree consists of:

**1. Root Node**: The starting point containing all training data

**2. Internal Nodes (Decision Nodes)**: Each represents a question about a feature
   - Example: "Is age > 30?"
   - Contains a splitting criterion

**3. Branches (Edges)**: Represent the answer to the question (Yes/No, or multiple categories)

**4. Leaf Nodes (Terminal Nodes)**: The final decision/prediction
   - For classification: The predicted class
   - For regression: The predicted value (usually the mean)

### A Simple Example: Predicting Customer Purchases

Let's say we're predicting whether a customer will buy a product based on age and income:

```
                    [Root: All Customers]
                            |
                    Is Income > $50k?
                    /              \
                  Yes              No
                  /                  \
        [Is Age > 35?]          [Predict: No Buy]
           /        \
         Yes        No
         /            \
[Predict: Buy]  [Predict: No Buy]
```

**Reading this tree:**
- If income ≤ $50k → Predict "No Buy"
- If income > $50k AND age > 35 → Predict "Buy"
- If income > $50k AND age ≤ 35 → Predict "No Buy"

### Mathematical Representation

We can represent a decision tree as a function:

$$f(x) = \sum_{m=1}^{M} c_m \cdot \mathbb{1}(x \in R_m)$$

Where:
- $M$ is the number of leaf nodes
- $R_m$ is the region (feature space) corresponding to leaf $m$
- $c_m$ is the predicted value for region $m$
- $\mathbb{1}(x \in R_m)$ is an indicator function (1 if true, 0 otherwise)

## How Decision Trees Make Decisions

Now let's dive into the magic of how decision trees actually make decisions.

### The Learning Process

Building a decision tree involves **recursive partitioning**:

1. **Start with all data** at the root node
2. **Find the best feature and split point** that divides the data
3. **Create child nodes** for each split
4. **Repeat** the process for each child node
5. **Stop** when a stopping criterion is met

This is actually a **greedy algorithm**, at each step, we make the locally optimal choice without looking ahead.

### What Makes a "Good" Split?

A good split should:
- **Increase purity**: Child nodes should be more homogeneous than the parent
- **Reduce uncertainty**: We should be more confident about predictions
- **Maximize information gain**: Learn as much as possible from each question

### The Splitting Process: A Concrete Example

Suppose we have this dataset of 10 people:

| Age | Income | Bought |
|-----|--------|--------|
| 25  | 40k    | No     |
| 30  | 45k    | No     |
| 35  | 60k    | Yes    |
| 40  | 65k    | Yes    |
| 45  | 70k    | Yes    |
| 28  | 50k    | No     |
| 50  | 75k    | Yes    |
| 22  | 35k    | No     |
| 38  | 58k    | Yes    |
| 42  | 62k    | Yes    |

**Current distribution**: 6 "Yes", 4 "No" (impure)

**Candidate split: Income > 55k**
- Left child (Income ≤ 55k): 1 Yes, 4 No → More pure!
- Right child (Income > 55k): 5 Yes, 0 No → Pure!

This split significantly reduces impurity, making it a good choice.

## Splitting Criteria: Measuring Impurity

To find the best split, we need to quantify "impurity" or "disorder" in a node. To do this, three main metrics are used:

### 1. Gini Impurity (CART)

**The Intuition**: Gini impurity measures the probability of incorrectly classifying a randomly chosen element if we randomly assign a label according to the distribution in the node.

**Mathematical Definition**:

$$\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2$$

Where:
- $S$ is the set of examples in the node
- $C$ is the number of classes
- $p_i$ is the proportion of examples of class $i$

**Properties**:
- Range: [0, 0.5] for binary classification, [0, 1-1/C] for C classes
- Gini = 0: Perfect purity (all examples same class)
- Gini = 0.5: Maximum impurity for binary (50-50 split)

**Example Calculation**:

For a node with 6 "Yes" and 4 "No":

$$p_{\text{Yes}} = \frac{6}{10} = 0.6, \quad p_{\text{No}} = \frac{4}{10} = 0.4$$

$$\text{Gini} = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48$$

**Interpretation**: 0.48 indicates high impurity (close to maximum of 0.5).

### 2. Entropy (ID3, C4.5)

**The Intuition**: Entropy comes from information theory and measures the average amount of information needed to identify the class of an example.

**Mathematical Definition**:

$$\text{Entropy}(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

Where:
- $p_i$ is the proportion of class $i$
- By convention, $0 \log_2(0) = 0$

**Properties**:
- Range: [0, log₂(C)]
- Entropy = 0: Perfect purity
- Entropy = log₂(C): Maximum impurity (uniform distribution)

**Example Calculation**:

Same node (6 "Yes", 4 "No"):

$$\text{Entropy} = -(0.6 \log_2(0.6) + 0.4 \log_2(0.4))$$
$$= -(0.6 \times (-0.737) + 0.4 \times (-1.322))$$
$$= -(-0.442 - 0.529) = 0.971$$

**Interpretation**: 0.971 out of maximum 1.0 indicates high disorder.

### 3. Classification Error

**The Intuition**: Simply the fraction of examples that would be misclassified if we assign the majority class to all examples.

**Mathematical Definition**:

$$\text{Error}(S) = 1 - \max_i(p_i)$$

**Example Calculation**:

Same node (6 "Yes", 4 "No"):

$$\text{Error} = 1 - \max(0.6, 0.4) = 1 - 0.6 = 0.4$$

**When to Use Each Metric**:

- **Gini Impurity**:
  - Default choice for CART
  - Computationally faster (no logarithms)
  - Works well in practice

- **Entropy/Information Gain**:
  - Theoretically grounded in information theory
  - Tends to produce more balanced trees
  - Slightly slower (logarithm computation)

- **Classification Error**:
  - Less sensitive to changes in probabilities
  - Rarely used in practice

### Comparing Gini vs Entropy

For binary classification:

```
Probability p₁   Gini         Entropy
0.0              0.0          0.0
0.1              0.18         0.469
0.2              0.32         0.722
0.3              0.42         0.881
0.4              0.48         0.971
0.5              0.5          1.0
```

**Key observation**: Both reach maximum at 50-50 split, both reach 0 at pure nodes. Entropy is slightly more sensitive to probability changes.

## Information Gain: Choosing the Best Split

### Definition

**Information Gain** measures how much a split reduces impurity:

$$\text{IG}(S, A) = \text{Impurity}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Impurity}(S_v)$$

Where:
- $S$ is the parent node
- $A$ is the attribute/feature we're considering
- $S_v$ is the subset where attribute $A$ has value $v$
- $|S|$ is the number of examples in $S$

### Concrete Example: Choosing Between Features

Dataset (10 examples): 6 "Yes", 4 "No"

**Parent Gini**: 0.48

**Option 1: Split on Income > 55k**

Left (Income ≤ 55k): 5 examples (1 Yes, 4 No)
$$\text{Gini}_{\text{left}} = 1 - (0.2^2 + 0.8^2) = 1 - 0.68 = 0.32$$

Right (Income > 55k): 5 examples (5 Yes, 0 No)
$$\text{Gini}_{\text{right}} = 1 - (1^2 + 0^2) = 0$$

Weighted Gini after split:
$$\text{Gini}_{\text{split}} = \frac{5}{10}(0.32) + \frac{5}{10}(0) = 0.16$$

**Gini Gain**:
$$\text{Gain} = 0.48 - 0.16 = 0.32$$

**Option 2: Split on Age > 35**

Left (Age ≤ 35): 6 examples (2 Yes, 4 No)
$$\text{Gini}_{\text{left}} = 1 - (0.333^2 + 0.667^2) = 0.444$$

Right (Age > 35): 4 examples (4 Yes, 0 No)
$$\text{Gini}_{\text{right}} = 0$$

Weighted Gini after split:
$$\text{Gini}_{\text{split}} = \frac{6}{10}(0.444) + \frac{4}{10}(0) = 0.267$$

**Gini Gain**:
$$\text{Gain} = 0.48 - 0.267 = 0.213$$

**Decision**: Choose Income > 55k (Gini Gain of 0.32 > 0.213)

### Gain Ratio (C4.5 Improvement)

**Problem with Information Gain**: It favors features with many distinct values.

**Example**: An "ID" feature with unique values for each example would perfectly split the data (Gini = 0 for each child), but it's useless for generalization!

**Solution - Gain Ratio**:

$$\text{GainRatio}(S, A) = \frac{\text{IG}(S, A)}{\text{SplitInfo}(S, A)}$$

Where **Split Information** penalizes features with many splits:

$$\text{SplitInfo}(S, A) = -\sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$$

This normalizes the information gain, making comparisons fairer across features with different numbers of values.

## Tree Construction Algorithms

### ID3 (Iterative Dichotomiser 3)

**Algorithm**: Developed by Ross Quinlan in 1986

```
function ID3(examples, attributes):
    if all examples have same class:
        return leaf node with that class

    if attributes is empty:
        return leaf with majority class

    best_attribute = attribute with highest information gain
    tree = new decision node for best_attribute

    for each value v of best_attribute:
        examples_v = subset where best_attribute = v

        if examples_v is empty:
            add leaf with majority class
        else:
            subtree = ID3(examples_v, attributes - {best_attribute})
            add branch to tree for value v with subtree

    return tree
```

**Limitations**:
- Only handles categorical features
- Prone to overfitting
- No pruning
- Doesn't handle missing values

### C4.5 (Successor to ID3)

**Improvements**:
- Handles continuous attributes (finds optimal split points)
- Uses Gain Ratio instead of Information Gain
- Includes pruning to reduce overfitting
- Handles missing values
- Different costs for different errors

**Handling Continuous Attributes**:

For a continuous feature like Age:

1. Sort examples by Age: [22, 25, 28, 30, 35, 38, 40, 42, 45, 50]
2. Consider splits between each consecutive pair
3. For each candidate: Age > 26.5, Age > 29, Age > 32.5, etc.
4. Calculate information gain for each
5. Choose the split with maximum gain

### CART (Classification and Regression Trees)

**Algorithm**: Developed by Breiman, Friedman, Olshen, and Stone (1984)

**Key Differences**:
- Uses **Gini impurity** instead of entropy
- Creates **binary trees only** (each split has exactly 2 children)
- Supports **both classification and regression**
- Uses **cost-complexity pruning**

**CART Algorithm**:

```
function CART(examples):
    if stopping criterion met:
        return leaf node with prediction

    best_split = None
    best_gain = 0

    for each feature:
        for each possible split point:
            gain = calculate_gini_gain(split)
            if gain > best_gain:
                best_gain = gain
                best_split = split

    left_node = CART(examples where feature < threshold)
    right_node = CART(examples where feature >= threshold)

    return decision node with best_split, left_node, right_node
```

**Stopping Criteria**:
1. All examples in node have same class
2. Maximum depth reached
3. Minimum samples per node reached
4. Information gain below threshold
5. Node purity exceeds threshold

## Decision Trees for Regression

### Key Differences from Classification

**Prediction**: Instead of a class label, predict a **continuous value**

**Leaf Node Values**:
- Classification: Majority class
- Regression: **Mean** of target values (or median for robustness)

$$\hat{y} = \frac{1}{|S|} \sum_{i \in S} y_i$$

### Splitting Criteria for Regression

**Mean Squared Error (MSE)**:

$$\text{MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2$$

Where $\bar{y}$ is the mean of $y$ values in node $S$.

**MSE Reduction** (equivalent to information gain):

$$\text{MSE}_{\text{reduction}} = \text{MSE}(S) - \left(\frac{|S_L|}{|S|}\text{MSE}(S_L) + \frac{|S_R|}{|S|}\text{MSE}(S_R)\right)$$

**Mean Absolute Error (MAE)** (alternative):

$$\text{MAE}(S) = \frac{1}{|S|} \sum_{i \in S} |y_i - \text{median}(S)|$$

- More robust to outliers
- Leaf prediction uses median instead of mean

### Regression Example

**Task**: Predict house price based on size

| Size (sq ft) | Price ($k) |
|--------------|------------|
| 800          | 150        |
| 1000         | 180        |
| 1200         | 220        |
| 1500         | 280        |
| 2000         | 350        |
| 2500         | 400        |

**Current MSE** (all data):
- Mean price: 263.33
- MSE: 9,222.22

**Try split: Size ≤ 1200**

Left (Size ≤ 1200): [150, 180, 220]
- Mean: 183.33
- MSE: 816.67

Right (Size > 1200): [280, 350, 400]
- Mean: 343.33
- MSE: 2,222.22

**Weighted MSE after split**:
$$\text{MSE}_{\text{split}} = \frac{3}{6}(816.67) + \frac{3}{6}(2222.22) = 1,519.44$$

**MSE Reduction**:
$$9,222.22 - 1,519.44 = 7,702.78 \quad \text{(Excellent split!)}$$

### Piecewise Constant Functions

Regression trees create **piecewise constant** predictions:

```
Price = 183.33  if Size ≤ 1200
Price = 343.33  if Size > 1200
```

This creates a step function rather than smooth curve. For smoother predictions, ensemble methods (Random Forests, Gradient Boosting) work better.

## Implementation from Scratch

### Node Class

```python
import numpy as np
from collections import Counter

class Node:
    """Represents a node in the decision tree."""

    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        impurity=None,
        n_samples=None
    ):
        # Internal node properties
        self.feature_index = feature_index  # Index of feature to split on
        self.threshold = threshold          # Threshold value for split
        self.left = left                   # Left child node
        self.right = right                 # Right child node

        # Leaf node properties
        self.value = value                 # Predicted class/value for leaf

        # Node statistics
        self.impurity = impurity           # Gini or MSE
        self.n_samples = n_samples         # Number of samples in node
```

### Decision Tree Classifier

```python
class DecisionTreeClassifier:
    """Decision Tree Classifier using Gini impurity."""

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None

    def fit(self, X, y):
        """Build decision tree classifier."""
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _gini(self, y):
        """Calculate Gini impurity."""
        n_samples = len(y)
        if n_samples == 0:
            return 0

        # Count occurrences of each class
        class_counts = np.bincount(y)
        probabilities = class_counts / n_samples

        # Gini = 1 - sum(p_i^2)
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini

    def _best_split(self, X, y):
        """Find the best split for a node."""
        n_samples, n_features = X.shape

        if n_samples <= 1:
            return None, None

        # Current impurity
        parent_impurity = self._gini(y)

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        # Try each feature
        for feature_idx in range(n_features):
            # Get unique values and sort them
            thresholds = np.unique(X[:, feature_idx])

            # Try each unique value as threshold
            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                # Skip if split doesn't satisfy min_samples_leaf
                if (np.sum(left_mask) < self.min_samples_leaf or
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue

                # Calculate weighted impurity of children
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                left_impurity = self._gini(y[left_mask])
                right_impurity = self._gini(y[right_mask])

                weighted_impurity = (
                    (n_left / n_samples) * left_impurity +
                    (n_right / n_samples) * right_impurity
                )

                # Calculate information gain
                gain = parent_impurity - weighted_impurity

                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        # Check minimum impurity decrease
        if best_gain < self.min_impurity_decrease:
            return None, None

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        """Recursively grow the decision tree."""
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Calculate current node impurity
        impurity = self._gini(y)

        # Determine majority class for this node
        class_counts = np.bincount(y, minlength=self.n_classes)
        predicted_class = np.argmax(class_counts)

        # Create node
        node = Node(
            value=predicted_class,
            impurity=impurity,
            n_samples=n_samples
        )

        # Stopping criteria
        if (depth >= self.max_depth if self.max_depth else False):
            return node

        if n_samples < self.min_samples_split:
            return node

        if n_classes == 1:  # Pure node
            return node

        # Find best split
        feature_idx, threshold = self._best_split(X, y)

        if feature_idx is None:  # No valid split found
            return node

        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Recursively build left and right subtrees
        node.feature_index = feature_idx
        node.threshold = threshold
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction for a single sample."""
        # If leaf node, return prediction
        if node.feature_index is None:
            return node.value

        # Traverse left or right based on feature value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        """Predict class labels for samples."""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def predict_proba(self, X):
        """Predict class probabilities for samples."""
        # For simplicity, return one-hot encoded predictions
        # A full implementation would track class distributions in leaves
        predictions = self.predict(X)
        n_samples = len(predictions)
        proba = np.zeros((n_samples, self.n_classes))
        proba[np.arange(n_samples), predictions] = 1
        return proba
```

### Usage Example

```python
# Generate sample data
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=200,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train decision tree
tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)
tree.fit(X_train, y_train)

# Make predictions
y_pred = tree.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2%}")  # Output: ~85-95%
```

### Decision Tree Regressor

```python
class DecisionTreeRegressor:
    """Decision Tree Regressor using MSE."""

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def _mse(self, y):
        """Calculate Mean Squared Error."""
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)

    def _best_split(self, X, y):
        """Find best split for regression."""
        n_samples, n_features = X.shape

        if n_samples <= 1:
            return None, None

        parent_mse = self._mse(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if (np.sum(left_mask) < self.min_samples_leaf or
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])

                weighted_mse = (
                    (n_left / n_samples) * left_mse +
                    (n_right / n_samples) * right_mse
                )

                gain = parent_mse - weighted_mse

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        """Recursively grow regression tree."""
        n_samples = len(y)

        # Leaf value is mean of target values
        predicted_value = np.mean(y)
        node = Node(value=predicted_value, n_samples=n_samples)

        # Stopping criteria
        if (depth >= self.max_depth if self.max_depth else False):
            return node

        if n_samples < self.min_samples_split:
            return node

        # Find best split
        feature_idx, threshold = self._best_split(X, y)

        if feature_idx is None:
            return node

        # Split and recurse
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        node.feature_index = feature_idx
        node.threshold = threshold
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X, y):
        """Build regression tree."""
        self.root = self._grow_tree(X, y)
        return self

    def _traverse_tree(self, x, node):
        """Traverse tree for prediction."""
        if node.feature_index is None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        """Predict continuous values."""
        return np.array([self._traverse_tree(x, self.root) for x in X])
```

## Advantages and Limitations

### Advantages

**1. Interpretability**
- Can visualize entire decision process
- Extract human-readable rules
- Explain predictions to non-technical stakeholders

**2. No Data Preprocessing**
- No feature scaling needed
- Handles mixed data types naturally
- Robust to outliers (they just create splits)

**3. Non-Linear Relationships**
- Captures complex decision boundaries
- Automatic feature interactions
- No assumptions about data distribution

**4. Fast Prediction**
- O(log n) traversal
- Suitable for real-time applications

**5. Feature Selection**
- Automatically identifies important features
- Irrelevant features ignored in splits

**6. Handles Missing Values**
- Multiple strategies available
- Can learn from incomplete data

### Limitations

**1. High Variance (Overfitting)**
- Small data changes → completely different trees
- Solution: Pruning, ensembles (Random Forests, Boosting)

**2. Greedy Learning**
- Locally optimal decisions may miss global optimum
- Can't backtrack to reconsider earlier splits

**3. Bias Toward Features with Many Values**
- Features with more unique values get unfair advantage
- Solution: Use Gain Ratio instead of Information Gain

**4. Difficulty with XOR and Diagonal Boundaries**
- Example: XOR problem requires multiple splits
- Linear boundaries would solve in one step

**5. Unstable**
- Small variance in data → large variance in structure
- Not ideal when reproducibility is critical

**6. Biased with Imbalanced Data**
- Tends to favor majority class
- Solution: Class weights, resampling

**7. Extrapolation Problems (Regression)**
- Can't predict beyond training range
- Predictions are constant in each region

### When to Use Decision Trees

**Good For:**
- Problems requiring interpretability (healthcare, finance, legal)
- Mixed feature types (categorical + numerical)
- Datasets with missing values
- Quick baseline models
- Feature importance analysis
- Non-linear relationships with interactions

**Not Ideal For:**
- When highest accuracy is critical (use ensembles instead)
- When model stability is important
- Linear relationships (use linear models)
- Very high-dimensional sparse data (text classification)
- Extrapolation tasks

## Conclusion

While a single decision tree may seem simple compared to modern deep learning models, understanding decision trees is crucial because they:

1. **Form the foundation** of powerful ensemble methods:
   - **Random Forests**: Combine hundreds of trees
   - **Gradient Boosting** (XGBoost, LightGBM, CatBoost): Sequential tree building
   - These methods dominate tabular data competitions

2. **Offer unmatched interpretability** in many domains where explainability is legally or ethically required

3. **Provide intuition** for how machines make decisions through hierarchical reasoning

Decision trees are not just another algorithm, they're a fundamental way of thinking about how machines can learn to make decisions. Master them, and you'll have intuition that carries through to the most advanced ensemble methods used in industry today.

## Jupyter Notebook

For hands-on practice, check out the companion notebooks - [Decision Trees Tutorial](https://drive.google.com/file/d/11Vj4hUljWJ_t2WD2T9ecRAjWT-wmdY-V/view?usp=sharing)