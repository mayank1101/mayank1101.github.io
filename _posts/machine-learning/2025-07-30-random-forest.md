---
layout: post
title: "Random Forests: Ensemble Learning with Decision Trees"
date: 2025-07-30
series: "Machine Learning for Engineers"
series_author: "Mayank Sharma"
excerpt: "Master random forests, understanding bagging, feature importance, and how ensemble methods improve prediction accuracy."
---

Continuing our journey through machine learning, today we explore **Random Forests**, the ensemble method that combines the wisdom of multiple decision trees. Imagine you're trying to decide which movie to watch. You could ask one friend for their opinion, but they might have peculiar taste. Instead, you poll ten friends, and go with the majority vote. Even if some friends have quirky preferences, the group consensus is usually more reliable than any single opinion. This is the power of ensemble learning: combining multiple "weak" predictors to create one strong predictor. Random Forests apply this idea to decision trees, creating one of machine learning's most powerful and widely-used algorithms.

As with the [previous post on decision trees](/2025/07/23/decision-trees.html), every formula below comes with a plain-English explanation and a worked example, so you don't need a stats background to follow along.

## Table of Contents

1. [Introduction: From Single Trees to Forests](#introduction-from-single-trees-to-forests)
2. [The Problem with Single Decision Trees](#the-problem-with-single-decision-trees)
3. [Ensemble Learning Fundamentals](#ensemble-learning-fundamentals)
4. [Bootstrap Aggregating (Bagging)](#bootstrap-aggregating-bagging)
5. [Random Forests: Adding Random Feature Selection](#random-forests-adding-random-feature-selection)
6. [The Random Forest Algorithm](#the-random-forest-algorithm)
7. [Out-of-Bag (OOB) Error Estimation](#out-of-bag-oob-error-estimation)
8. [Feature Importance in Random Forests](#feature-importance-in-random-forests)
9. [Random Forests for Regression](#random-forests-for-regression)
10. [Advantages and Limitations](#advantages-and-limitations)
11. [Comparison with Other Methods](#comparison-with-other-methods)
12. [Conclusion](#conclusion)

## Introduction: From Single Trees to Forests

### The Journey from Trees to Forests

In the previous tutorial, we learned that decision trees are:
- Highly interpretable
- Fast to train and predict
- Able to capture non-linear relationships

But they have a critical weakness:
- **High variance**: Small changes in the training data can produce a completely different tree

Random Forests solve this by training many trees and combining their predictions. The result is a model that:
- Maintains the strengths of decision trees
- Dramatically reduces variance through averaging
- Achieves state-of-the-art performance on many tasks

### What Makes Random Forests Special?

Random Forests excel because they:

1. **Reduce overfitting** by averaging many trees that individually overfit
2. **Handle high-dimensional data** naturally
3. **Provide feature importance** rankings
4. **Require minimal preprocessing** (no scaling, handles missing values)
5. **Are robust to outliers** and noise
6. **Parallelize easily** (trees can be trained independently of one another)
7. **Work well "out of the box"** with default parameters

## The Problem with Single Decision Trees

### High Variance Demonstration

Decision trees are highly sensitive to small changes in training data. Consider training a decision tree on three slightly different samples of the same underlying data:

- **Dataset 1**: 100 points, 60% accuracy on test set, tree depth = 8
- **Dataset 2**: Same distribution, different 100 points, 58% accuracy, depth = 12
- **Dataset 3**: Another sample, 62% accuracy, depth = 6

The trees look completely different despite coming from the same underlying distribution! This sensitivity to which exact data points you happen to train on is called **high variance**.

**Mathematical Definition**:

In statistical learning theory, variance measures how much a model's predictions would change if you retrained it on a different sample of data:

$$\text{Variance}[\hat{f}(x)] = \mathbb{E}_{\mathcal{D}}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$$

This looks dense, but it's the same idea as the "spread" formula from the decision trees post, just applied to *models* instead of data points. In plain English: imagine training your model on many different random samples of data ($\mathcal{D}$), and for each one, recording its prediction for the same input $x$. Now look at how far each of those predictions strays from the *average* prediction across all those retrainings, square each distance, and average the squared distances. That average is the variance.

- $\hat{f}(x)$ is the prediction a trained model makes for input $x$
- $\mathcal{D}$ represents a particular random training dataset drawn from the underlying population
- $\mathbb{E}[\cdot]$ (pronounced "expectation") just means "average over many repeats" — $\mathbb{E}_{\mathcal{D}}[\cdot]$ means "average over many different training datasets $\mathcal{D}$"

Decision trees have **high variance** because their greedy, hierarchical structure amplifies small data changes: one different data point can change an early split, which cascades into a totally different tree below it.

### The Bias-Variance Tradeoff

Decision trees are **low bias** but **high variance** models. Total prediction error can be broken down into three pieces that add together:

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

- **Bias**: Error from a model that's too simple to capture the real pattern (underfitting)
- **Variance**: Error from a model being overly sensitive to exactly which training data it saw (overfitting)
- **Irreducible Error**: Random noise in the data itself, which no model can ever eliminate

**Single Decision Trees**:
- Low bias (flexible enough to fit complex patterns)
- **High variance** (too sensitive to which data points it happened to train on)

**Random Forests**:
- Low bias (inherited from the individual trees)
- **Low variance** (averaging many trees cancels out their individual quirks)

## Ensemble Learning Fundamentals

Ensemble methods combine multiple models to improve performance. The key idea is that a group of "weak" learners (individually mediocre models) can become a "strong" learner when their predictions are combined. If you have several independent predictors, each making somewhat random errors, averaging their predictions cancels out a lot of that randomness.

**For Regression**:

Suppose we have $B$ independent models $\hat{f}_1(x), \hat{f}_2(x), \ldots, \hat{f}_B(x)$ (think of $B$ as "number of trees in the forest"), and each one has variance $\sigma^2$ (i.e., each tree, on its own, is equally noisy).

The average prediction across all $B$ models:
$$\hat{f}_{\text{avg}}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}_b(x)$$

This just says: add up the prediction from every tree ($b$ going from 1 to $B$), then divide by how many trees there are — an ordinary average, exactly like averaging test scores.

A basic result from statistics tells us the variance of that average is:
$$\text{Var}[\hat{f}_{\text{avg}}(x)] = \frac{\sigma^2}{B}$$

**Result**: If each individual tree has variance $\sigma^2$, averaging $B$ *independent* trees shrinks the variance down to $\sigma^2$ divided by $B$. Concretely, if a single tree has $\sigma^2 = 100$, then averaging 100 independent trees brings that down to $100 / 100 = 1$, a 100x reduction, assuming the trees don't share any patterns.

**With Correlation**:

In reality, trees aren't fully independent, they're all trained on samples of the same dataset, so they tend to make similar mistakes. We measure how similar their errors are with a correlation value $\rho$ (the Greek letter "rho"), which ranges from 0 (completely independent) to 1 (identical mistakes every time). Once we account for that:

$$\text{Var}[\hat{f}_{\text{avg}}(x)] = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$

This formula blends two effects: a fixed penalty ($\rho \sigma^2$) that never goes away no matter how many trees you add, plus a shrinking penalty ($\frac{1-\rho}{B}\sigma^2$) that does shrink as $B$ grows.

Try plugging in numbers to see this concretely. Say $\sigma^2 = 1$ and $\rho = 0.2$:
- With $B = 10$ trees: $0.2(1) + \frac{0.8}{10}(1) = 0.2 + 0.08 = 0.28$
- With $B = 100$ trees: $0.2(1) + \frac{0.8}{100}(1) = 0.2 + 0.008 = 0.208$
- With $B = 10{,}000$ trees: $0.2(1) + \frac{0.8}{10{,}000}(1) = 0.2 + 0.00008 \approx 0.2$

Notice the total keeps sliding toward $0.2$ but never gets below it, no matter how many more trees you throw at it. That's the floor set by $\rho \sigma^2$.

As $B$ gets very large ($B \to \infty$, meaning "as the number of trees approaches infinity"), the second term shrinks to zero, leaving:
$$\text{Var}[\hat{f}_{\text{avg}}(x)] \to \rho \sigma^2$$

In other words: no matter how many trees you add, the variance can never drop below $\rho \sigma^2$. This is the key insight behind Random Forests:

- Adding more trees reduces variance, but only up to a point set by $\rho$
- **Decorrelating trees (making $\rho$ smaller) is what actually breaks through that floor** — this is exactly the problem Random Forests solve, as we'll see in a moment

### Types of Ensemble Methods

**1. Bagging (Bootstrap Aggregating)**:
- Train models on **different data subsets** (bootstrap samples, explained below)
- Average predictions (regression) or vote (classification)
- Examples: Random Forests, Bagged Trees

**2. Boosting**:
- Train models **sequentially**, each new model focused on correcting the previous models' errors
- Weight training examples by how difficult they were to predict
- Examples: AdaBoost, Gradient Boosting, XGBoost

**3. Stacking**:
- Train several diverse base models
- Train a **meta-model** whose job is to combine their outputs
- Learn the optimal way to weight/combine the base models

Random Forests use **Bagging** plus one extra trick: random feature selection, covered next.

## Bootstrap Aggregating (Bagging)

### What is Bootstrapping?

**Bootstrap Sampling**: From a dataset with $n$ observations, randomly draw $n$ new observations **with replacement**. "With replacement" means after you pick a data point, you put it back before picking again, so the same point can be picked more than once.

**Example**:

Original data: [1, 2, 3, 4, 5]

Bootstrap sample 1: [1, 1, 3, 5, 5]
Bootstrap sample 2: [2, 2, 3, 4, 5]
Bootstrap sample 3: [1, 2, 2, 4, 5]

Notice each bootstrap sample is still 5 items long (the same size as the original), but some values repeat and others are missing entirely. That's expected, and is actually the whole point.

**Properties**:
- Each sample has the same size as the original dataset
- Some observations appear multiple times
- Some observations don't appear at all (about 37%, as we'll calculate next)

**Mathematical**: What's the chance a specific observation is *not* picked in a single draw? Since there are $n$ items and we pick 1 uniformly at random, the chance of picking that particular item is $\frac{1}{n}$, so the chance of *not* picking it is $\left(1 - \frac{1}{n}\right)$.

Since we draw $n$ times independently (with replacement), we multiply that "not picked" probability by itself $n$ times:
$$P(\text{not selected}) = \left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.368$$

The middle step uses a well-known calculus fact: as $n$ gets large, $\left(1-\frac{1}{n}\right)^n$ gets very close to $\frac{1}{e} \approx 0.368$ ($e$ here is Euler's number, roughly 2.718, the same constant behind compound interest and exponential growth). You don't need to derive this yourself, but you can sanity-check it with a small example: for $n = 10$, $(1 - \frac{1}{10})^{10} = 0.9^{10} \approx 0.349$, already close to $0.368$. The bigger $n$ gets, the closer this number creeps toward $1/e$. Just remember the punchline:

So about **63.2%** of the original data appears in each bootstrap sample (that's $1 - 0.368$), and the remaining **36.8%** is left out entirely. That leftover chunk becomes important later, in the Out-of-Bag section.

### The Bagging Algorithm

```
function Bagging(data, B):
    models = []

    for b = 1 to B:
        # 1. Create bootstrap sample
        sample_b = bootstrap_sample(data)

        # 2. Train model on this sample
        model_b = train_model(sample_b)

        # 3. Store model
        models.append(model_b)

    return models

function Predict(x, models):
    # For regression: average predictions
    predictions = [model.predict(x) for model in models]
    return mean(predictions)

    # For classification: majority vote
    votes = [model.predict(x) for model in models]
    return most_common(votes)
```

### Why Bagging Reduces Variance

**Intuition**: Different bootstrap samples produce different trees, since each tree sees a slightly different slice of the data. Averaging their predictions smooths out each individual tree's quirks, the same way averaging ten friends' movie ratings smooths out any one friend's odd taste.

**The Math, Revisited**:

We already saw the formula for $B$ independent models, each with variance $\sigma^2$:

$$\text{Var}[\text{avg}] = \frac{\sigma^2}{B}$$

With $B = 100$ trees, this says the ensemble's variance is just $\frac{\sigma^2}{100}$, i.e., **1% of a single tree's variance**, if the trees were fully independent.

Of course, trees trained via bagging aren't fully independent, they share the same underlying dataset. Even so, accounting for realistic correlation still helps a lot. Suppose $\rho = 0.5$ (a fairly high, pessimistic correlation):

$$\text{Var}[\text{avg}] = 0.5\sigma^2 + \frac{0.5}{100}\sigma^2 = 0.5\sigma^2 + 0.005\sigma^2 = 0.505\sigma^2$$

Even in this less optimistic case, we still cut variance roughly in half compared to a single tree ($\sigma^2$) — still a **~50% variance reduction**, just from bagging alone.

## Random Forests: Adding Random Feature Selection

### The Limitation of Pure Bagging

Bagging alone has a hidden weakness: bootstrap samples, while different, are still quite similar to each other (remember, ~63% of each sample overlaps with the original data). If one feature is very strong (e.g., "income" for loan prediction), **almost every tree** will independently discover it and split on it first, near the root. This makes the trees correlated with each other, which — as we saw above — puts a floor on how much variance bagging alone can remove.

### Random Feature Selection

Random Forests fix this with one extra trick: **at every single split, hide most of the features from the tree, and force it to choose only from a small random subset.** Concretely: at each split, randomly sample $m$ features out of the total $p$ available features, and only consider those $m$ when deciding how to split.

**Typical values**:
- Classification: $m = \sqrt{p}$ (the square root of the total feature count)
- Regression: $m = \frac{p}{3}$ (one third of the total feature count)

**Example**:

Total features: 16
Random Forest samples: $\sqrt{16} = 4$ features per split

**Tree 1, Root split**: Consider features [2, 7, 10, 15]
**Tree 1, Next split**: Consider features [1, 3, 8, 12]
**Tree 2, Root split**: Consider features [4, 6, 9, 11]
...

Notice the feature subset is re-randomized at *every single split*, not just once per tree. So even within one tree, different branches end up considering different features.

### Why This Works

By forcing each split to ignore most features, we prevent every tree from converging on the same "obviously strong" feature at the root. As a result:

- Some trees are forced to discover and rely on weaker, secondary features
- Others explore entirely different combinations of features
- The final ensemble ends up covering a much wider variety of patterns than bagging alone would

**Mathematical Insight**:

Recall that correlation ($\rho$) between trees is what limits how much variance averaging can remove. Shrinking $m$ (fewer features considered per split) directly drives that correlation down:

$$m \downarrow \implies \rho \downarrow \implies \text{Var}[\text{ensemble}] \downarrow$$

(Read the arrows as "leads to": smaller $m$ leads to smaller $\rho$, which leads to smaller ensemble variance.)

But there's a tradeoff. If $m$ gets too small, each individual tree is forced to make decisions using only weak, less-informative features, so the quality of each individual tree suffers:

$$m \downarrow \text{ too much} \implies \text{individual tree quality} \downarrow$$

The optimal $m$ balances these two competing effects: diverse-but-weak trees versus strong-but-correlated trees. The $\sqrt{p}$ and $\frac{p}{3}$ defaults mentioned above are well-tested starting points that work well across most datasets, though it's still worth tuning $m$ for your specific problem.

## The Random Forest Algorithm

### Complete Algorithm

```
function RandomForest(X, y, B, m):
    """
    X: feature matrix (n × p)
    y: target vector (n)
    B: number of trees
    m: number of features to sample per split
    """
    forest = []

    for b = 1 to B:
        # 1. Bootstrap sample
        X_b, y_b = bootstrap_sample(X, y)

        # 2. Build tree with random feature selection
        tree_b = build_tree(X_b, y_b, m)

        # 3. Add to forest
        forest.append(tree_b)

    return forest

function build_tree(X, y, m):
    """Build single decision tree with random features."""
    if stopping_criterion_met:
        return leaf_node(y)

    # Randomly select m features
    features = random_sample(p, m)

    # Find best split among these m features
    best_feature, best_threshold = find_best_split(X[:, features], y)

    # Split data
    X_left, y_left, X_right, y_right = split(X, y, best_feature, best_threshold)

    # Recursively build subtrees
    left_tree = build_tree(X_left, y_left, m)
    right_tree = build_tree(X_right, y_right, m)

    return decision_node(best_feature, best_threshold, left_tree, right_tree)

function predict_forest(x, forest):
    """Make prediction with random forest."""
    # Classification: majority vote
    votes = [tree.predict(x) for tree in forest]
    return most_common(votes)

    # Regression: average
    predictions = [tree.predict(x) for tree in forest]
    return mean(predictions)
```

A quick note on reading this pseudocode: `X` (capital) is the whole feature matrix, one row per example and one column per feature; `y` is the corresponding list of correct answers. The notation `n × p` just means "n rows by p columns."

### Classification vs Regression

**Classification**:
- Each tree votes for a class
- Final prediction: **majority vote** (most common class wins) or **soft voting** (average the trees' predicted probabilities instead of just their final answer)

**Soft Voting Example**:

Tree 1: P(Class A) = 0.8, P(Class B) = 0.2
Tree 2: P(Class A) = 0.6, P(Class B) = 0.4
Tree 3: P(Class A) = 0.7, P(Class B) = 0.3

Average: P(Class A) = 0.7, P(Class B) = 0.3 → Predict Class A

Here we simply averaged each tree's confidence for Class A: $(0.8 + 0.6 + 0.7) / 3 = 0.7$, and likewise for Class B. Since Class A's average confidence (0.7) beats Class B's (0.3), Class A wins.

**Regression**:
- Each tree predicts a continuous value
- Final prediction: **mean** (or median, for robustness against outlier trees)

**Mean Example**:

Tree 1: 145,000
Tree 2: 152,000
Tree 3: 148,000

Average: (145,000 + 152,000 + 148,000) / 3 = 148,333

## Out-of-Bag (OOB) Error Estimation

### What is OOB Data?

OOB data is a powerful, mostly-free feature of random forests: it lets us estimate how well the model will generalize to new data, without setting aside a separate validation set. Recall from the bootstrap section that each tree's bootstrap sample uses only about 63% of the original data. The remaining ~37%, the data points a given tree never saw during training, are called **Out-of-Bag (OOB)** samples for that tree.

**For each observation $i$**:
- Some trees in the forest never saw observation $i$ during training (it was "out of the bag" for those trees)
- We can use *just those trees* to predict observation $i$, as if it were unseen test data
- Then compare that prediction to the true, known value

### OOB Error Calculation

```
function calculate_oob_error(X, y, forest):
    n = len(y)
    oob_predictions = [None] * n

    for i in range(n):
        # Find trees where observation i was OOB
        oob_trees = [tree for tree in forest if i not in tree.training_indices]

        if len(oob_trees) > 0:
            # Predict using only OOB trees
            votes = [tree.predict(X[i]) for tree in oob_trees]
            oob_predictions[i] = most_common(votes)

    # Calculate error on OOB predictions
    oob_error = error_rate(y, oob_predictions)
    return oob_error
```

### OOB as Cross-Validation

The key insight: every observation ends up predicted only by trees that never trained on it, exactly the situation you want when estimating real-world performance. Because of this, OOB error is an **unbiased estimate** of test error. "Unbiased" here just means it doesn't systematically skew too optimistic or too pessimistic, on average it lands right on the true error you'd see on brand-new data.

**Advantages**:
- No need for a separate validation set
- Comes essentially for free (no extra computation beyond training the forest itself)
- Lets you use all your data for training, instead of holding some back
- Useful for model selection (comparing different hyperparameter settings, i.e., the settings you choose before training, like how many trees to grow or how many features to sample at each split)

It's a neat property of Random Forests that OOB error closely approximates leave-one-out cross-validation error, a technique that would normally mean training the model $n$ separate times (once per data point, leaving that one point out each time) just to test how well it generalizes. Random Forests get a very similar estimate essentially for free, at a fraction of that computational cost.

### OOB Score

**OOB Score** = 1 - OOB Error Rate (so a higher OOB score means a better-performing model)

Typically:
- OOB Score ≈ 80-95% indicates good fit
- OOB Score < 70% suggests underfitting or poor features
- A large gap between training accuracy and OOB accuracy suggests overfitting (though this is less common with Random Forests than with single trees)

## Feature Importance in Random Forests

Feature importance is a powerful tool for understanding which features drive predictions. Random Forests provide **interpretability** through feature importance scores, which help answer:

- Which features are most predictive?
- Can we remove irrelevant features?
- What drives the predictions?

### Method 1: Mean Decrease in Impurity (Gini Importance)

**Idea**: Add up how much every split that used feature $j$ reduced impurity, across every tree in the forest. Features that consistently produce big impurity drops get high importance scores.

**Algorithm**:

For each feature $j$:

$$\text{Importance}(j) = \sum_{t \in \text{Trees}} \sum_{s \in \text{Splits using } j} \Delta \text{Impurity}(s)$$

In plain English: look at every tree $t$ in the forest, and within each tree, look at every split $s$ that happened to use feature $j$. For each of those splits, calculate how much it reduced impurity ($\Delta \text{Impurity}(s)$, where $\Delta$ just means "the change in"). Add up all of those impurity drops across every tree.

Each individual impurity drop is calculated exactly like the Gini Gain from the decision trees post:
$$\Delta \text{Impurity}(s) = \text{Gini}_{\text{parent}} - \left(p_{\text{left}} \cdot \text{Gini}_{\text{left}} + p_{\text{right}} \cdot \text{Gini}_{\text{right}}\right)$$

(Here $p_{\text{left}}$ and $p_{\text{right}}$ are the fraction of examples that went to the left and right child respectively.)

**Normalize**, so all feature importances add up to 1 (making them easy to read as percentages):
$$\text{Importance}_{\text{normalized}}(j) = \frac{\text{Importance}(j)}{\sum_{k=1}^{p} \text{Importance}(k)}$$

This just divides each feature's raw importance by the sum of *all* features' raw importances, the same normalization trick used for any set of scores you want to turn into percentages.

**Advantages**:
- Fast to compute (comes as a byproduct of training, no retraining needed)
- Built-in to scikit-learn (`model.feature_importances_`)

**Disadvantages**:
- Biased toward high-cardinality features (features with many possible values look artificially important, for the same reason a "Customer ID" feature looked artificially good in the decision trees post)
- Can be misleading with correlated features (if two features carry the same information, the importance gets arbitrarily split between them)

### Method 2: Permutation Importance

**Idea**: If a feature is actually useful, scrambling its values should hurt the model's accuracy. If a feature is useless, scrambling it shouldn't matter at all. So we can measure importance directly, by scrambling one feature at a time and seeing how much worse the model gets.

**Algorithm**:

```
function permutation_importance(model, X_test, y_test, feature_j):
    # 1. Get baseline error
    baseline_error = error(model.predict(X_test), y_test)

    # 2. Permute feature j
    X_permuted = X_test.copy()
    X_permuted[:, j] = shuffle(X_permuted[:, j])

    # 3. Calculate error with permuted feature
    permuted_error = error(model.predict(X_permuted), y_test)

    # 4. Importance = increase in error
    importance_j = permuted_error - baseline_error

    return importance_j
```

In words: measure the model's error normally (`baseline_error`), then randomly shuffle just one feature's column so its values are scrambled but every other feature stays the same, measure error again (`permuted_error`), and take the difference. A big jump in error means that feature mattered a lot.

**Advantages**:
- Unbiased (doesn't have the high-cardinality bias that Gini importance has)
- Works with any model, not just tree-based ones
- Captures feature interactions, since the model is evaluated as a whole

**Disadvantages**:
- Slower (requires re-running predictions for every feature you want to test)
- Requires a separate test/validation set to evaluate on

### Practical Example

Suppose we're predicting house prices with features:
- Size: 45% importance
- Location: 30% importance
- Age: 15% importance
- Color: 5% importance
- Owner's name: 5% importance

**Interpretation**:
- Size and Location are the key drivers of price
- Age has a moderate impact
- Color and Owner's name are nearly irrelevant → these could likely be removed without hurting accuracy

### Feature Selection with Random Forests

**Strategy**:

1. Train a Random Forest on all available features
2. Compute feature importances
3. Remove features below some importance threshold (e.g., < 0.01)
4. Retrain the model using only the selected features
5. Compare performance before and after, to confirm nothing important was lost

**Benefits**:
- Reduces overfitting
- Faster training and prediction (fewer features to consider)
- A simpler, more interpretable final model

## Random Forests for Regression

### Algorithm Differences

**Classification**:
- Leaf prediction: Majority class
- Ensemble: Majority vote
- Splitting criterion: Gini or Entropy

**Regression**:
- Leaf prediction: Mean of target values in that leaf
- Ensemble: Average of all trees' predictions
- Splitting criterion: MSE (Mean Squared Error)

### MSE Splitting Criterion

**For regression**, at each node, we use the same Mean Squared Error formula introduced in the decision trees post:

$$\text{MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2$$

Where $\bar{y}$ is the mean (average) of the target values in node $S$, and the formula measures the average squared distance of each value from that mean.

**MSE Reduction** (this plays the same role as information gain did for classification, i.e., "how much did this split help?"):

$$\text{MSE Reduction} = \text{MSE}(S) - \left(\frac{|S_L|}{|S|}\text{MSE}(S_L) + \frac{|S_R|}{|S|}\text{MSE}(S_R)\right)$$

As before: parent MSE, minus the weighted average of the two children's MSE ($S_L$ and $S_R$ being the left and right child).

### Regression Example

**Predicting house prices**, using 100 trees in the forest:

Tree 1 predicts: $245,000
Tree 2 predicts: $238,000
Tree 3 predicts: $251,000
...
Tree 100 predicts: $247,000

**Final prediction**: Average across all 100 trees = $245,500

**Standard deviation of predictions**: $\sigma = \$8,000$ (this measures how much the 100 individual tree predictions disagree with each other, in the original dollar units — a useful built-in signal of how confident the model is. Note this is $\sigma$, not $\sigma^2$: taking the square root of variance converts it back into the same units as the predictions themselves, which is what makes it usable directly below)

**Prediction intervals**: A rough 95%-confidence range can be built as $245,500 ± 2 \times 8,000 = [\$229,500, \$261,500]$. (The "×2" comes from the fact that roughly 95% of a bell-curve-shaped distribution falls within two standard deviations of the mean — a standard shortcut in statistics.)

### Advantages for Regression

- Captures non-linear relationships naturally
- Robust to outliers (individual trees may overfit to an outlier, but the ensemble average smooths it out)
- Provides free uncertainty estimates via the variance of the trees' predictions
- Handles interaction effects between features automatically

## Advantages and Limitations

### Advantages

**1. Excellent Performance**
- Achieves state-of-the-art results on many tasks
- Often competitive with deep learning on tabular data
- Reduces overfitting compared to single trees

**2. Versatility**
- Works for classification and regression
- Handles numerical and categorical features
- Robust to outliers and noise
- Minimal data preprocessing required

**3. Feature Insights**
- Provides feature importance rankings
- Helps identify irrelevant features
- Useful for exploratory data analysis

**4. Robustness**
- Works well with default parameters
- Less prone to overfitting than single trees
- OOB error provides built-in validation, for free

**5. Parallelization**
- Trees can be trained independently
- Easy to parallelize across CPU cores
- Scalable to large datasets

**6. No Assumptions**
- Non-parametric, meaning it doesn't assume your data follows some fixed shape (like a straight line or a bell curve) before it even starts learning
- Captures non-linear relationships
- Handles feature interactions automatically

### Limitations

**1. Interpretability**
- An ensemble of 100+ trees is effectively a "black box"
- Can't easily visualize the decision logic the way you can with a single tree
- Feature importance helps, but doesn't give the same clarity as tracing a single tree's path

**2. Prediction Speed**
- Slower than a single tree, since a prediction requires querying every tree in the forest
- May not meet real-time requirements for very large forests
- There's a direct trade-off: more trees means more accuracy, but also more prediction time

**3. Memory Usage**
- Must store every tree in the forest
- Can become large for deep trees combined with many features
- Not ideal for memory-constrained environments

**4. Extrapolation**
- Can't predict beyond the range of values seen in training
- Regression predictions are limited to somewhere between the min and max of the training targets
- Example: if trained only on houses priced $100k–$500k, the model can't predict $600k, no matter how large the input house is

**5. Bias Toward High-Cardinality Features**
- Features with many unique values can get inflated importance scores (the same issue covered in the Feature Importance section)
- Worth double-checking feature importance results with permutation importance if this matters for your use case

**6. Not Optimal for All Tasks**
- Linear relationships: simpler models (linear/logistic regression) tend to work better and are easier to interpret
- Text/sparse data: linear models or neural networks are usually a better fit
- Time series: specialized methods (ARIMA, LSTM) often outperform Random Forests

## Comparison with Other Methods

### Random Forests vs Single Decision Tree

| Aspect | Single Tree | Random Forest |
|--------|------------|---------------|
| **Variance** | High | Low (averaging) |
| **Bias** | Low | Low |
| **Interpretability** | High (visualize) | Low (ensemble) |
| **Accuracy** | Moderate | High |
| **Training Time** | Fast | Slower |
| **Prediction Time** | Very Fast | Moderate |
| **Overfitting** | Prone | Resistant |

### Random Forests vs Gradient Boosting

| Aspect | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| **Training** | Parallel | Sequential |
| **Speed** | Faster training | Slower training |
| **Accuracy** | Very good | Often better |
| **Overfitting** | Less prone | More prone |
| **Tuning** | Less sensitive | More sensitive |
| **Interpretability** | Moderate | Moderate |

### Random Forests vs Linear Models

| Aspect | Random Forest | Linear Models |
|--------|---------------|---------------|
| **Linear Relationships** | Overkill | Optimal |
| **Non-linear Relationships** | Excellent | Poor |
| **High-Dimensional Sparse** | Moderate | Excellent |
| **Interpretability** | Feature importance | Coefficients |
| **Preprocessing** | Minimal | Scaling required |

## Conclusion

Random Forests represent a crucial step in the evolution of machine learning algorithms, moving from single trees to ensembles that combine the strengths of multiple models. Their ability to handle complex, non-linear relationships while staying robust to noise makes them one of the most widely used algorithms in practice.

Random Forests remain a go-to algorithm because they:
- Work well out-of-the-box
- Provide feature insights
- Balance accuracy and simplicity
- Serve as excellent baselines

Master Random Forests, and you'll have a powerful tool that serves you well across countless machine learning problems.

### References

**Original Papers**:
- [Breiman, L. (2001). "Random Forests" - Original paper introducing the algorithm](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [Breiman, L. (1996). "Bagging Predictors" - Foundation of bagging](https://cdn.aaai.org/AAAI/1996/AAAI96-108.pdf)
- [Ho, T.K. (1995). "Random Decision Forests" - Early random subspace method](https://pdfs.semanticscholar.org/b41d/0fa5fdaadd47fc882d3db04277d03fb21832.pdf)
