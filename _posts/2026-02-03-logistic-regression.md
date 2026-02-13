---
layout: post
title: "Logistic Regression: Binary and Multiclass Classification"
date: 2026-02-03
series: "Machine Learning Series"
series_author: "Mayank Sharma"
series_image: "/assets/images/2026-02-03-logistic-regression/logistic-regression.png"
excerpt: "Understanding logistic regression for classification problems, from binary to multiclass with sigmoid and softmax functions."
---

Continuing our journey through machine learning, today we turn to **logistic regression**, the workhorse of classification problems. Imagine you're a doctor looking at a patient's medical data—blood pressure, cholesterol level, age, weight and you need to answer a simple but critical question: **Will this patient develop heart disease?** The answer isn't a number like "$245,000" (as in house price prediction). It's a **yes or no**. Welcome to the world of classification, where logistic regression is the fundamental algorithm you need to master.

## Table of Contents

1. [Introduction: From Regression to Classification](#introduction-from-regression-to-classification)
2. [Why Not Just Use Linear Regression?](#why-not-just-use-linear-regression)
3. [The Sigmoid Function: The Heart of Logistic Regression](#the-sigmoid-function-the-heart-of-logistic-regression)
4. [The Logistic Regression Model](#the-logistic-regression-model)
5. [Decision Boundary: Drawing the Line](#decision-boundary-drawing-the-line)
6. [Cost Function: Cross-Entropy Loss](#cost-function-cross-entropy-loss)
7. [Gradient Descent for Logistic Regression](#gradient-descent-for-logistic-regression)
8. [Multiclass Classification](#multiclass-classification)
9. [Regularization in Logistic Regression](#regularization-in-logistic-regression)
10. [Evaluation Metrics for Classification](#evaluation-metrics-for-classification)
11. [Implementation from Scratch](#implementation-from-scratch)
12. [Conclusion](#conclusion)
13. [Jupyter Notebook](#jupyter-notebook)

## Introduction: From Regression to Classification

### What is Classification?

In the [previous tutorial](https://www.hellomayank.in/2026/02/01/linear-regression.html), we learned how linear regression predicts **continuous** values such as house prices, temperatures, stock returns. But many real-world problems require predicting **categories**:

- **Email**: Spam or not spam?
- **Medical Diagnosis**: Disease present or absent?
- **Credit Card**: Fraudulent transaction or legitimate?
- **Image Recognition**: Cat, dog, or bird?

These are **classification problems**, and logistic regression is the foundational algorithm for solving them.

### What is Logistic Regression?

Despite its name, logistic regression is a **classification** algorithm, not a regression one. The "regression" in its name comes from the mathematical technique it builds upon. Logistic regression models the **probability** that an input belongs to a particular class:

$$P(y = 1 \mid x) = \text{What is the probability that this email is spam, given its features?}$$

## Why Not Just Use Linear Regression?

### The Problem with Linear Regression for Classification

You might wonder: "Can I just use linear regression and interpret outputs as probabilities?" Let's see why this fails.

Suppose we're predicting whether a tumor is malignant (1) or benign (0) based on tumor size:

```
Tumor Size (cm) | Malignant?
----------------|----------
     1.0        |     0
     1.5        |     0
     2.0        |     0
     3.0        |     0
     4.5        |     1
     5.0        |     1
     6.0        |     1
     7.0        |     1
```

If we fit a linear regression line $\hat{y} = \theta_0 + \theta_1 x$, we get predictions that:

1. **Go below 0**: A tumor size of 0.5 cm might yield $\hat{y} = -0.3$. What does a probability of $-0.3$ mean? Nothing, probabilities must be between 0 and 1.

2. **Go above 1**: A tumor size of 10 cm might yield $\hat{y} = 1.4$. A probability of 140% is nonsensical.

3. **Are sensitive to outliers**: Adding one extreme data point (say a 20 cm tumor) would dramatically shift the line, changing predictions for all other points.

### What We Need

We need a function that:
- Always outputs values between 0 and 1
- Produces an S-shaped curve (gradual transition from 0 to 1)
- Can be interpreted as a probability

Enter the **sigmoid function**.

## The Sigmoid Function: The Heart of Logistic Regression

### Definition

The sigmoid function (also called the **logistic function**) is defined as:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z$ can be any real number and $e$ is Euler's number ($\approx 2.71828$).

### Properties

The sigmoid function has several elegant properties that make it perfect for classification:

1. **Range**: Output is always in $(0, 1)$—perfect for probabilities
2. **Monotonically increasing**: Larger inputs produce larger outputs
3. **S-shaped curve**: Smooth transition from 0 to 1
4. **Symmetry**: $\sigma(-z) = 1 - \sigma(z)$
5. **At z = 0**: $\sigma(0) = 0.5$ (natural threshold)
6. **As z → +∞**: $\sigma(z) \to 1$
7. **As z → -∞**: $\sigma(z) \to 0$

### Concrete Examples

Let's compute some values to build intuition:

| $z$ | $e^{-z}$ | $\sigma(z)$ | Interpretation |
|-----|-----------|-------------|----------------|
| -5 | 148.41 | 0.0067 | Very unlikely (class 0) |
| -2 | 7.39 | 0.119 | Unlikely |
| -1 | 2.72 | 0.269 | Somewhat unlikely |
| 0 | 1.00 | 0.500 | Toss-up |
| 1 | 0.37 | 0.731 | Somewhat likely |
| 2 | 0.14 | 0.881 | Likely |
| 5 | 0.0067 | 0.993 | Very likely (class 1) |

### The Derivative: A Beautiful Result

The derivative of the sigmoid function has a remarkably elegant form:

$$\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$$

**Proof:**

Starting with $\sigma(z) = (1 + e^{-z})^{-1}$, apply the chain rule:

$$\frac{d\sigma}{dz} = -1 \cdot (1 + e^{-z})^{-2} \cdot (-e^{-z})$$

$$= \frac{e^{-z}}{(1 + e^{-z})^2}$$

$$= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}}$$

$$= \sigma(z) \cdot \frac{1 + e^{-z} - 1}{1 + e^{-z}}$$

$$= \sigma(z) \cdot \left(1 - \frac{1}{1 + e^{-z}}\right)$$

$$= \sigma(z)(1 - \sigma(z))$$

This is important because:
- The gradient is easy and efficient to compute
- Maximum gradient occurs at $z = 0$ where $\sigma'(0) = 0.25$
- The gradient approaches zero for very large or very small $z$ (this causes the **vanishing gradient** problem in deep networks)

## The Logistic Regression Model

### Putting It Together

Logistic regression combines the linear model from linear regression with the sigmoid function:

$$\hat{y} = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

Where:
- $\theta^T x = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$ is the linear combination (same as linear regression)
- $\sigma(\cdot)$ squashes this to a probability between 0 and 1

The output $\hat{y}$ represents $P(y = 1 \mid x; \theta)$ i.e the probability that the input belongs to class 1.

### Making Predictions

Given the probability, we classify using a **threshold** (typically 0.5):

$$\text{predicted class} = \begin{cases} 1 & \text{if } \hat{y} \geq 0.5 \\ 0 & \text{if } \hat{y} < 0.5 \end{cases}$$

Since $\sigma(z) \geq 0.5$ when $z \geq 0$, this is equivalent to:

$$\text{predicted class} = \begin{cases} 1 & \text{if } \theta^T x \geq 0 \\ 0 & \text{if } \theta^T x < 0 \end{cases}$$

### Log-Odds Interpretation

We can rearrange the logistic equation to reveal a powerful interpretation. If $p = \sigma(\theta^T x)$ is the probability of class 1:

$$\ln\left(\frac{p}{1-p}\right) = \theta^T x = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

The left side is the **log-odds** (also called the **logit**), and it turns out that logistic regression is linear in the log-odds. This means:

- Each unit increase in $x_j$ changes the log-odds by $\theta_j$
- Equivalently, each unit increase in $x_j$ **multiplies the odds** by $e^{\theta_j}$

**Example**: If $\theta_{\text{cholesterol}} = 0.02$, then each unit increase in cholesterol multiplies the odds of heart disease by $e^{0.02} \approx 1.02$, i.e., a 2% increase in odds.

## Decision Boundary: Drawing the Line

### Linear Decision Boundary

The **decision boundary** is the surface where the model is equally unsure about both classes, where $P(y = 1 \mid x) = 0.5$, or equivalently, where $\theta^T x = 0$.

For two features, the decision boundary is a **straight line**:

$$\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0$$

Solving for $x_2$:

$$x_2 = -\frac{\theta_0}{\theta_2} - \frac{\theta_1}{\theta_2} x_1$$

**Numerical Example**: If $\theta = [-3, 1, 1]$, the decision boundary is:

$$-3 + x_1 + x_2 = 0 \implies x_2 = 3 - x_1$$

This means:
- Points where $x_1 + x_2 > 3$ are classified as class 1
- Points where $x_1 + x_2 < 3$ are classified as class 0
- The line $x_2 = 3 - x_1$ separates the two regions

### Non-Linear Decision Boundaries

By adding **polynomial features**, logistic regression can model non-linear boundaries. For example, with features $x_1, x_2, x_1^2, x_2^2, x_1 x_2$:

$$\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_1^2 + \theta_4 x_2^2 + \theta_5 x_1 x_2 = 0$$

This can describe circles, ellipses, or more complex shapes. For instance, with $\theta = [-1, 0, 0, 1, 1, 0]$:

$$-1 + x_1^2 + x_2^2 = 0 \implies x_1^2 + x_2^2 = 1$$

This is a **circle** of radius 1 centered at the origin—points inside the circle belong to one class, points outside to another.

## Cost Function: Cross-Entropy Loss

### Why Not MSE?

For linear regression, we used Mean Squared Error. Why not use it for logistic regression?

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 \quad \text{(MSE)}$$

The problem is that when $\hat{y} = \sigma(\theta^T x)$, MSE creates a **non-convex** cost function with many local minima. Gradient descent could get stuck in a local minimum and never find the global one.

We need a cost function that is **convex** (bowl-shaped) so gradient descent is guaranteed to find the global minimum.

### Binary Cross-Entropy Loss

Binary cross-entropy loss is designed for binary classification. It measures the difference between the true label and the predicted probability. Let's look at the formula for a single training example:

$$\mathcal{L}(\hat{y}, y) = -\left[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})\right]$$

Let's understand why this works by examining both cases:

**When $y = 1$ (actual class is positive):**

$$\mathcal{L} = -\log(\hat{y})$$

- If $\hat{y} \to 1$ (correct prediction): $\mathcal{L} \to 0$ (no penalty)
- If $\hat{y} \to 0$ (wrong prediction): $\mathcal{L} \to \infty$ (huge penalty)

**When $y = 0$ (actual class is negative):**

$$\mathcal{L} = -\log(1 - \hat{y})$$

- If $\hat{y} \to 0$ (correct prediction): $\mathcal{L} \to 0$ (no penalty)
- If $\hat{y} \to 1$ (wrong prediction): $\mathcal{L} \to \infty$ (huge penalty)

### The Full Cost Function

Over all $m$ training examples:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\right]$$

### Maximum Likelihood Interpretation

The cross-entropy loss isn't arbitrary, it comes from **Maximum Likelihood Estimation (MLE)**. Here's the connection:

We model the probability of a single data point as:

$$P(y \mid x; \theta) = \hat{y}^{y} \cdot (1 - \hat{y})^{(1-y)}$$

This compact formula captures both cases:
- When $y = 1$: $P = \hat{y}$
- When $y = 0$: $P = 1 - \hat{y}$

The **likelihood** over all independent training examples is:

$$L(\theta) = \prod_{i=1}^{m} P(y^{(i)} \mid x^{(i)}; \theta) = \prod_{i=1}^{m} (\hat{y}^{(i)})^{y^{(i)}} (1 - \hat{y}^{(i)})^{(1-y^{(i)})}$$

Taking the log (to turn products into sums) and negating (to turn maximization into minimization):

$$-\log L(\theta) = -\sum_{i=1}^{m} \left[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\right]$$

Dividing by $m$ gives us exactly our cost function $J(\theta)$. So minimizing cross-entropy loss is equivalent to maximizing the likelihood of the data.

## Gradient Descent for Logistic Regression

### Computing the Gradient

To apply gradient descent, we need the partial derivative of $J$ with respect to each parameter $\theta_j$. The result is:

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}$$

### Full Derivation

Let's derive this step by step. For a single training example with $z = \theta^T x$ and $\hat{y} = \sigma(z)$:

**Step 1**: Apply the chain rule:

$$\frac{\partial \mathcal{L}}{\partial \theta_j} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial \theta_j}$$

**Step 2**: Compute each component:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}$$

$$\frac{\partial \hat{y}}{\partial z} = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$$

$$\frac{\partial z}{\partial \theta_j} = x_j$$

**Step 3**: Multiply them together:

$$\frac{\partial \mathcal{L}}{\partial \theta_j} = \left(-\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}\right) \cdot \hat{y}(1 - \hat{y}) \cdot x_j$$

$$= \left(\frac{-y(1-\hat{y}) + (1-y)\hat{y}}{\hat{y}(1-\hat{y})}\right) \cdot \hat{y}(1-\hat{y}) \cdot x_j$$

$$= (-y + y\hat{y} + \hat{y} - y\hat{y}) \cdot x_j$$

$$= (\hat{y} - y) \cdot x_j$$

**Step 4**: Average over all examples:

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}$$

### A Surprising Result

Notice that the gradient formula is **identical** in form to the gradient for linear regression! The only difference is that $\hat{y} = \sigma(\theta^T x)$ for logistic regression, whereas $\hat{y} = \theta^T x$ for linear regression.

### The Update Rule

The gradient descent update for all parameters simultaneously:

$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)} \quad \text{for } j = 0, 1, \ldots, n$$

In vectorized form:

$$\theta := \theta - \frac{\alpha}{m} X^T (\hat{y} - y)$$

Where:
- $X$ is the $(m \times (n+1))$ design matrix (with bias column)
- $\hat{y}$ is the $(m \times 1)$ vector of predicted probabilities
- $y$ is the $(m \times 1)$ vector of true labels

## Multiclass Classification

So far we've discussed **binary** classification (two classes). What about problems with three or more classes?

### One-vs-Rest (OvR) Strategy

Also called **One-vs-All (OvA)**, this strategy trains $K$ separate binary classifiers for $K$ classes.

**How it works**:
1. For each class $k = 1, 2, \ldots, K$:
   - Treat class $k$ as the positive class
   - Treat all other classes as the negative class
   - Train a binary logistic regression classifier $h_\theta^{(k)}(x)$
2. At prediction time, run all $K$ classifiers and pick the class with the highest probability:

$$\hat{y} = \arg\max_k \; h_\theta^{(k)}(x)$$

**Example with 3 classes (Cat, Dog, Bird)**:
- Classifier 1: Cat vs. (Dog + Bird) → $P(\text{Cat} \mid x) = 0.7$
- Classifier 2: Dog vs. (Cat + Bird) → $P(\text{Dog} \mid x) = 0.2$
- Classifier 3: Bird vs. (Cat + Dog) → $P(\text{Bird} \mid x) = 0.1$
- **Prediction**: Cat (highest probability)

**Limitation**: The probabilities from different classifiers aren't calibrated and don't sum to 1.

### Softmax Regression (Multinomial Logistic Regression)

A more principled approach for multiclass classification is **softmax regression**, which directly generalizes logistic regression to multiple classes.

#### The Softmax Function

For $K$ classes, we compute a score $z_k$ for each class:

$$z_k = \theta_k^T x \quad \text{for } k = 1, 2, \ldots, K$$

The softmax function converts these scores into probabilities:

$$P(y = k \mid x) = \text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**Properties of Softmax**:
1. All outputs are in $(0, 1)$
2. All outputs **sum to 1** (a proper probability distribution)
3. Larger scores get exponentially larger probabilities
4. When $K = 2$, softmax reduces to the sigmoid function

#### Numerical Example

Suppose we have 3 classes and compute scores $z = [2.0, 1.0, 0.1]$:

$$e^{z} = [e^{2.0}, e^{1.0}, e^{0.1}] = [7.389, 2.718, 1.105]$$

$$\text{sum} = 7.389 + 2.718 + 1.105 = 11.212$$

$$\text{softmax}(z) = \left[\frac{7.389}{11.212}, \frac{2.718}{11.212}, \frac{1.105}{11.212}\right] = [0.659, 0.242, 0.099]$$

The model predicts class 1 with 65.9% confidence, class 2 with 24.2%, and class 3 with 9.9%.

#### Cross-Entropy Loss for Multiclass

The multiclass cross-entropy loss (also called **categorical cross-entropy**) is:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{y}_k^{(i)})$$

Where $y_k^{(i)}$ is 1 if example $i$ belongs to class $k$ and 0 otherwise (one-hot encoding).

Since only one $y_k^{(i)} = 1$ per example, this simplifies to:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \log(\hat{y}_{c_i}^{(i)})$$

Where $c_i$ is the true class of example $i$.

#### Gradient for Softmax

The gradient of the cross-entropy loss with respect to the score $z_k$ for class $k$ is:

$$\frac{\partial J}{\partial z_k} = \hat{y}_k - y_k$$

This elegant result mirrors the binary case and makes implementation straightforward.

### OvR vs. Softmax: When to Use Which

| Aspect | One-vs-Rest | Softmax |
|--------|-------------|---------|
| Classes are mutually exclusive | Not required | Required |
| Probability calibration | Poor (don't sum to 1) | Good (sum to 1) |
| Training | K separate models | One joint model |
| Scalability | Easy to parallelize | All classes coupled |
| Use case | Multi-label problems | Single-label classification |

## Regularization in Logistic Regression

### The Overfitting Problem

With many features (especially polynomial features), logistic regression can overfit, learning the training data's noise rather than the underlying pattern.

### L2 Regularization (Ridge)

Add a penalty term that discourages large parameter values:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\right] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

Where:
- $\lambda$ is the **regularization parameter** (controls the trade-off between fit and complexity)
- The sum starts at $j = 1$ (we don't regularize $\theta_0$, the bias)

**Effect**: Shrinks all coefficients toward zero but doesn't eliminate any. Produces smoother decision boundaries.

The gradient with regularization becomes:

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} \theta_j \quad \text{for } j \geq 1$$

### L1 Regularization (Lasso)

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\right] + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|$$

**Effect**: Can drive some coefficients to exactly zero, performing feature selection. Useful when you suspect many features are irrelevant.

### Elastic Net (L1 + L2)

Combines both penalties:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\right] + \frac{\lambda}{m} \left[\frac{(1 - \alpha)}{2} \sum_{j=1}^{n} \theta_j^2 + \alpha \sum_{j=1}^{n} |\theta_j|\right]$$

Where $\alpha \in [0, 1]$ balances L1 and L2 penalties.

### Choosing $\lambda$

- **$\lambda$ too small**: Minimal regularization, possible overfitting
- **$\lambda$ too large**: Heavy regularization, underfitting (model too simple)
- **Best practice**: Use cross-validation to find the optimal $\lambda$

## Evaluation Metrics for Classification

Unlike regression where we use MSE and R², classification requires different metrics.

### Confusion Matrix

For binary classification, predictions fall into four categories:

|  | Predicted Positive | Predicted Negative |
|--|-------------------|--------------------|
| **Actually Positive** | True Positive (TP) | False Negative (FN) |
| **Actually Negative** | False Positive (FP) | True Negative (TN) |

### Key Metrics

**Accuracy**:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Simple but misleading with imbalanced classes. If 99% of emails are not spam, a model that always predicts "not spam" has 99% accuracy but is useless.

**Precision** (of positive predictions, how many were correct?):

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall / Sensitivity** (of actual positives, how many did we find?):

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1 Score** (harmonic mean of precision and recall):

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### ROC Curve and AUC

The **Receiver Operating Characteristic (ROC)** curve plots the True Positive Rate (Recall) against the False Positive Rate at various classification thresholds.

$$\text{TPR} = \frac{TP}{TP + FN}, \quad \text{FPR} = \frac{FP}{FP + TN}$$

The **Area Under the Curve (AUC)** summarizes the ROC curve:
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing
- AUC < 0.5: Worse than random (invert predictions)

### Precision-Recall Trade-off

By adjusting the classification threshold (from the default 0.5), we can trade precision for recall:

- **Lower threshold** (e.g., 0.3): More positive predictions → Higher recall, lower precision
- **Higher threshold** (e.g., 0.7): Fewer positive predictions → Higher precision, lower recall

The right threshold depends on the application:
- **Cancer screening**: Low threshold (don't miss any cases) → prioritize recall
- **Spam filtering**: Higher threshold (don't block legitimate emails) → prioritize precision

## Implementation from Scratch

### Binary Logistic Regression

```python
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionScratch:
    """
    Logistic Regression implemented from scratch using gradient descent.

    Parameters:
    -----------
    learning_rate : float, default=0.01
        Step size for gradient descent
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    reg_lambda : float, default=0.0
        L2 regularization parameter (0 means no regularization)
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, reg_lambda=0.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        self.theta = None
        self.cost_history = []

    @staticmethod
    def sigmoid(z):
        """Compute sigmoid function with numerical stability."""
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit logistic regression model using gradient descent.

        Parameters:
        -----------
        X : array-like, shape (m, n)
            Training features
        y : array-like, shape (m,)
            Binary target values (0 or 1)
        """
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]  # Add bias column
        y = y.reshape(-1, 1)

        # Initialize parameters to zeros
        self.theta = np.zeros((n + 1, 1))

        for iteration in range(self.n_iterations):
            # Forward pass: compute predictions
            z = X_b.dot(self.theta)
            predictions = self.sigmoid(z)

            # Compute cross-entropy cost
            epsilon = 1e-15  # Prevent log(0)
            cost = -(1 / m) * np.sum(
                y * np.log(predictions + epsilon) +
                (1 - y) * np.log(1 - predictions + epsilon)
            )

            # Add regularization cost (excluding bias term)
            if self.reg_lambda > 0:
                cost += (self.reg_lambda / (2 * m)) * np.sum(self.theta[1:] ** 2)

            self.cost_history.append(cost)

            # Compute gradients
            errors = predictions - y
            gradients = (1 / m) * X_b.T.dot(errors)

            # Add regularization gradient (excluding bias term)
            if self.reg_lambda > 0:
                reg_term = (self.reg_lambda / m) * self.theta
                reg_term[0] = 0  # Don't regularize bias
                gradients += reg_term

            # Update parameters
            self.theta -= self.learning_rate * gradients

    def predict_proba(self, X):
        """Return probability estimates for samples."""
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        return self.sigmoid(X_b.dot(self.theta)).flatten()

    def predict(self, X, threshold=0.5):
        """Predict class labels for samples."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        """Compute classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic binary classification data
    m = 200
    X_class0 = np.random.randn(m // 2, 2) + np.array([1, 1])
    X_class1 = np.random.randn(m // 2, 2) + np.array([3, 3])
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (m // 2) + [1] * (m // 2))

    # Train model
    model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    print(f"Training accuracy: {model.accuracy(X, y):.4f}")
    print(f"Learned parameters: {model.theta.flatten()}")
```

### Multiclass with Softmax

```python
class SoftmaxRegressionScratch:
    """
    Softmax Regression (Multinomial Logistic Regression) from scratch.

    Parameters:
    -----------
    learning_rate : float, default=0.01
        Step size for gradient descent
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.cost_history = []

    @staticmethod
    def softmax(z):
        """Compute softmax with numerical stability."""
        # Subtract max for numerical stability (prevents overflow)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        """
        Fit softmax regression model.

        Parameters:
        -----------
        X : array-like, shape (m, n)
            Training features
        y : array-like, shape (m,)
            Target class labels (integers 0, 1, ..., K-1)
        """
        m, n = X.shape
        K = len(np.unique(y))
        X_b = np.c_[np.ones((m, 1)), X]  # Add bias column

        # One-hot encode labels
        Y_onehot = np.zeros((m, K))
        Y_onehot[np.arange(m), y] = 1

        # Initialize parameters: (n+1) x K
        self.theta = np.zeros((n + 1, K))

        for iteration in range(self.n_iterations):
            # Compute scores and probabilities
            z = X_b.dot(self.theta)          # (m, K)
            probs = self.softmax(z)          # (m, K)

            # Compute cross-entropy cost
            epsilon = 1e-15
            cost = -(1 / m) * np.sum(Y_onehot * np.log(probs + epsilon))
            self.cost_history.append(cost)

            # Compute gradient
            gradients = (1 / m) * X_b.T.dot(probs - Y_onehot)  # (n+1, K)

            # Update parameters
            self.theta -= self.learning_rate * gradients

    def predict_proba(self, X):
        """Return probability estimates for each class."""
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        return self.softmax(X_b.dot(self.theta))

    def predict(self, X):
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X, y):
        """Compute classification accuracy."""
        return np.mean(self.predict(X) == y)
```

## Conclusion

### Limitations of Logistic Regression

- **Linear decision boundary**: Cannot capture complex non-linear relationships without feature engineering
- **Feature independence assumption**: Assumes features contribute independently to the log-odds
- **Sensitive to outliers**: Extreme values can pull the decision boundary
- **Doesn't handle feature interactions** automatically (need to create interaction terms manually)
- **Struggles with high-dimensional sparse data** without regularization

### When to Use Logistic Regression

Logistic regression is ideal when:

- You need **interpretable** predictions (understanding feature importance)
- The decision boundary is **approximately linear**
- You need **calibrated probabilities** (not just class labels)
- You have **limited training data** (fewer parameters = less overfitting)
- You want a **fast, reliable baseline** before trying complex models
- The application requires **explaining predictions** (e.g., healthcare, finance)


Now that you understand logistic regression—one of the most important algorithms in machine learning and the gateway to understanding neural networks. In the next tutorial, we'll explore **Decision Trees**, which take a completely different approach to classification by learning hierarchical rules from data.


## Jupyter Notebook

For hands-on practice, check out the companion notebooks - [Logistic Regression Tutorial](https://drive.google.com/file/d/17t8W2xP3aujSyPFM6Mc0AGFmjVgLUQYv/view?usp=sharing)