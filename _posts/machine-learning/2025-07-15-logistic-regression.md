---
layout: post
title: "Logistic Regression: Binary and Multiclass Classification"
date: 2025-07-15
series: "Machine Learning for Engineers"
series_author: "Mayank Sharma"
excerpt: "Understanding logistic regression for classification problems, from binary to multiclass with sigmoid and softmax functions."
---

Continuing our journey through machine learning, today we turn to **logistic regression**, the workhorse of classification problems. Imagine you're a doctor looking at a patient's medical data — blood pressure, cholesterol level, age, weight — and you need to answer a simple but critical question: **Will this patient develop heart disease?** The answer isn't a number like "$245,000" (as in house price prediction). It's a **yes or no**. Welcome to the world of classification, where logistic regression is the fundamental algorithm you need to master.

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
12. [Using Scikit-Learn](#using-scikit-learn)
13. [Conclusion](#conclusion)
14. [Jupyter Notebook](#jupyter-notebook)

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
1.0             |     0
1.5             |     0
2.0             |     0
3.0             |     0
4.5             |     1
5.0             |     1
6.0             |     1
7.0             |     1
```

If we fit a linear regression line $\hat{y} = \theta_0 + \theta_1 x$, we get predictions that:

1. **Go below 0**: A tumor size of 0.5 cm might yield $\hat{y} = -0.3$. What does a probability of $-0.3$ mean? Nothing — probabilities must be between 0 and 1.

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

1. **Range**: Output is always in $(0, 1)$ — perfect for probabilities
2. **Monotonically increasing**: Larger inputs produce larger outputs
3. **S-shaped curve**: Smooth transition from 0 to 1
4. **Symmetry**: $\sigma(-z) = 1 - \sigma(z)$
5. **At z = 0**: $\sigma(0) = 0.5$ (natural threshold)
6. **As z → +∞**: $\sigma(z) \to 1$
7. **As z → −∞**: $\sigma(z) \to 0$

### Concrete Examples

Let's compute some values to build intuition:

| $z$ | $e^{-z}$ | $\sigma(z) = \frac{1}{1+e^{-z}}$ | Interpretation |
|-----|-----------|-----------------------------------|----------------|
| −5 | 148.41 | $\frac{1}{149.41} = 0.007$ | Very unlikely (class 0) |
| −2 | 7.39 | $\frac{1}{8.39} = 0.119$ | Unlikely |
| −1 | 2.72 | $\frac{1}{3.72} = 0.269$ | Somewhat unlikely |
| 0 | 1.00 | $\frac{1}{2.00} = 0.500$ | Toss-up |
| 1 | 0.37 | $\frac{1}{1.37} = 0.731$ | Somewhat likely |
| 2 | 0.14 | $\frac{1}{1.14} = 0.881$ | Likely |
| 5 | 0.007 | $\frac{1}{1.007} = 0.993$ | Very likely (class 1) |

### Numerical Stability: A Practical Note

When $z$ is very large (e.g., $z = 1000$), computing $e^{-1000}$ underflows to 0 in floating-point arithmetic — that's fine, $\sigma(1000) \approx 1$. But when $z$ is very negative (e.g., $z = -1000$), computing $e^{1000}$ overflows to infinity, giving $\sigma(-1000) = \frac{1}{1 + \infty} =$ `NaN`. To avoid this, implementations clip $z$ to a safe range like $[-500, 500]$ before computing the exponential.

### The Derivative: A Beautiful Result

The derivative of the sigmoid function has a remarkably elegant form:

$$\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$$

**Proof — step by step:**

Starting with $\sigma(z) = (1 + e^{-z})^{-1}$, apply the chain rule:

$$\frac{d\sigma}{dz} = -1 \cdot (1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

Split the fraction:

$$= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z) \cdot \frac{(1 + e^{-z}) - 1}{1 + e^{-z}} = \sigma(z) \cdot \left(1 - \frac{1}{1 + e^{-z}}\right) = \sigma(z)(1 - \sigma(z))$$

This matters because:
- Computing the gradient only requires values we already computed ($\hat{y}$ and $1 - \hat{y}$) — very efficient
- Maximum gradient occurs at $z = 0$ where $\sigma'(0) = 0.5 \times 0.5 = 0.25$
- The gradient approaches zero for very large or very small $z$ — this causes the **vanishing gradient** problem in deep networks

## The Logistic Regression Model

### Putting It Together

Logistic regression combines the linear model from linear regression with the sigmoid function:

$$\hat{y} = \sigma(\theta^T x) = \frac{1}{1 + e^{-(\theta_0 + \theta_1 x_1 + \cdots + \theta_n x_n)}}$$

The output $\hat{y}$ is a number between 0 and 1 representing $P(y = 1 \mid x; \theta)$ — the probability that the input belongs to class 1.

### Making Predictions

Given the probability, we classify using a **threshold** (typically 0.5):

$$\text{predicted class} = \begin{cases} 1 & \text{if } \hat{y} \geq 0.5 \\ 0 & \text{if } \hat{y} < 0.5 \end{cases}$$

Since $\sigma(z) \geq 0.5$ exactly when $z \geq 0$, this is equivalent to:

$$\text{predicted class} = \begin{cases} 1 & \text{if } \theta^T x \geq 0 \\ 0 & \text{if } \theta^T x < 0 \end{cases}$$

**Concrete example**: A spam classifier with $\theta_0 = -3$, $\theta_1 = 0.5$ (word count feature):

- Email with 8 words: $z = -3 + 0.5(8) = 1.0$, $\hat{y} = \sigma(1.0) = 0.73$ → classified as **spam**
- Email with 4 words: $z = -3 + 0.5(4) = -1.0$, $\hat{y} = \sigma(-1.0) = 0.27$ → classified as **not spam**

### Log-Odds Interpretation

We can rearrange the logistic equation to reveal a powerful interpretation. If $p = \sigma(\theta^T x)$ is the probability of class 1, then solving for $\theta^T x$:

$$\theta^T x = \ln\left(\frac{p}{1-p}\right)$$

The left side is the **log-odds** (also called the **logit**). This means logistic regression is a linear model in the log-odds space:

- Each unit increase in $x_j$ changes the log-odds by $\theta_j$
- Equivalently, each unit increase in $x_j$ **multiplies the odds** by $e^{\theta_j}$

**Example**: If $\theta_{\text{cholesterol}} = 0.02$, then each unit increase in cholesterol multiplies the odds of heart disease by $e^{0.02} \approx 1.02$ — a 2% increase in odds.

## Decision Boundary: Drawing the Line

### Linear Decision Boundary

The **decision boundary** is the surface where the model is equally unsure about both classes, where $P(y = 1 \mid x) = 0.5$, or equivalently where $\theta^T x = 0$.

For two features, the decision boundary is a **straight line**:

$$\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0 \implies x_2 = -\frac{\theta_0}{\theta_2} - \frac{\theta_1}{\theta_2} x_1$$

**Numerical Example**: If $\theta = [-3, 1, 1]$:

$$-3 + x_1 + x_2 = 0 \implies x_2 = 3 - x_1$$

This means:
- Points where $x_1 + x_2 > 3$ are classified as class 1
- Points where $x_1 + x_2 < 3$ are classified as class 0
- The line $x_2 = 3 - x_1$ separates the two regions

### Non-Linear Decision Boundaries

By adding **polynomial features**, logistic regression can model non-linear boundaries. For example, with features $x_1, x_2, x_1^2, x_2^2, x_1 x_2$:

$$\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_1^2 + \theta_4 x_2^2 + \theta_5 x_1 x_2 = 0$$

With $\theta = [-1, 0, 0, 1, 1, 0]$:

$$-1 + x_1^2 + x_2^2 = 0 \implies x_1^2 + x_2^2 = 1$$

This is a **circle** of radius 1 centered at the origin — points inside the circle belong to one class, points outside to another. This shows logistic regression can handle non-linear problems if you engineer the right features.

## Cost Function: Cross-Entropy Loss

### Why Not MSE?

For linear regression, we used Mean Squared Error. Why not use it for logistic regression?

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 \quad \text{(MSE)}$$

The problem: when $\hat{y} = \sigma(\theta^T x)$, MSE creates a **non-convex** cost function with many local minima. Gradient descent could get stuck and never find the global minimum.

We need a cost function that is **convex** (a single bowl shape) so gradient descent is guaranteed to find the global minimum.

### Binary Cross-Entropy Loss

Binary cross-entropy is designed for binary classification. For a single training example:

$$\mathcal{L}(\hat{y}, y) = -\left[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})\right]$$

The trick here is that the two terms inside the bracket are never active at the same time:

**When $y = 1$ (positive example):**

The $(1-y)$ term vanishes, leaving $\mathcal{L} = -\log(\hat{y})$

| $\hat{y}$ | $-\log(\hat{y})$ | Verdict |
|-----------|------------------|---------|
| 0.99 | 0.01 | Almost no penalty — correct prediction |
| 0.50 | 0.69 | Moderate penalty — uncertain |
| 0.01 | 4.61 | Huge penalty — very wrong |

**When $y = 0$ (negative example):**

The $y$ term vanishes, leaving $\mathcal{L} = -\log(1 - \hat{y})$

| $\hat{y}$ | $-\log(1 - \hat{y})$ | Verdict |
|-----------|----------------------|---------|
| 0.01 | 0.01 | Almost no penalty — correct prediction |
| 0.50 | 0.69 | Moderate penalty — uncertain |
| 0.99 | 4.61 | Huge penalty — very wrong |

### The Full Cost Function

Over all $m$ training examples:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\right]$$

### Why This is the Right Loss: Maximum Likelihood

The cross-entropy loss isn't arbitrary — it comes from **Maximum Likelihood Estimation (MLE)**. We model the probability of a single data point as:

$$P(y \mid x; \theta) = \hat{y}^{y} \cdot (1 - \hat{y})^{(1-y)}$$

This compact formula captures both cases: when $y=1$ it gives $P = \hat{y}$; when $y=0$ it gives $P = 1 - \hat{y}$.

The **likelihood** over all independent training examples is:

$$L(\theta) = \prod_{i=1}^{m} (\hat{y}^{(i)})^{y^{(i)}} (1 - \hat{y}^{(i)})^{(1-y^{(i)})}$$

Taking the log (products become sums) and negating (turn maximization into minimization):

$$-\log L(\theta) = -\sum_{i=1}^{m} \left[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\right]$$

Dividing by $m$ gives exactly our cost function $J(\theta)$. **Minimizing cross-entropy loss = maximizing the probability of observing the training data.**

## Gradient Descent for Logistic Regression

### Full Derivation of the Gradient

To apply gradient descent, we need $\frac{\partial J}{\partial \theta_j}$. For a single training example with $z = \theta^T x$ and $\hat{y} = \sigma(z)$, apply the chain rule in three steps:

$$\frac{\partial \mathcal{L}}{\partial \theta_j} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial \theta_j}$$

**Step 1** — derivative of the loss with respect to $\hat{y}$:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}$$

**Step 2** — derivative of sigmoid (which we already derived):

$$\frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y})$$

**Step 3** — derivative of the linear part:

$$\frac{\partial z}{\partial \theta_j} = x_j$$

**Multiplying them together** (the $\hat{y}(1-\hat{y})$ in step 2 cancels with the denominators from step 1):

$$\frac{\partial \mathcal{L}}{\partial \theta_j} = \left(\frac{-y(1-\hat{y}) + (1-y)\hat{y}}{\hat{y}(1-\hat{y})}\right) \cdot \hat{y}(1-\hat{y}) \cdot x_j = (-y + \hat{y}) \cdot x_j = (\hat{y} - y) \cdot x_j$$

**Averaging over all $m$ examples:**

$$\boxed{\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}}$$

### A Surprising Result

This gradient formula is **identical in form to the one for linear regression**. The only difference is that $\hat{y} = \sigma(\theta^T x)$ here instead of $\hat{y} = \theta^T x$. This happens because the sigmoid derivative and the cross-entropy loss were specifically designed to pair together and simplify cleanly.

### The Vectorized Update Rule

$$\theta := \theta - \frac{\alpha}{m} X^T (\hat{y} - y)$$

Where $X$ is the $(m \times (n+1))$ design matrix (with bias column), $\hat{y}$ is the vector of predicted probabilities, and $y$ is the vector of true labels.

### Numerical Walkthrough: Two Iterations by Hand

Let's trace gradient descent on a tiny example to see exactly what happens:

```
Data: 2 examples, 1 feature
  x₁ = 1, y₁ = 0  (class 0)
  x₂ = 3, y₂ = 1  (class 1)

Start: θ₀ = 0, θ₁ = 0,  α = 0.5
```

**Iteration 1:**

Linear scores: $z_1 = 0 + 0(1) = 0$, $z_2 = 0 + 0(3) = 0$

Predictions: $\hat{y}_1 = \sigma(0) = 0.5$, $\hat{y}_2 = \sigma(0) = 0.5$

Errors: $\hat{y}_1 - y_1 = 0.5 - 0 = 0.5$, $\hat{y}_2 - y_2 = 0.5 - 1 = -0.5$

Cost: $J = -\frac{1}{2}[(0)\log(0.5) + (1)\log(0.5) + (1)\log(0.5) + (0)\log(0.5)] = \log 2 \approx 0.693$

Gradients:
$$\frac{\partial J}{\partial \theta_0} = \frac{1}{2}[(0.5)(1) + (-0.5)(1)] = 0$$
$$\frac{\partial J}{\partial \theta_1} = \frac{1}{2}[(0.5)(1) + (-0.5)(3)] = \frac{1}{2}[0.5 - 1.5] = -0.5$$

Update:
$$\theta_0 = 0 - 0.5(0) = 0$$
$$\theta_1 = 0 - 0.5(-0.5) = 0.25$$

**Iteration 2:**

Linear scores: $z_1 = 0 + 0.25(1) = 0.25$, $z_2 = 0 + 0.25(3) = 0.75$

Predictions: $\hat{y}_1 = \sigma(0.25) \approx 0.562$, $\hat{y}_2 = \sigma(0.75) \approx 0.679$

Now $\hat{y}_2 > \hat{y}_1$ — the model has already started learning that $x=3$ is more likely to be class 1 than $x=1$. After hundreds of iterations, $\theta_1$ will grow large enough that $\hat{y}_1 \approx 0$ and $\hat{y}_2 \approx 1$.

### Convergence

Just like linear regression, you can stop gradient descent early when the cost barely changes:

$$|\, J^{(t)} - J^{(t-1)}\, | < \epsilon \quad \text{(e.g., } \epsilon = 10^{-6}\text{)}$$

Watch the cost history: it should decrease smoothly. If it oscillates or rises, your learning rate is too large.

## Multiclass Classification

So far we've discussed **binary** classification (two classes). What about problems with three or more classes?

### One-vs-Rest (OvR) Strategy

Also called **One-vs-All (OvA)**, this trains $K$ separate binary classifiers for $K$ classes.

**How it works**:
1. For each class $k = 1, 2, \ldots, K$, treat class $k$ as positive and all others as negative, and train a binary logistic regression $h_\theta^{(k)}(x)$
2. At prediction time, run all $K$ classifiers and pick the highest probability:

$$\hat{y} = \arg\max_k \; h_\theta^{(k)}(x)$$

**Example with 3 classes (Cat, Dog, Bird)**:
- Classifier 1: Cat vs. rest → $P(\text{Cat} \mid x) = 0.7$
- Classifier 2: Dog vs. rest → $P(\text{Dog} \mid x) = 0.2$
- Classifier 3: Bird vs. rest → $P(\text{Bird} \mid x) = 0.1$
- **Prediction**: Cat (highest)

**Limitation**: The three probabilities (0.7, 0.2, 0.1) come from separate classifiers and don't sum to 1 — they are not a valid probability distribution.

### Softmax Regression (Multinomial Logistic Regression)

A more principled approach for multiclass classification is **softmax regression**, which directly generalizes logistic regression to multiple classes.

#### The Softmax Function

For $K$ classes, we compute a score $z_k$ for each class:

$$z_k = \theta_k^T x \quad \text{for } k = 1, 2, \ldots, K$$

The softmax function converts these scores into probabilities:

$$P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**Properties of Softmax**:
1. All outputs are in $(0, 1)$
2. All outputs **sum to 1** — a proper probability distribution
3. Larger scores get exponentially larger probabilities
4. When $K = 2$, softmax reduces to the sigmoid function

#### Numerical Stability of Softmax

There's a subtle numerical problem: if any $z_k$ is very large (e.g., 1000), $e^{1000}$ overflows to infinity. The fix is to subtract the maximum score before exponentiating. This doesn't change the result mathematically because the constant cancels in the fraction:

$$\frac{e^{z_k}}{\sum_j e^{z_j}} = \frac{e^{z_k - c}}{\sum_j e^{z_j - c}} \quad \text{for any constant } c$$

We choose $c = \max_j z_j$, making all exponents $\leq 0$ and thus safe to compute.

#### Numerical Example

Scores $z = [2.0, 1.0, 0.1]$:

$$e^{z} = [e^{2.0}, e^{1.0}, e^{0.1}] = [7.389, 2.718, 1.105], \quad \text{sum} = 11.212$$

$$P = \left[\frac{7.389}{11.212}, \frac{2.718}{11.212}, \frac{1.105}{11.212}\right] = [0.659, 0.242, 0.099]$$

The model predicts class 1 with 65.9% confidence, class 2 with 24.2%, class 3 with 9.9%.

#### One-Hot Encoding

To compute the loss for multiclass problems, we need to compare predicted probabilities ($K$ numbers per example) against the true label. But the true label is just a single integer like 2. We convert it to a **one-hot vector** — a vector of zeros with a single 1 in the position of the true class:

```
True class = 1  →  one-hot = [0, 1, 0]
True class = 2  →  one-hot = [0, 0, 1]
True class = 0  →  one-hot = [1, 0, 0]
```

This lets us subtract predictions from labels element-wise, which makes the math and code uniform.

#### Cross-Entropy Loss for Multiclass

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{y}_k^{(i)})$$

Where $y_k^{(i)}$ is 1 if example $i$ belongs to class $k$ (one-hot), 0 otherwise.

Since only one term $y_k^{(i)} = 1$ per example, this simplifies to:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \log(\hat{y}_{c_i}^{(i)})$$

Where $c_i$ is the true class of example $i$ — we only look at the predicted probability for the correct class.

#### Gradient for Softmax

The gradient of the cross-entropy loss with respect to the score $z_k$ is:

$$\frac{\partial J}{\partial z_k} = \hat{y}_k - y_k$$

This elegant result mirrors the binary case exactly.

### OvR vs. Softmax: When to Use Which

| Aspect | One-vs-Rest | Softmax |
|--------|-------------|---------|
| Classes are mutually exclusive | Not required | Required |
| Probability calibration | Poor (don't sum to 1) | Good (sum to 1) |
| Training | K separate models | One joint model |
| Use case | Multi-label problems | Single-label classification |

## Regularization in Logistic Regression

### The Overfitting Problem

With many features (especially polynomial features), logistic regression can overfit — learning the training data's noise rather than the underlying pattern. The model performs well on training data but fails on new examples.

### L2 Regularization (Ridge)

Add a penalty term that discourages large parameter values:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\right] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

Where:
- $\lambda$ is the **regularization parameter** (controls the trade-off between fit and complexity)
- The sum starts at $j = 1$ — we do **not** regularize $\theta_0$ (the bias), because the bias doesn't control complexity

**Effect**: Shrinks all coefficients toward zero but doesn't eliminate any. Produces smoother decision boundaries.

The gradient with regularization:

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} \theta_j \quad \text{for } j \geq 1$$

### L1 Regularization (Lasso)

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\right] + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|$$

**Effect**: Can drive some coefficients to exactly zero, performing automatic feature selection. Useful when you suspect many features are irrelevant.

### Elastic Net (L1 + L2)

Combines both penalties, with $\alpha \in [0, 1]$ balancing L1 and L2:

$$J(\theta) = \text{cross-entropy} + \frac{\lambda}{m} \left[\frac{(1 - \alpha)}{2} \sum_{j=1}^{n} \theta_j^2 + \alpha \sum_{j=1}^{n} |\theta_j|\right]$$

### Choosing $\lambda$ in Practice

- **$\lambda$ too small**: Minimal regularization → possible overfitting
- **$\lambda$ too large**: Heavy regularization → underfitting (model too simple, high training error)

The standard approach is **k-fold cross-validation**: split the training set into $k$ folds, train on $k-1$ folds and validate on the remaining one, repeat $k$ times, and pick the $\lambda$ with the best average validation score. In scikit-learn, `LogisticRegressionCV` does this automatically.

## Evaluation Metrics for Classification

Unlike regression where MSE and R² work well, classification needs different metrics. Accuracy alone is dangerously misleading.

### Confusion Matrix

For binary classification, predictions fall into four categories:

|  | **Predicted Positive** | **Predicted Negative** |
|--|------------------------|------------------------|
| **Actually Positive** | True Positive (TP) | False Negative (FN) — "missed it" |
| **Actually Negative** | False Positive (FP) — "false alarm" | True Negative (TN) |

**Concrete example** — a cancer screening test on 100 patients (10 actually have cancer):

```
Predicted cancer | Predicted healthy
----------------------------------
TP = 8           | FN = 2    (actually had cancer)
FP = 5           | TN = 85   (actually healthy)
```

### Key Metrics (with Formulas and Intuition)

**Accuracy** — what fraction of all predictions were correct?

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{8 + 85}{100} = 93\%$$

Sounds great, but misleading — 90% accuracy is achievable by just predicting "no cancer" for everyone, yet that model would miss every cancer patient.

**Precision** — of the patients we flagged as positive, how many actually were?

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{8}{8 + 5} = 61.5\%$$

High precision = few false alarms. Important when a false positive is costly (e.g., unnecessary surgery).

**Recall / Sensitivity** — of the actual positives, how many did we catch?

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{8}{8 + 2} = 80\%$$

High recall = few missed cases. Critical in cancer screening — missing a case (FN) is far worse than a false alarm (FP).

**F1 Score** — harmonic mean of precision and recall, useful when you need a single number that balances both:

$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \cdot \frac{0.615 \times 0.80}{0.615 + 0.80} = 0.696$$

### Precision-Recall Trade-off and Threshold Selection

The default threshold of 0.5 is not always optimal. By adjusting it, you trade precision against recall:

| Threshold | Behavior | Use when |
|-----------|----------|----------|
| Lower (e.g., 0.3) | More positives predicted → Higher recall, lower precision | Cancer screening — don't miss any cases |
| Default (0.5) | Balanced starting point | General classification |
| Higher (e.g., 0.7) | Fewer positives predicted → Higher precision, lower recall | Spam filter — don't block legitimate emails |

**How to pick the right threshold**: Plot the **Precision-Recall curve** (or the ROC curve) across all thresholds and choose the threshold that gives the best trade-off for your specific application.

### ROC Curve and AUC

The **Receiver Operating Characteristic (ROC)** curve plots the True Positive Rate (Recall) against the False Positive Rate at every possible threshold:

$$\text{TPR} = \frac{TP}{TP + FN}, \quad \text{FPR} = \frac{FP}{FP + TN}$$

The **Area Under the Curve (AUC)** summarizes the ROC curve in a single number:
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: Random guessing (diagonal line)
- **AUC < 0.5**: Worse than random — invert predictions

AUC is useful because it evaluates the model independently of the threshold, telling you how well the model ranks positives above negatives overall.

## Implementation from Scratch

### Binary Logistic Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, RocCurveDisplay)


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000,
                 reg_lambda=0.0, tol=1e-6):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        self.tol = tol
        self.theta = None
        self.cost_history = []

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)  # prevent overflow
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]   # prepend bias column
        y = y.reshape(-1, 1)
        self.theta = np.zeros((n + 1, 1))

        for iteration in range(self.n_iterations):
            predictions = self.sigmoid(X_b.dot(self.theta))

            eps = 1e-15  # prevent log(0)
            cost = -(1 / m) * np.sum(
                y * np.log(predictions + eps) +
                (1 - y) * np.log(1 - predictions + eps)
            )

            if self.reg_lambda > 0:
                cost += (self.reg_lambda / (2 * m)) * np.sum(self.theta[1:] ** 2)

            self.cost_history.append(cost)

            # Early stopping if cost barely changes
            if iteration > 0 and abs(self.cost_history[-2] - cost) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

            errors = predictions - y
            gradients = (1 / m) * X_b.T.dot(errors)

            if self.reg_lambda > 0:
                reg_term = (self.reg_lambda / m) * self.theta
                reg_term[0] = 0   # don't regularize bias
                gradients += reg_term

            self.theta -= self.learning_rate * gradients

    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X_b.dot(self.theta)).flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ── End-to-End Example ──────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # 1. Generate synthetic binary classification data
    m = 300
    X_class0 = np.random.randn(m // 2, 2) + np.array([1, 1])
    X_class1 = np.random.randn(m // 2, 2) + np.array([4, 4])
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (m // 2) + [1] * (m // 2))

    # 2. Split into train and test sets BEFORE scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 3. Scale features using training statistics only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 4. Train the model
    model = LogisticRegressionScratch(learning_rate=0.5, n_iterations=500)
    model.fit(X_train_s, y_train)

    # 5. Evaluate on the test set
    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Class 0", "Class 1"]))

    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

    # 6. Plot cost history and ROC curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(model.cost_history)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Cost J(θ)")
    ax1.set_title("Cost Function Convergence")
    ax1.grid(True)

    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax2)
    ax2.set_title("ROC Curve")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
```

### Multiclass with Softmax

```python
class SoftmaxRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.theta = None
        self.cost_history = []

    @staticmethod
    def softmax(z):
        # Subtract max per row for numerical stability (prevents e^1000 overflow)
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        m, n = X.shape
        K = len(np.unique(y))
        X_b = np.c_[np.ones((m, 1)), X]

        # One-hot encode labels: integer class → vector of 0s with a 1 at class index
        Y_onehot = np.zeros((m, K))
        Y_onehot[np.arange(m), y] = 1

        self.theta = np.zeros((n + 1, K))  # shape: (features+1) × num_classes

        for iteration in range(self.n_iterations):
            probs = self.softmax(X_b.dot(self.theta))   # shape: (m, K)

            eps = 1e-15
            cost = -(1 / m) * np.sum(Y_onehot * np.log(probs + eps))
            self.cost_history.append(cost)

            if iteration > 0 and abs(self.cost_history[-2] - cost) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

            # Gradient: (probs - Y_onehot) has shape (m, K); X_b.T has shape (n+1, m)
            gradients = (1 / m) * X_b.T.dot(probs - Y_onehot)
            self.theta -= self.learning_rate * gradients

    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self.softmax(X_b.dot(self.theta))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)
```

## Using Scikit-Learn

For production use, scikit-learn provides optimized implementations with built-in cross-validation for regularization tuning. Here is a complete, self-contained example:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score)
import matplotlib.pyplot as plt

# 1. Create a synthetic dataset
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_redundant=2, random_state=42
)

# 2. Split — always before scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale — fit on train, transform both
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 4a. Basic logistic regression with L2 regularization
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
#   C = 1/λ: larger C = less regularization; smaller C = more regularization
model.fit(X_train_s, y_train)

# 4b. Auto-tune regularization with cross-validation
model_cv = LogisticRegressionCV(
    Cs=10, cv=5, max_iter=1000, random_state=42
)
model_cv.fit(X_train_s, y_train)
print(f"Best C found by CV: {model_cv.C_[0]:.4f}")

# 5. Evaluate on test set
y_pred  = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

# 6. Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"]).plot()
plt.title("Confusion Matrix")
plt.show()

# 7. Multiclass example (iris dataset)
from sklearn.datasets import load_iris

X_iris, y_iris = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_iris,
                                            test_size=0.2, random_state=42)
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s  = sc.transform(X_te)

# multi_class='multinomial' uses softmax; solver='lbfgs' handles it efficiently
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                         max_iter=1000, random_state=42)
clf.fit(X_tr_s, y_tr)
print("\nIris Multiclass Report:")
print(classification_report(y_te, clf.predict(X_te_s),
                            target_names=load_iris().target_names))
```

**Note on `C` vs. `λ`**: scikit-learn uses `C = 1/λ` as the regularization parameter. A smaller `C` means stronger regularization (more penalty on large coefficients). This is the inverse of what you'd expect if you're used to thinking of $\lambda$ directly.

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

Now that you understand logistic regression — one of the most important algorithms in machine learning and the gateway to understanding neural networks. In the next tutorial, we'll explore **Decision Trees**, which take a completely different approach to classification by learning hierarchical rules from data.

## Jupyter Notebook

For hands-on practice, check out the companion notebooks - [Logistic Regression Tutorial](https://drive.google.com/file/d/17t8W2xP3aujSyPFM6Mc0AGFmjVgLUQYv/view?usp=sharing)
