---
layout: post
title: "Naive Bayes: Probabilistic Classification"
date: 2025-12-13
series: "Machine Learning Series"
series_author: "Mayank Sharma"
excerpt: "Understand Naive Bayes classification using Bayes theorem, conditional independence, and variants for different data types."
---

Continuing in our journey through machine learning, today we turn to **Naive Bayes**, a simple yet powerful probabilistic classifier. Imagine you wake up feeling feverish and congested. A doctor, before running any tests, immediately starts updating their mental model: "Given this patient has a fever, what is the probability they have influenza versus a common cold versus COVID-19?" Without consciously realizing it, the doctor is applying Bayesian reasoning, combining prior knowledge about disease prevalence with the observed evidence to form a diagnosis. This is the essence of Naive Bayes: a family of probabilistic classifiers built on elegant probability theory that is both mathematically rigorous and surprisingly practical.

## Table of Contents

1. [Introduction: Thinking Probabilistically](#introduction-thinking-probabilistically)
2. [Probability Fundamentals](#probability-fundamentals)
3. [Bayes' Theorem: The Core of Everything](#bayes-theorem-the-core-of-everything)
4. [From Bayes' Theorem to a Classifier](#from-bayes-theorem-to-a-classifier)
5. [The Naive Conditional Independence Assumption](#the-naive-conditional-independence-assumption)
6. [Gaussian Naive Bayes: Continuous Features](#gaussian-naive-bayes-continuous-features)
7. [Multinomial Naive Bayes: Count Features](#multinomial-naive-bayes-count-features)
8. [Bernoulli Naive Bayes: Binary Features](#bernoulli-naive-bayes-binary-features)
9. [Laplace Smoothing: Handling Zero Probabilities](#laplace-smoothing-handling-zero-probabilities)
10. [Log Probabilities: Numerical Stability](#log-probabilities-numerical-stability)
11. [Implementation from Scratch](#implementation-from-scratch)
12. [Advantages and Limitations](#advantages-and-limitations)
13. [Conclusion](#conclusion)

## Introduction: Thinking Probabilistically

### What is Naive Bayes?

Naive Bayes is a family of simple yet powerful probabilistic classifiers based on applying **Bayes' theorem** with a strong (naive) assumption of **conditional independence** between features given the class label.

The name comes from two parts:
- **Bayes**: The algorithm is based on Bayes' theorem, a fundamental rule of probability
- **Naive**: It makes the simplifying (and often unrealistic) assumption that all features are independent of each other given the class

Despite this seemingly unrealistic assumption, Naive Bayes works remarkably well in practice across many domains and remains one of the most widely used classifiers.

## Probability Fundamentals

Before diving into Naive Bayes, we need a solid grounding in probability theory.

### Basic Probability

For an event $A$, the probability $P(A)$ satisfies:
- $0 \leq P(A) \leq 1$
- $P(A) + P(\neg A) = 1$ (complement rule)

### Joint Probability

The probability that both $A$ and $B$ occur:

$$P(A \cap B) = P(A, B)$$

### Conditional Probability

The probability of $A$ **given** that $B$ has occurred:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Intuition**: We restrict our universe to cases where $B$ is true, then ask how often $A$ also occurs within that restricted universe.

**Example**: In a class of 100 students, 60 passed. Of those 60 who passed, 40 studied hard. The probability a student studied hard *given* they passed is:

$$P(\text{studied} \mid \text{passed}) = \frac{40}{60} = \frac{2}{3}$$

### Product Rule

Rearranging the conditional probability definition:

$$P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

### Marginalisation (Law of Total Probability)

If $\{B_1, B_2, ..., B_k\}$ partition the sample space (mutually exclusive and exhaustive):

$$P(A) = \sum_{i=1}^{k} P(A \mid B_i) \cdot P(B_i)$$

**Example**: Two factories produce widgets. Factory 1 produces 70% of all widgets and has a 2% defect rate. Factory 2 produces 30% and has a 5% defect rate. Probability of a defective widget:

$$P(\text{defect}) = P(\text{defect} \mid F_1) \cdot P(F_1) + P(\text{defect} \mid F_2) \cdot P(F_2)$$
$$= 0.02 \times 0.70 + 0.05 \times 0.30 = 0.014 + 0.015 = 0.029$$

### Statistical Independence

Two events $A$ and $B$ are **independent** if:

$$P(A \cap B) = P(A) \cdot P(B) \quad \Longleftrightarrow \quad P(A \mid B) = P(A)$$

Knowing $B$ occurred gives no information about $A$.

## Bayes' Theorem: The Core of Everything

### Derivation

Starting from the product rule applied in two ways:

$$P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

Dividing both sides by $P(B)$:

$$\boxed{P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}}$$

This is **Bayes' theorem**. Simple to derive, profound in its implications.

### The Four Terms

In the classification context, let $C$ be the class and $\mathbf{x}$ be the observed features:

$$P(C \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid C) \cdot P(C)}{P(\mathbf{x})}$$

Each term has a name and meaning:

| Term | Name | Meaning |
|------|------|---------|
| $P(C \mid \mathbf{x})$ | **Posterior** | Probability of class $C$ after seeing features $\mathbf{x}$ |
| $P(\mathbf{x} \mid C)$ | **Likelihood** | How probable are these features under class $C$? |
| $P(C)$ | **Prior** | Baseline probability of class $C$ before seeing any features |
| $P(\mathbf{x})$ | **Evidence** | Probability of observing these features (normalisation constant) |

**The key insight**: Bayes' theorem tells us how to update our prior belief $P(C)$ after observing evidence $\mathbf{x}$ to get the posterior $P(C \mid \mathbf{x})$.

### Worked Example: Medical Diagnosis

**Setup**:
- Disease prevalence: $P(\text{disease}) = 0.01$ (1% of population)
- Test sensitivity: $P(\text{positive} \mid \text{disease}) = 0.99$
- Test specificity: $P(\text{negative} \mid \text{no disease}) = 0.95$, so $P(\text{positive} \mid \text{no disease}) = 0.05$

**Question**: If a patient tests positive, what is the probability they have the disease?

**Step 1**: Prior is $P(\text{disease}) = 0.01$, $P(\text{no disease}) = 0.99$

**Step 2**: Compute the evidence using total probability:

$$P(\text{positive}) = P(\text{positive} \mid \text{disease}) \cdot P(\text{disease}) + P(\text{positive} \mid \text{no disease}) \cdot P(\text{no disease})$$
$$= 0.99 \times 0.01 + 0.05 \times 0.99 = 0.0099 + 0.0495 = 0.0594$$

**Step 3**: Apply Bayes' theorem:

$$P(\text{disease} \mid \text{positive}) = \frac{0.99 \times 0.01}{0.0594} = \frac{0.0099}{0.0594} \approx 0.167$$

**Surprising result**: Even with a 99% accurate test, a positive result only means a ~17% chance of disease because the disease is rare! This is the power of Bayesian reasoning.

## From Bayes' Theorem to a Classifier

### The Decision Rule

For classification with $K$ classes $\{C_1, C_2, ..., C_K\}$ and feature vector $\mathbf{x} = (x_1, x_2, ..., x_d)$, we want the class with the highest posterior probability:

$$\hat{C} = \arg\max_{C_k} P(C_k \mid \mathbf{x})$$

Applying Bayes' theorem:

$$\hat{C} = \arg\max_{C_k} \frac{P(\mathbf{x} \mid C_k) \cdot P(C_k)}{P(\mathbf{x})}$$

Since $P(\mathbf{x})$ is the same for all classes, we can drop it (it doesn't affect the argmax):

$$\hat{C} = \arg\max_{C_k} P(\mathbf{x} \mid C_k) \cdot P(C_k)$$

This is the **Maximum A Posteriori (MAP) decision rule**.

### The Challenge: Computing $P(\mathbf{x} \mid C_k)$

The likelihood $P(\mathbf{x} \mid C_k) = P(x_1, x_2, ..., x_d \mid C_k)$ is a **joint probability over all $d$ features**.

For discrete features with $v$ possible values each, this requires estimating $v^d$ parameters — exponential in $d$! With $d = 100$ binary features, we need $2^{100} \approx 10^{30}$ parameters. Completely infeasible.

This is where the "naive" assumption comes in.

## The Naive Conditional Independence Assumption

### The Assumption

Naive Bayes assumes that all features $x_1, x_2, ..., x_d$ are **conditionally independent** given the class $C_k$:

$$P(x_1, x_2, ..., x_d \mid C_k) = \prod_{j=1}^{d} P(x_j \mid C_k)$$

This factorises the joint likelihood into a product of per-feature likelihoods.

### Why It's "Naive"

In reality, features are rarely independent. Consider email spam classification:
- The words "Nigerian" and "prince" often appear together in spam
- Knowing the email contains "Nigerian" makes it much more likely "prince" also appears
- They are **correlated**, not independent

Despite violating this assumption, Naive Bayes still produces accurate classifications because the **ranking** of class probabilities is often preserved even when exact probabilities are wrong.

### The Full Naive Bayes Formula

Combining the MAP rule with the independence assumption:

$$\hat{C} = \arg\max_{C_k} P(C_k) \prod_{j=1}^{d} P(x_j \mid C_k)$$

This requires only $K \times d$ parameters for the likelihoods (plus $K$ priors) — linear in both classes and features!

**Why this works**:
- With $K = 2$ classes, $d = 10{,}000$ words, we need $2 \times 10{,}000 = 20{,}000$ parameters
- Without the assumption: $2^{10{,}000}$ parameters — impossible

### Estimating the Parameters

**Prior** $P(C_k)$: Fraction of training examples with class $C_k$:

$$\hat{P}(C_k) = \frac{N_k}{N}$$

Where $N_k$ = number of examples of class $k$, $N$ = total training examples.

**Likelihood** $P(x_j \mid C_k)$: Depends on the type of feature (continuous, count, binary). Each variant of Naive Bayes handles this differently.

## Gaussian Naive Bayes: Continuous Features

### When to Use

Gaussian Naive Bayes (GNB) is used when features are **continuous**, such as height, weight, temperature, or sensor readings.

### The Gaussian Likelihood

For continuous feature $j$, we model $P(x_j \mid C_k)$ as a Gaussian (normal) distribution:

$$P(x_j \mid C_k) = \frac{1}{\sqrt{2\pi \sigma_{jk}^2}} \exp\!\left(-\frac{(x_j - \mu_{jk})^2}{2\sigma_{jk}^2}\right)$$

Where:
- $\mu_{jk}$ is the **mean** of feature $j$ within class $k$
- $\sigma_{jk}^2$ is the **variance** of feature $j$ within class $k$

### Parameter Estimation (MLE)

From the training data, for each class $k$ and feature $j$:

$$\hat{\mu}_{jk} = \frac{1}{N_k} \sum_{i: y^{(i)} = k} x_j^{(i)}$$

$$\hat{\sigma}_{jk}^2 = \frac{1}{N_k} \sum_{i: y^{(i)} = k} \left(x_j^{(i)} - \hat{\mu}_{jk}\right)^2$$

These are simply the **class-conditional mean and variance** for each feature — one pass over the training data.

### Prediction

For a new point $\mathbf{x}$:

$$\hat{C} = \arg\max_{C_k} P(C_k) \prod_{j=1}^{d} \frac{1}{\sqrt{2\pi \hat{\sigma}_{jk}^2}} \exp\!\left(-\frac{(x_j - \hat{\mu}_{jk})^2}{2\hat{\sigma}_{jk}^2}\right)$$

### Concrete Example

**Task**: Classify an animal as Dog or Cat based on Weight (kg) and Height (cm).

**Training data** (class statistics):

| Class | Weight μ | Weight σ | Height μ | Height σ |
|-------|----------|----------|----------|----------|
| Dog   | 20       | 5        | 55       | 8        |
| Cat   | 4        | 1        | 28       | 4        |

**Priors**: $P(\text{Dog}) = 0.6$, $P(\text{Cat}) = 0.4$

**New animal**: Weight = 18 kg, Height = 50 cm

**Dog likelihood**:
$$P(\text{Weight}=18 \mid \text{Dog}) = \frac{1}{\sqrt{2\pi \cdot 25}} \exp\!\left(-\frac{(18-20)^2}{2 \cdot 25}\right) = 0.0736$$

$$P(\text{Height}=50 \mid \text{Dog}) = \frac{1}{\sqrt{2\pi \cdot 64}} \exp\!\left(-\frac{(50-55)^2}{2 \cdot 64}\right) = 0.0439$$

$$\text{Score(Dog)} = 0.6 \times 0.0736 \times 0.0439 = 0.00194$$

**Cat likelihood**:
$$P(\text{Weight}=18 \mid \text{Cat}) = \frac{1}{\sqrt{2\pi \cdot 1}} \exp\!\left(-\frac{(18-4)^2}{2 \cdot 1}\right) \approx 0$$

**Prediction**: Dog (the weight alone rules out Cat).

## Multinomial Naive Bayes: Count Features

### When to Use

Multinomial Naive Bayes (MNB) is used when features represent **counts**, the most natural choice for **text classification** where features are word frequencies.

### The Bag-of-Words Model

In text classification, we represent each document as a vector of word counts. Given a vocabulary $\mathcal{V} = \{w_1, w_2, ..., w_V\}$, a document is represented as:

$$\mathbf{x} = (x_1, x_2, ..., x_V)$$

where $x_j$ = number of times word $w_j$ appears.

### The Multinomial Likelihood

The likelihood of observing document $\mathbf{x}$ given class $C_k$ follows a **multinomial distribution**:

$$P(\mathbf{x} \mid C_k) = \frac{(\sum_j x_j)!}{\prod_j x_j!} \prod_{j=1}^{V} \theta_{jk}^{x_j}$$

Where $\theta_{jk} = P(w_j \mid C_k)$ is the probability of word $w_j$ appearing in a document of class $k$.

Since the multinomial coefficient $\frac{(\sum_j x_j)!}{\prod_j x_j!}$ is constant across classes (same document length), we only care about:

$$P(\mathbf{x} \mid C_k) \propto \prod_{j=1}^{V} \theta_{jk}^{x_j}$$

### Parameter Estimation (MLE)

$$\hat{\theta}_{jk} = \frac{\text{count of word } j \text{ in class } k \text{ documents}}{\text{total word count in class } k \text{ documents}} = \frac{N_{jk}}{N_k^{\text{words}}}$$

Where:
- $N_{jk} = \sum_{i: y^{(i)}=k} x_j^{(i)}$ (total occurrences of word $j$ in class $k$)
- $N_k^{\text{words}} = \sum_j N_{jk}$ (total words in class $k$)

### Concrete Example: Spam Classification

**Training corpus** (simplified, 3 keywords):

| Document | "free" | "money" | "hello" | Class |
|----------|--------|---------|---------|-------|
| Email 1  | 3      | 2       | 0       | Spam  |
| Email 2  | 2      | 1       | 0       | Spam  |
| Email 3  | 0      | 0       | 4       | Ham   |
| Email 4  | 0      | 0       | 3       | Ham   |

**Priors**: $P(\text{Spam}) = 0.5$, $P(\text{Ham}) = 0.5$

**Word probabilities**:

For Spam (total = 3+2+2+1 = 8 words):
- $\hat{\theta}_{\text{free}, \text{Spam}} = (3+2)/8 = 5/8 = 0.625$
- $\hat{\theta}_{\text{money}, \text{Spam}} = (2+1)/8 = 3/8 = 0.375$
- $\hat{\theta}_{\text{hello}, \text{Spam}} = 0/8 = 0$ ← **Problem!**

The zero probability for "hello" in spam will zero-out the entire product. This is solved by Laplace smoothing (covered below).

## Bernoulli Naive Bayes: Binary Features

### When to Use

Bernoulli Naive Bayes (BNB) is used when features are **binary** (0 or 1). In text classification, this corresponds to whether a word is **present or absent** in a document (regardless of how many times it appears).

### The Bernoulli Likelihood

For binary feature $x_j \in \{0, 1\}$:

$$P(x_j \mid C_k) = \theta_{jk}^{x_j} \cdot (1 - \theta_{jk})^{1-x_j}$$

Where $\theta_{jk} = P(x_j = 1 \mid C_k)$ is the probability that feature $j$ is present in class $k$.

**Key difference from Multinomial**: BNB explicitly models the **absence** of features. If word $j$ does not appear ($x_j = 0$), Multinomial ignores it, but Bernoulli includes the factor $(1 - \theta_{jk})$.

### Parameter Estimation

$$\hat{\theta}_{jk} = \frac{\text{number of class-}k\text{ documents containing feature }j}{\text{number of class-}k\text{ documents}} = \frac{D_{jk}}{D_k}$$

Where $D_{jk}$ = number of class-$k$ documents where feature $j$ appears, $D_k$ = total class-$k$ documents.

### Multinomial vs Bernoulli: Key Differences

| Aspect | Multinomial NB | Bernoulli NB |
|--------|---------------|--------------|
| Feature representation | Word counts | Word presence/absence |
| Missing words | Ignored | Modelled explicitly |
| Document length | Sensitive (longer = more evidence) | Insensitive |
| Best for | Long documents | Short documents |
| Typical use | News categorisation | Sentiment (short reviews) |

## Laplace Smoothing: Handling Zero Probabilities

### The Zero Frequency Problem

If a feature value never appears with a given class in training, its estimated probability is zero:

$$\hat{P}(x_j = v \mid C_k) = 0$$

This zeroes out the **entire product** in the likelihood, regardless of all other evidence — an extreme and incorrect conclusion.

**Example**: If the word "exquisite" never appears in spam emails during training, then any email containing "exquisite" will get a spam probability of exactly zero, even if it also contains "free money lottery prize" fifteen times.

### Laplace (Add-One) Smoothing

Add a pseudo-count $\alpha$ (often $\alpha = 1$) to every possible feature value before computing probabilities:

**For Multinomial NB**:

$$\hat{\theta}_{jk} = \frac{N_{jk} + \alpha}{N_k^{\text{words}} + \alpha \cdot V}$$

Where $V$ = vocabulary size. The denominator ensures probabilities still sum to 1.

**For Bernoulli NB**:

$$\hat{\theta}_{jk} = \frac{D_{jk} + \alpha}{D_k + 2\alpha}$$

The denominator adds $\alpha$ for each of the two outcomes (present/absent).

**For Gaussian NB**: Smoothing is usually not required since the Gaussian density is always positive. Instead, a small variance floor is used.

### Effect of Smoothing Parameter $\alpha$

$$\hat{\theta}_{jk} = \frac{N_{jk} + \alpha}{N_k^{\text{words}} + \alpha \cdot V}$$

- **$\alpha = 0$**: No smoothing (MLE estimate, zero probabilities possible)
- **$\alpha = 1$**: Laplace smoothing (uniform pseudo-counts)
- **$\alpha \to \infty$**: Uniform distribution over all words (ignores data)
- **$\alpha < 1$**: Lidstone smoothing (weaker than Laplace)

### Bayesian Interpretation

Laplace smoothing is equivalent to using a **Dirichlet prior** with parameter $\alpha$ on the word probabilities. Instead of pure maximum likelihood estimation, we compute the **Maximum A Posteriori (MAP)** estimate. The prior encodes our belief that all words are equally likely before seeing any data.

### Re-visiting the Spam Example with Smoothing

For Spam, vocabulary size $V = 3$, $\alpha = 1$:

$$\hat{\theta}_{\text{free}, \text{Spam}} = \frac{5 + 1}{8 + 3} = \frac{6}{11} \approx 0.545$$

$$\hat{\theta}_{\text{money}, \text{Spam}} = \frac{3 + 1}{8 + 3} = \frac{4}{11} \approx 0.364$$

$$\hat{\theta}_{\text{hello}, \text{Spam}} = \frac{0 + 1}{8 + 3} = \frac{1}{11} \approx 0.091$$

No more zero probabilities! The classifier can now handle any word, even ones unseen during training.

## Log Probabilities: Numerical Stability

### The Underflow Problem

The Naive Bayes prediction requires multiplying many probabilities:

$$P(C_k) \prod_{j=1}^{d} P(x_j \mid C_k)$$

Each probability is $\leq 1$, so multiplying thousands of them together produces a number very close to zero. With $d = 10{,}000$ features and each probability around 0.001:

$$0.001^{10{,}000} = 10^{-30{,}000}$$

This is smaller than the smallest float64 number ($\approx 10^{-308}$) — **numerical underflow**!

### The Log-Sum-Exp Trick

Taking the logarithm converts products to sums (which don't underflow) and monotonically preserves the argmax:

$$\hat{C} = \arg\max_{C_k} \log P(C_k) + \sum_{j=1}^{d} \log P(x_j \mid C_k)$$

Since $\log$ is strictly monotonically increasing:

$$\arg\max_x f(x) = \arg\max_x \log f(x)$$

**Practical rule**: Always compute in log space. Sum log-probabilities instead of multiplying probabilities.

### Recovering Probabilities

To recover actual class probabilities from log-scores $s_k = \log P(C_k) + \sum_j \log P(x_j \mid C_k)$, use the **softmax** function:

$$P(C_k \mid \mathbf{x}) = \frac{e^{s_k}}{\sum_{k'} e^{s_{k'}}}$$

To avoid overflow in the softmax, first subtract the maximum score:

$$P(C_k \mid \mathbf{x}) = \frac{e^{s_k - s_{\max}}}{\sum_{k'} e^{s_{k'} - s_{\max}}}$$

## Implementation from Scratch

### Gaussian Naive Bayes

```python
import numpy as np
from scipy.stats import norm

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier for continuous features.

    Assumes P(x_j | C_k) follows a Gaussian distribution.
    Parameters are estimated using Maximum Likelihood Estimation.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        """
        Parameters
        ----------
        var_smoothing : float
            Portion of the largest variance added to all variances
            for numerical stability (prevents zero variance).
        """
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_priors_ = None    # log P(C_k)
        self.means_ = None           # μ_{jk}
        self.variances_ = None       # σ²_{jk}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        """
        Estimate class priors and class-conditional Gaussian parameters.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,)
        """
        X, y = np.array(X, dtype=float), np.array(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.means_     = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        self.class_priors_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_priors_[idx] = X_c.shape[0] / X.shape[0]
            self.means_[idx]        = X_c.mean(axis=0)
            self.variances_[idx]    = X_c.var(axis=0)

        # Numerical stability: add a fraction of the max variance
        self.variances_ += self.var_smoothing * self.variances_.max()
        return self

    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log P(x_j | C_k) for every sample and class.

        Returns shape (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_likelihood = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            mu    = self.means_[idx]       # (n_features,)
            var   = self.variances_[idx]   # (n_features,)
            # log of Gaussian PDF: -0.5 * log(2π σ²) - (x - μ)² / (2σ²)
            log_likelihood[:, idx] = (
                -0.5 * np.sum(np.log(2 * np.pi * var))
                - 0.5 * np.sum(((X - mu) ** 2) / var, axis=1)
            )
        return log_likelihood

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Return unnormalised log posterior for each class."""
        X = np.array(X, dtype=float)
        log_prior = np.log(self.class_priors_)             # (n_classes,)
        log_likelihood = self._log_likelihood(X)           # (n_samples, n_classes)
        return log_prior + log_likelihood                  # broadcasts

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalised posterior probabilities (softmax)."""
        log_scores = self.predict_log_proba(X)
        log_scores -= log_scores.max(axis=1, keepdims=True)  # numerical stability
        proba = np.exp(log_scores)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy."""
        return np.mean(self.predict(X) == y)
```

### Multinomial Naive Bayes

```python
class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier for count-based features.

    Ideal for text classification with bag-of-words representation.
    Uses Laplace (add-alpha) smoothing to handle zero counts.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Parameters
        ----------
        alpha : float
            Laplace smoothing parameter. alpha=1 is add-one smoothing.
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_log_priors_ = None   # log P(C_k)
        self.log_theta_ = None          # log P(w_j | C_k), shape (n_classes, n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNaiveBayes":
        """
        Estimate class priors and word probabilities.

        Parameters
        ----------
        X : (n_samples, n_features)  — word count matrix (non-negative integers)
        y : (n_samples,)
        """
        X, y = np.array(X, dtype=float), np.array(y)
        self.classes_ = np.unique(y)
        n_classes  = len(self.classes_)
        n_features = X.shape[1]

        # Log priors
        class_counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        self.class_log_priors_ = np.log(class_counts / class_counts.sum())

        # Word counts per class: shape (n_classes, n_features)
        feature_counts = np.zeros((n_classes, n_features))
        for idx, c in enumerate(self.classes_):
            feature_counts[idx] = X[y == c].sum(axis=0)

        # Laplace smoothing and log-probabilities
        smoothed_counts = feature_counts + self.alpha
        # Denominator: total smoothed word count per class
        smoothed_totals = smoothed_counts.sum(axis=1, keepdims=True)
        self.log_theta_ = np.log(smoothed_counts / smoothed_totals)

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Log posterior: log P(C_k) + Σ_j x_j * log θ_{jk}

        The multinomial log-likelihood (ignoring the document-length constant)
        is X @ log_theta.T — a simple matrix multiply.
        """
        X = np.array(X, dtype=float)
        return self.class_log_priors_ + X @ self.log_theta_.T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_scores = self.predict_log_proba(X)
        log_scores -= log_scores.max(axis=1, keepdims=True)
        proba = np.exp(log_scores)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)
```

### Bernoulli Naive Bayes

```python
class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes classifier for binary/boolean features.

    Explicitly models absence of features, making it suitable for
    short text or binary-encoded categorical data.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_priors_ = None
        self.log_theta_     = None   # log P(x_j=1 | C_k)
        self.log_neg_theta_ = None   # log P(x_j=0 | C_k) = log(1 - θ_{jk})

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BernoulliNaiveBayes":
        X, y = (np.array(X, dtype=float) > 0).astype(float), np.array(y)
        self.classes_ = np.unique(y)
        n_classes  = len(self.classes_)
        n_features = X.shape[1]

        class_counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        self.class_log_priors_ = np.log(class_counts / class_counts.sum())

        # Number of documents per class containing each feature
        feature_present = np.zeros((n_classes, n_features))
        for idx, c in enumerate(self.classes_):
            feature_present[idx] = X[y == c].sum(axis=0)

        # Laplace smoothed: add alpha to presence and absence counts
        theta = (feature_present + self.alpha) / (class_counts[:, None] + 2 * self.alpha)
        self.log_theta_     = np.log(theta)
        self.log_neg_theta_ = np.log(1 - theta)

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        X = (np.array(X, dtype=float) > 0).astype(float)
        # For each sample: sum over features of x_j*log(θ) + (1-x_j)*log(1-θ)
        return (self.class_log_priors_
                + X @ self.log_theta_.T
                + (1 - X) @ self.log_neg_theta_.T)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_scores = self.predict_log_proba(X)
        log_scores -= log_scores.max(axis=1, keepdims=True)
        proba = np.exp(log_scores)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)
```

## Advantages and Limitations

### Advantages

1. **Fast training**: Single pass over training data, $O(n \cdot d)$
2. **Fast prediction**: Linear in number of features, $O(d \cdot K)$
3. **Scalable**: Handles massive vocabularies and datasets
4. **Small data**: Strong assumptions help generalise with few examples
5. **Probabilistic output**: Returns calibrated class probabilities
6. **Online learning**: Parameters update incrementally with new data
7. **Multi-class natural**: No binary decomposition needed
8. **Interpretable**: Can inspect per-feature, per-class probabilities
9. **Missing data**: Easy to handle — skip missing features in the product

### Limitations

1. **Independence assumption violated**: Features are often correlated in practice
2. **Probability estimates poorly calibrated**: Posterior probabilities are often extreme (very close to 0 or 1) due to compounding independent estimates
3. **Continuous features require Gaussian assumption**: May be wrong for skewed or multimodal data
4. **Zero frequency**: Requires smoothing; rare features are over-smoothed
5. **Not a universal approximator**: Linear decision boundaries for Gaussian NB with equal covariance
6. **Feature engineering required**: Bag-of-words loses word order ("not good" vs "good")

### When to Use Naive Bayes

**Use Naive Bayes when:**
- Text classification is the task (spam, sentiment, topic)
- Very high-dimensional features (large vocabulary)
- Dataset is small and a simple model is appropriate
- Need fast online learning or streaming updates
- Need interpretable probabilities
- Baseline is needed quickly

**Avoid Naive Bayes when:**
- Features have strong interactions that affect the class (e.g., XOR problems)
- Accurate probability calibration is critical (e.g., cost-sensitive decisions)
- Continuous features are heavily non-Gaussian

### Comparison with Other Classifiers

| Aspect | Naive Bayes | Logistic Regression | Decision Tree | SVM |
|--------|------------|---------------------|---------------|-----|
| Training speed | Very fast | Fast | Fast | Medium–Slow |
| Prediction speed | Very fast | Fast | Fast | Medium |
| Feature interactions | Ignored | Modelled | Modelled | Kernel-dependent |
| Probabilistic output | Yes | Yes | No (gini/entropy) | No (margin) |
| High-dimensional | Excellent | Good | Poor | Good (linear) |
| Small data | Good | OK | Overfit risk | Good |
| Text data | Excellent | Good | Poor | Good |

## Conclusion

Naive Bayes is a versatile and powerful classifier that excels in text classification tasks. It's a simple yet effective baseline, making it a popular choice for many applications. Its simplicity and speed make it a good choice for small datasets, while its accuracy and interpretability make it a good choice for larger datasets. It's a strong baseline and a good starting point for many machine learning tasks. In summary, we can say that Naive Bayes is a great choice for text classification tasks with small datasets.

### Key Takeaways

1. **Bayes' theorem** is the foundation: posterior = likelihood × prior / evidence
2. **The naive assumption** — conditional independence — makes the computation tractable while often preserving accuracy
3. **Three main variants**: Gaussian (continuous), Multinomial (counts/text), Bernoulli (binary)
4. **Laplace smoothing** is essential to prevent zero-probability catastrophes
5. **Work in log space** to avoid numerical underflow with many features
6. **Training is a single pass**: compute class-conditional statistics and priors
7. **Despite its simplicity**, Naive Bayes is a strong baseline and production-worthy classifier for text

### Further Reading

**Reference papers**:
- [Domingos, P. & Pazzani, M. (1997). "On the Optimality of the Simple Bayesian Classifier under Zero-One Loss"](https://www.cs.ucdavis.edu/~vemuri/classes/ecs271/Bayesian.pdf)
- [McCallum, A. & Nigam, K. (1998). "A Comparison of Event Models for Naive Bayes Text Classification"](https://cdn.aaai.org/Workshops/1998/WS-98-05/WS98-05-007.pdf)
