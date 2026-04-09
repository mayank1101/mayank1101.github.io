---
layout: post
title: "Model Evaluation Metrics: Measuring ML Performance"
date: 2025-12-14
series: "Machine Learning Series"
series_author: "Mayank Sharma"
series_image: "/assets/images/2025-12-14-evaluation-metrics/evaluation-metrics.png"
excerpt: "Learn essential evaluation metrics including accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices."
---

Continuing in our series on machine learning, today we explore different types of model evaluation metrics. Imagine you built a cancer screening model that achieves 99% accuracy. Sounds impressive right? But until you discover the dataset has 99% healthy patients and 1% with cancer, and your model just predicts "healthy" for everyone. It is perfectly accurate and completely useless. This is the central lesson of evaluation metrics: **a model is only as good as the metric used to measure it**. Choosing the right metric is not just a technical afterthought, it is a fundamental design decision that determines whether your model actually solves the real-world problem.

## Table of Contents

1. [Introduction: Why Metrics Matter](#introduction-why-metrics-matter)
2. [The Confusion Matrix: Foundation of Everything](#the-confusion-matrix-foundation-of-everything)
3. [Classification Metrics from the Confusion Matrix](#classification-metrics-from-the-confusion-matrix)
4. [The Precision–Recall Trade-off](#the-precision-recall-trade-off)
5. [F-Score: Balancing Precision and Recall](#f-score-balancing-precision-and-recall)
6. [ROC Curve and AUC](#roc-curve-and-auc)
7. [Precision-Recall Curve: For Imbalanced Data](#precision-recall-curve-for-imbalanced-data)
8. [Multi-Class Classification Metrics](#multi-class-classification-metrics)
9. [Regression Metrics](#regression-metrics)
10. [Cross-Validation Strategies](#cross-validation-strategies)
11. [Choosing the Right Metric](#choosing-the-right-metric)
12. [Implementation from Scratch](#implementation-from-scratch)
13. [Conclusion](#conclusion)

## Introduction: Why Metrics Matter

### The Metric Shapes the Model

When you train a model, you minimise a **loss function**. When you evaluate it, you compute a **metric**. These are distinct quantities:

- **Loss function**: Differentiable surrogate used during training (e.g., cross-entropy, MSE)
- **Evaluation metric**: Business-meaningful measure used to compare models (e.g., F1, AUC)

The disconnect between loss and metric is where most real-world ML failures originate. Optimising cross-entropy does not directly optimise recall. A model with the lowest validation loss may not have the highest AUC.

### The Imbalanced Data Problem

Consider a fraud detection system: 99.9% of transactions are legitimate; 0.1% are fraudulent.

- **Naïve model**: Always predict "legitimate"
- **Accuracy**: 99.9% ← looks great!
- **Recall for fraud**: 0% ← catastrophic

The naïve model is worse than useless for it catches zero fraud, and it yet dominates on accuracy. This motivates metrics that account for class imbalance and the different costs of different errors.

### Two Types of Errors

Every binary classifier makes two types of mistakes:

| Error Type | Also Called | Consequence |
|-----------|-------------|-------------|
| Predicting positive when actually negative | **False Positive** (Type I) | Wasted resources, false alarms |
| Predicting negative when actually positive | **False Negative** (Type II) | Missed threats, missed cases |

The relative cost of these errors depends entirely on the application:

- **Spam filter**: A false positive (legitimate email in spam) is annoying but tolerable; a false negative (spam in inbox) is minor — lean toward high precision
- **Cancer screening**: A false negative (missed cancer) is life-threatening; a false positive (unnecessary biopsy) is costly but survivable, so lean toward high recall
- **Self-driving car braking**: Both errors are dangerous, so need balance

## The Confusion Matrix: Foundation of Everything

### Definition

For a binary classifier with classes Positive (P) and Negative (N), the **confusion matrix** tabulates the four possible prediction outcomes:

$$\begin{array}{c|cc}
 & \text{Predicted Positive} & \text{Predicted Negative} \\
\hline
\text{Actual Positive} & \text{TP} & \text{FN} \\
\text{Actual Negative} & \text{FP} & \text{TN}
\end{array}$$

- **TP (True Positive)**: Correctly predicted positive — model said yes, reality is yes
- **TN (True Negative)**: Correctly predicted negative — model said no, reality is no
- **FP (False Positive)**: Incorrectly predicted positive — model said yes, reality is no
- **FN (False Negative)**: Incorrectly predicted negative — model said no, reality is yes

**Total predictions**: $N = \text{TP} + \text{TN} + \text{FP} + \text{FN}$

### Concrete Example: Spam Detection

You test a spam filter on 1,000 emails:
- 200 are actually spam; 800 are legitimate
- The filter flags 180 as spam

| | Predicted Spam | Predicted Legitimate |
|--|---------------|---------------------|
| **Actual Spam** | 150 (TP) | 50 (FN) |
| **Actual Legitimate** | 30 (FP) | 770 (TN) |

From these four numbers, every classification metric derives.

### Why the Confusion Matrix is Fundamental

The confusion matrix is the complete summary of binary classifier behaviour. Every scalar metric (accuracy, precision, recall, F1, MCC) is a function of TP, TN, FP, FN. Understanding the matrix lets you:
- Diagnose exactly where a model fails
- Choose the metric appropriate for your cost structure
- Set the decision threshold correctly

## Classification Metrics from the Confusion Matrix

### Accuracy

The fraction of all predictions that are correct:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{\text{TP} + \text{TN}}{N}$$

**Spam example**: $(150 + 770) / 1000 = 0.92$

**Failure mode**: Misleading when classes are imbalanced. A model predicting all-negative on a 95/5 split gets 95% accuracy while being useless.

**Use when**: Classes are roughly balanced and both error types have similar cost.

### Precision (Positive Predictive Value)

Of all instances the model predicted as positive, what fraction are actually positive?

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

**Spam example**: $150 / (150 + 30) = 0.833$

**Intuition**: If I flag an email as spam, how likely is it actually spam? A precision of 83.3% means 1 in 6 flagged emails is legitimate spam. This is potentially an issue for important messages.

**High precision requires**: Few false positives. Achieved by being conservative i.e., only predict positive when very confident.

### Recall (Sensitivity, True Positive Rate)

Of all instances that are actually positive, what fraction did the model correctly identify?

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

**Spam example**: $150 / (150 + 50) = 0.75$

**Intuition**: Of all the spam emails, what fraction did we catch? A recall of 75% means 1 in 4 spam emails slips through.

**High recall requires**: Few false negatives. Achieved by being aggressive i.e., predict positive whenever there's any chance.

### Specificity (True Negative Rate)

Of all actual negatives, what fraction did the model correctly identify?

$$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

**Spam example**: $770 / (770 + 30) = 0.9625$

**Intuition**: Of all legitimate emails, what fraction did we correctly leave in the inbox? The complement $1 - \text{Specificity} = \text{FPR}$ is the **False Positive Rate**, one axis of the ROC curve.

### False Positive Rate (FPR)

$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = 1 - \text{Specificity}$$

**Spam example**: $30 / (30 + 770) = 0.0375$

### False Negative Rate (FNR)

$$\text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}} = 1 - \text{Recall}$$

**Spam example**: $50 / (50 + 150) = 0.25$

### Matthews Correlation Coefficient (MCC)

A balanced metric that works well even for very imbalanced classes:

$$\text{MCC} = \frac{\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}$$

**Range**: $[-1, +1]$
- $+1$: Perfect prediction
- $0$: No better than random guessing
- $-1$: Perfectly wrong predictions

**Spam example**:
$$\text{MCC} = \frac{150 \cdot 770 - 30 \cdot 50}{\sqrt{180 \cdot 200 \cdot 800 \cdot 820}} = \frac{115500 - 1500}{\sqrt{23616000000}} \approx \frac{114000}{153,676} \approx 0.742$$

MCC is often the single most informative metric for binary classification, especially with imbalanced data.

## The Precision–Recall Trade-off

### Decision Threshold

Most classifiers output a probability or score, not a hard class label. The **decision threshold** $\tau$ converts this:

$$\hat{y} = \begin{cases} 1 & \text{if } P(\text{positive} \mid \mathbf{x}) \geq \tau \\ 0 & \text{otherwise} \end{cases}$$

The default is $\tau = 0.5$, but this is rarely optimal.

### How Threshold Affects Precision and Recall

- **Increasing $\tau$** (more conservative): Only predict positive when very confident
  - Fewer positives predicted → Fewer FP → **Higher Precision**
  - More positives missed → More FN → **Lower Recall**

- **Decreasing $\tau$** (more aggressive): Predict positive at the slightest signal
  - More positives predicted → More FP → **Lower Precision**
  - Fewer positives missed → Fewer FN → **Higher Recall**

**The trade-off is unavoidable**: you cannot simultaneously maximise both without a better model.

### Finding the Right Threshold

Given the cost of false positives $C_{FP}$ and false negatives $C_{FN}$, the optimal threshold satisfies:

$$\tau^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$$

**Example**: In fraud detection, missing a $10,000 fraud ($C_{FN} = 10{,}000$) is far worse than triggering a review incorrectly ($C_{FP} = 50$):
$$\tau^* = \frac{50}{50 + 10000} \approx 0.005$$

Set the threshold very low ($\tau \approx 0$) to flag almost everything as fraudulent to minimise missed frauds.

## F-Score: Balancing Precision and Recall

### F1-Score

The **F1-score** is the harmonic mean of precision and recall:

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}$$

**Why harmonic mean, not arithmetic mean?**

The harmonic mean punishes extreme imbalances between precision and recall. A model with Precision = 1.0 and Recall = 0.0 is useless, yet:
- Arithmetic mean: $(1.0 + 0.0) / 2 = 0.5$ (looks mediocre but acceptable)
- Harmonic mean: $2 \cdot 1.0 \cdot 0.0 / (1.0 + 0.0) = 0$ (correctly reflects worthlessness)

**Spam example**: $F_1 = 2 \cdot 0.833 \cdot 0.75 / (0.833 + 0.75) = 0.789$

**Use when**: You want a single number that balances precision and recall and classes are imbalanced.

### F$\beta$-Score: Weighting the Trade-off

When one error type is more costly than the other, use the generalised $F_\beta$-score:

$$F_\beta = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

- $\beta > 1$: Recall is more important (e.g., cancer screening: missing a case is worse)
- $\beta < 1$: Precision is more important (e.g., spam filter: false positives annoy users)
- $\beta = 1$: Equal weight → F1-score

**F2-score** ($\beta = 2$, recall twice as important):

$$F_2 = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$

**F0.5-score** ($\beta = 0.5$, precision twice as important):

$$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

### Derivation: Why This Form?

The $F_\beta$ score is defined as the weighted harmonic mean of precision and recall:

$$\frac{1}{F_\beta} = \frac{1}{1+\beta^2} \cdot \frac{1}{\text{Recall}} + \frac{\beta^2}{1+\beta^2} \cdot \frac{1}{\text{Precision}}$$

The weight $\beta^2$ on the recall term means we weight recall $\beta$ times as much as precision. Solving for $F_\beta$ gives the formula above.

## ROC Curve and AUC

### The Receiver Operating Characteristic Curve

The **ROC curve** plots **True Positive Rate (Recall)** against **False Positive Rate** as the decision threshold varies from 1 to 0:

$$\text{TPR}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau)}, \qquad \text{FPR}(\tau) = \frac{\text{FP}(\tau)}{\text{FP}(\tau) + \text{TN}(\tau)}$$

**Tracing the curve**: As $\tau$ decreases from 1 to 0, both TPR and FPR increase, tracing a curve from $(0, 0)$ to $(1, 1)$.

### Three Reference Points

| Point | $\tau$ | Meaning |
|-------|--------|---------|
| $(0, 0)$ | $\tau = 1$ | Never predict positive: TPR = 0, FPR = 0 |
| $(1, 1)$ | $\tau = 0$ | Always predict positive: TPR = 1, FPR = 1 |
| $(0, 1)$ | Optimal | Perfect classifier: TPR = 1, FPR = 0 |

The **diagonal line** $\text{TPR} = \text{FPR}$ represents a random classifier.

### Area Under the Curve (AUC-ROC)

The **AUC** summarises the ROC curve as a single number:

$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})$$

**Range**: $[0, 1]$
- $\text{AUC} = 1.0$: Perfect classifier
- $\text{AUC} = 0.5$: Random classifier (diagonal)
- $\text{AUC} < 0.5$: Worse than random (flip predictions)

**Probabilistic interpretation**: AUC equals the probability that the model assigns a higher score to a randomly chosen positive instance than to a randomly chosen negative instance:

$$\text{AUC} = P(\text{score}(\text{positive}) > \text{score}(\text{negative}))$$

This makes AUC a **threshold-free** metric — it evaluates the quality of the ranking, not the quality of any particular threshold.

### Computing AUC with the Trapezoid Rule

Given sorted thresholds with corresponding $(FPR_i, TPR_i)$ pairs:

$$\text{AUC} \approx \sum_{i=1}^{m} (FPR_i - FPR_{i-1}) \cdot \frac{TPR_i + TPR_{i-1}}{2}$$

This is the trapezoidal approximation of the integral.

### AUC Advantages

- **Threshold-independent**: Evaluates the full ranking quality
- **Scale-invariant**: Measures ranking quality regardless of probability calibration
- **Class-balance-robust**: Less sensitive to imbalance than accuracy
- **Comparable across models**: Even if optimal thresholds differ

### AUC Limitations

- **Does not reflect calibration**: A model with perfect AUC can have wildly wrong probability estimates
- **Misleading for very imbalanced data**: A model that ranks well but puts most positives in the top 30% looks great on ROC but may miss most positives in practice
- **Averaged over all thresholds**: Includes operating points never used in practice

## Precision-Recall Curve: For Imbalanced Data

### Why PR Curves Complement ROC Curves

For highly imbalanced datasets (e.g., fraud at 0.1%), the ROC curve can be misleading because FPR involves TN, which is enormous and makes FPR appear small even when many false positives exist.

The **Precision-Recall curve** plots Precision vs. Recall across thresholds, and is more informative when:
- The positive class is rare
- You care more about performance on the positive class
- False positives have significant cost

### Average Precision (AP)

The **Average Precision** is the area under the PR curve, approximated as:

$$\text{AP} = \sum_{k=1}^{n} (R_k - R_{k-1}) \cdot P_k$$

Where $P_k$ and $R_k$ are precision and recall at the $k$-th threshold.

**Range**: $[0, 1]$; higher is better. A random classifier's baseline is the class prevalence $\pi = \text{TP} / (\text{TP} + \text{FN})$.

### Interpolated Precision

The PR curve is typically jagged. The **interpolated precision** at recall level $r$ takes the maximum precision achievable at recall $\geq r$:

$$P_{interp}(r) = \max_{\tilde{r} \geq r} P(\tilde{r})$$

This is used in object detection evaluation (mAP).

### ROC vs PR Curve: Rule of Thumb

- **Balanced classes**: Either works; ROC is more standard
- **Imbalanced classes**: PR curve is more informative
- **Comparing models globally**: ROC-AUC
- **Evaluating at a specific operating point**: PR curve at that recall level

## Multi-Class Classification Metrics

When there are $K > 2$ classes, metrics extend through **averaging strategies**.

### Per-Class Metrics

For each class $k$ treated as the "positive" class (all others are "negative"):

$$\text{Precision}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}, \qquad \text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}, \qquad F_{1,k} = \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

### Macro Averaging

Compute metric per class, then take the **unweighted mean**:

$$\text{Precision}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{Precision}_k$$

**Properties**: Treats all classes equally regardless of support (number of instances). Good when all classes are equally important. Penalises poor performance on rare classes.

### Micro Averaging

**Pool** all TP, FP, FN across classes, then compute metric:

$$\text{Precision}_{\text{micro}} = \frac{\sum_{k=1}^{K} \text{TP}_k}{\sum_{k=1}^{K} (\text{TP}_k + \text{FP}_k)}$$

**Properties**: Gives equal weight to each **instance**, not each class. Dominated by the most frequent classes. For multi-class, micro-precision equals micro-recall equals accuracy (when each instance is assigned exactly one class).

### Weighted Averaging

Compute metric per class, then take the **support-weighted mean**:

$$\text{Precision}_{\text{weighted}} = \frac{\sum_{k=1}^{K} n_k \cdot \text{Precision}_k}{\sum_{k=1}^{K} n_k}$$

Where $n_k$ is the number of actual instances of class $k$.

**Properties**: Accounts for class imbalance while still computing per-class metrics. Best for imbalanced multi-class problems.

### When to Use Which

| Strategy | Use When |
|----------|----------|
| Macro | All classes equally important; want to penalise poor performance on rare classes |
| Micro | Individual instances equally important; dominated by frequent classes |
| Weighted | Imbalanced classes; want aggregate that reflects class distribution |
| Per-class | Need to understand performance on a specific class |

### Multi-Class Confusion Matrix

Generalises naturally: an $n \times n$ matrix where row $i$ is the true class and column $j$ is the predicted class. Diagonal elements are correct predictions; off-diagonal shows what the model confuses.

## Regression Metrics

When the target is continuous, we need metrics that measure the magnitude of errors.

### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Properties**:
- Same units as target variable — highly interpretable
- Robust to outliers (linear penalty)
- Non-differentiable at zero (not directly usable as a loss function in some frameworks)

**Example**: Predicting house prices in $1,000s. MAE = 15 means predictions are off by $15,000 on average.

### Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Properties**:
- Penalises large errors quadratically — sensitive to outliers
- Units are squared (less interpretable than MAE)
- Smooth and differentiable — preferred as a loss function
- Equivalent to variance of errors when mean error is zero

**Mathematical relationship**: $\text{MSE} = \text{Bias}^2 + \text{Variance}$ when decomposed.

### Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Properties**:
- Same units as target variable (more interpretable than MSE)
- Still penalises large errors more than MAE
- Most commonly reported regression metric

### Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

**Properties**:
- Scale-independent (percentage) — comparable across different scales
- Undefined when $y_i = 0$
- Asymmetric: penalises under-predictions more than over-predictions
- Biased toward models that under-predict

### R-Squared (Coefficient of Determination)

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

Where:
- $\text{SS}_{\text{res}} = \sum_i (y_i - \hat{y}_i)^2$ is the **residual sum of squares** (unexplained variance)
- $\text{SS}_{\text{tot}} = \sum_i (y_i - \bar{y})^2$ is the **total sum of squares** (total variance)

**Interpretation**: Fraction of the target's variance explained by the model.

**Range**: $(-\infty, 1]$
- $R^2 = 1$: Perfect fit
- $R^2 = 0$: Model performs no better than predicting the mean
- $R^2 < 0$: Model is worse than predicting the mean

**Warning**: $R^2$ always increases when you add more features, even if they're noise. Use **Adjusted R²** to penalise unnecessary complexity:

$$R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

Where $p$ is the number of predictors and $n$ is sample size.

### Huber Loss

A compromise between MAE and MSE that is quadratic for small errors and linear for large ones:

$$L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta \cdot \left(|y - \hat{y}| - \frac{\delta}{2}\right) & \text{otherwise}
\end{cases}$$

**Properties**: Differentiable everywhere, robust to outliers, controlled by $\delta$ (the transition point).

### Comparing Regression Metrics

| Metric | Units | Outlier Sensitivity | Interpretability | Use When |
|--------|-------|---------------------|------------------|----------|
| MAE | Same as $y$ | Low | High | Outliers present; interpretability needed |
| MSE | Squared | High | Low | Differentiable training loss; outliers matter |
| RMSE | Same as $y$ | High | Medium | Standard reporting; same scale as MAE |
| MAPE | % | Medium | Very High | Comparing across scales; no zeros in $y$ |
| R² | Unitless | High | Very High | Explaining variance; comparing models |

## Cross-Validation Strategies

A single train/test split can give misleading estimates. **Cross-validation** uses the data more efficiently to get a reliable performance estimate with uncertainty.

### k-Fold Cross-Validation

Divide the data into $k$ equal folds. For each fold $i$:
1. Train on the remaining $k-1$ folds
2. Evaluate on fold $i$

Final estimate: mean and standard deviation across $k$ scores.

$$\text{CV score} = \frac{1}{k} \sum_{i=1}^{k} \text{metric}_i, \qquad \text{std} = \sqrt{\frac{1}{k}\sum_{i=1}^{k}(\text{metric}_i - \text{CV score})^2}$$

**Common choices**: $k = 5$ or $k = 10$. Larger $k$:
- ↑ Bias reduction (more training data per fold)
- ↑ Variance (fewer points per validation fold)
- ↑ Computational cost

### Stratified k-Fold

When classes are imbalanced, standard k-fold may create folds with no minority class instances. **Stratified k-fold** ensures each fold has approximately the same class proportions as the full dataset.

**Essential for**: Imbalanced classification. Gives more reliable estimates than standard k-fold.

### Leave-One-Out Cross-Validation (LOO-CV)

Special case of k-fold where $k = n$: each fold has one sample as validation.

$$\text{LOO-CV} = \frac{1}{n} \sum_{i=1}^{n} \text{metric}(y_i, \hat{y}_{-i})$$

Where $\hat{y}_{-i}$ is the prediction for sample $i$ when trained on all other samples.

**Properties**:
- Least biased estimate (trains on maximum data each time)
- Highest variance (each validation set is a single point)
- Very expensive: $n$ model fits required
- Best for very small datasets

### Time-Series Cross-Validation (Walk-Forward Validation)

For time-series data, future data must never be used to predict the past. Standard k-fold would leak future information.

**Walk-forward validation** uses an expanding window:
- Fold 1: Train on $[1, t_1]$, evaluate on $(t_1, t_2]$
- Fold 2: Train on $[1, t_2]$, evaluate on $(t_2, t_3]$
- ...

This strictly respects temporal ordering.

### Repeated k-Fold

Run k-fold $m$ times with different random splits, giving $k \times m$ estimates. Reduces variance of the CV estimate at the cost of $m \times$ computation. Common: $5 \times 2$-fold or $10 \times 3$-fold.

### Nested Cross-Validation

When tuning hyperparameters and evaluating simultaneously, use **nested CV**:
- **Outer loop**: $k$-fold to estimate generalisation performance
- **Inner loop**: $j$-fold within each outer training fold to tune hyperparameters

Prevents the optimistic bias from selecting the best hyperparameter on the test fold.

## Choosing the Right Metric

### Decision Framework

```
Start: What type of problem?
├── Classification
│   ├── Binary
│   │   ├── Balanced classes?
│   │   │   ├── Yes → Accuracy or F1
│   │   │   └── No → F1, MCC, or AUC-PR
│   │   ├── FP cost >> FN cost? → High Precision, use F-beta (β < 1)
│   │   ├── FN cost >> FP cost? → High Recall, use F-beta (β > 1)
│   │   └── Need threshold-free eval? → ROC-AUC
│   └── Multi-class
│       ├── All classes equally important? → Macro F1
│       ├── Instance-level importance? → Micro F1 (= Accuracy)
│       └── Imbalanced? → Weighted F1 or per-class metrics
└── Regression
    ├── Outliers present? → MAE or Huber
    ├── Outliers important? → MSE or RMSE
    ├── Need scale-invariant? → MAPE or RMSLE
    └── Need explained variance? → R²
```

### Metric Selection by Domain

| Domain | Preferred Metrics | Reason |
|--------|------------------|--------|
| Medical diagnosis | Recall, F2-score | Missing a disease (FN) is worse than over-diagnosing (FP) |
| Spam filtering | Precision, F0.5-score | Misclassifying legitimate email (FP) is worse than missing spam (FN) |
| Fraud detection | Recall, AUC-PR | Rare positives; missing fraud is costly |
| Search ranking | NDCG, MAP | Ranking quality matters; not binary |
| Autonomous driving | Recall, Specificity | Both FP and FN are dangerous |
| House price prediction | RMSE, R² | Symmetric errors; interpretability |
| Demand forecasting | MAPE | Need scale-independent comparison across products |

## Implementation from Scratch

### All Classification Metrics from Scratch

```python
import numpy as np

def confusion_matrix_binary(y_true, y_pred):
    """Compute TP, TN, FP, FN from binary arrays."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TP, TN, FP, FN


def accuracy(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix_binary(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)


def precision(y_true, y_pred):
    TP, _, FP, _ = confusion_matrix_binary(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0


def recall(y_true, y_pred):
    TP, _, _, FN = confusion_matrix_binary(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def f_beta(y_true, y_pred, beta=1.0):
    """
    F-beta score: harmonic mean of precision and recall,
    with recall weighted beta times as much as precision.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    b2 = beta ** 2
    denom = b2 * p + r
    return (1 + b2) * p * r / denom if denom > 0 else 0.0


def mcc(y_true, y_pred):
    """Matthews Correlation Coefficient."""
    TP, TN, FP, FN = confusion_matrix_binary(y_true, y_pred)
    denom = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return (TP*TN - FP*FN) / denom if denom > 0 else 0.0


def roc_curve_scratch(y_true, y_scores):
    """
    Compute ROC curve points by sweeping the decision threshold.

    Returns (fpr, tpr, thresholds) sorted ascending by FPR.
    """
    y_true   = np.array(y_true)
    y_scores = np.array(y_scores)
    thresholds = np.sort(np.unique(y_scores))[::-1]  # descending

    P = y_true.sum()
    N = len(y_true) - P

    fpr_list, tpr_list = [0.0], [0.0]
    for tau in thresholds:
        y_pred = (y_scores >= tau).astype(int)
        TP, TN, FP, FN = confusion_matrix_binary(y_true, y_pred)
        fpr_list.append(FP / N if N > 0 else 0.0)
        tpr_list.append(TP / P if P > 0 else 0.0)

    fpr_list.append(1.0)
    tpr_list.append(1.0)
    return np.array(fpr_list), np.array(tpr_list), thresholds


def auc_trapezoid(fpr, tpr):
    """Area under curve using the trapezoid rule."""
    return float(np.trapz(tpr, fpr))


# ── Regression metrics ─────────────────────────────────────────────────────────

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def r_squared(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

def adjusted_r_squared(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r_squared(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


# ── k-Fold Cross-Validation ────────────────────────────────────────────────────

def kfold_cv(model, X, y, k=5, metric_fn=None, random_state=42):
    """
    k-fold cross-validation returning per-fold scores.

    Parameters
    ----------
    model      : object with .fit(X, y) and .predict(X)
    X          : array (n_samples, n_features)
    y          : array (n_samples,)
    k          : int, number of folds
    metric_fn  : callable(y_true, y_pred) → float; defaults to accuracy
    """
    if metric_fn is None:
        metric_fn = accuracy

    np.random.seed(random_state)
    X, y = np.array(X), np.array(y)
    n = len(y)
    indices = np.random.permutation(n)
    fold_size = n // k
    scores = []

    for i in range(k):
        val_idx   = indices[i * fold_size: (i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size],
                                    indices[(i + 1) * fold_size:]])
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        scores.append(metric_fn(y[val_idx], y_pred))

    return np.array(scores)


# Example usage
if __name__ == "__main__":
    y_true   = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred   = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    y_scores = [0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.6, 0.3, 0.85, 0.15]

    print("Classification Metrics:")
    print(f"  Accuracy  : {accuracy(y_true, y_pred):.4f}")
    print(f"  Precision : {precision(y_true, y_pred):.4f}")
    print(f"  Recall    : {recall(y_true, y_pred):.4f}")
    print(f"  F1-Score  : {f_beta(y_true, y_pred, beta=1.0):.4f}")
    print(f"  F2-Score  : {f_beta(y_true, y_pred, beta=2.0):.4f}")
    print(f"  MCC       : {mcc(y_true, y_pred):.4f}")

    fpr, tpr, _ = roc_curve_scratch(y_true, y_scores)
    print(f"  AUC-ROC   : {auc_trapezoid(fpr, tpr):.4f}")

    y_reg_true = [3.0, -0.5, 2.0, 7.0]
    y_reg_pred = [2.5,  0.0, 2.0, 8.0]
    print("\nRegression Metrics:")
    print(f"  MAE  : {mae(y_reg_true, y_reg_pred):.4f}")
    print(f"  MSE  : {mse(y_reg_true, y_reg_pred):.4f}")
    print(f"  RMSE : {rmse(y_reg_true, y_reg_pred):.4f}")
    print(f"  R²   : {r_squared(y_reg_true, y_reg_pred):.4f}")
```

## Advantages and Limitations Summary

### Classification Metrics at a Glance

| Metric | Range | Imbalance-robust | Threshold-free | Interpretable |
|--------|-------|-----------------|----------------|---------------|
| Accuracy | [0,1] | ❌ | ❌ | ✅ |
| Precision | [0,1] | Partly | ❌ | ✅ |
| Recall | [0,1] | Partly | ❌ | ✅ |
| F1 | [0,1] | ✅ | ❌ | ✅ |
| F-beta | [0,1] | ✅ | ❌ | ✅ |
| MCC | [-1,1] | ✅ | ❌ | ✅ |
| ROC-AUC | [0,1] | Partly | ✅ | Medium |
| AUC-PR | [0,1] | ✅ | ✅ | Medium |

## Conclusion

In summary, classification metrics are a valuable tool for model evaluation, with a range of advantages and limitations. So, choosing the right metric depends on the problem, model, and evaluation goals.

### Key Takeaways

1. **Accuracy misleads** on imbalanced datasets — never use it alone
2. **Precision and recall trade off**: tuning the threshold shifts between them; $\beta$ controls the preference
3. **F1 = harmonic mean** of precision and recall; punishes models that sacrifice one for the other
4. **AUC-ROC is threshold-free**: measures ranking quality across all operating points
5. **AUC-PR beats AUC-ROC for imbalanced data**: PR curve is more sensitive to minority-class performance
6. **Multi-class metrics aggregate** per-class scores; choose macro/micro/weighted based on class importance
7. **Regression metrics differ in outlier sensitivity**: MAE is robust; MSE/RMSE are not
8. **R² measures explained variance**; adjusted R² penalises unnecessary features
9. **Cross-validation gives reliable estimates**: stratified for classification, walk-forward for time series

### Practical Checklist

Here are some best practices to follow when evaluating classification models:

- **Define your error costs before choosing a metric**: Is FP or FN more expensive?
- **Check class balance before reporting accuracy**: switch to F1 or MCC if imbalanced
- **Report both the metric value and its CV standard deviation**: a model at 0.85 ± 0.01 beats 0.86 ± 0.08
- **Plot the full ROC and PR curves**: aggregate AUC hides where the model works well vs. poorly
- **Use stratified k-fold** for all classification problems with imbalanced classes
- **Separate hyperparameter tuning from evaluation** using nested CV
- **Report multiple metrics**: no single metric tells the complete story