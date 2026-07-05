---
layout: post
title: "Linear Regression: From Simple to Multiple Regression"
date: 2025-07-07
series: "Machine Learning for Engineers"
series_author: "Mayank Sharma"
excerpt: "Master linear regression from scratch, understanding simple and multiple regression, cost functions, and gradient descent."
---

Today we dive deep into linear regression, the cornerstone of predictive modeling. Imagine you're a real estate agent trying to estimate house prices. You notice that larger houses generally sell for more money. Intuitively, you understand there's a relationship between size and price, but can you quantify it? Can you predict the price of a 2,000 square foot house based on past sales data? This is exactly what linear regression does, it finds the mathematical relationship between variables and uses it to make predictions.

## Table of Contents

1. [Introduction: The Power of Linear Relationships](#introduction-the-power-of-linear-relationships)
2. [Simple Linear Regression: The Foundation](#simple-linear-regression-the-foundation)
3. [The Mathematics Behind the Line](#the-mathematics-behind-the-line)
4. [Cost Function: Measuring Our Mistakes](#cost-function-measuring-our-mistakes)
5. [Gradient Descent: Learning the Best Fit](#gradient-descent-learning-the-best-fit)
6. [Feature Scaling: Making Gradient Descent Work](#feature-scaling-making-gradient-descent-work)
7. [Multiple Linear Regression: Beyond One Variable](#multiple-linear-regression-beyond-one-variable)
8. [The Normal Equation: Analytical Solution](#the-normal-equation-analytical-solution)
9. [Train/Test Split: Honest Evaluation](#traintest-split-honest-evaluation)
10. [Assumptions of Linear Regression](#assumptions-of-linear-regression)
11. [Model Evaluation and Interpretation](#model-evaluation-and-interpretation)
12. [Conclusion](#conclusion)

## Introduction: The Power of Linear Relationships

### What is Linear Regression?

Linear regression is one of the most fundamental algorithms in machine learning and statistics. Despite its simplicity, it's incredibly powerful and forms the foundation for understanding more complex algorithms. At its core, linear regression models the relationship between:

- **Independent variables** (also called features, predictors, or inputs): The variables we use to make predictions
- **Dependent variable** (also called target, response, or output): The variable we want to predict

The "linear" part means we assume the relationship between these variables can be represented by a straight line (in 2D) or a hyperplane (in higher dimensions).

## Simple Linear Regression: The Foundation

### The Intuition

Let's start with the simplest case: predicting one variable (y) from another variable (x). Think of it as drawing the "best fit" line through a scatter plot of data points.

Consider predicting a student's test score based on hours studied:

```
Hours Studied (x) | Test Score (y)
------------------|---------------
     1            |     50
     2            |     55
     3            |     65
     4            |     70
     5            |     75
```

If you plotted these points, you'd see they roughly follow a straight line. Simple linear regression finds the equation of the line that best fits these points.

### The Linear Equation

The relationship is modeled as:

$$y = mx + b$$

Or in machine learning notation:

$$\hat{y} = \theta_0 + \theta_1 x$$

Where:
- $\hat{y}$ (y-hat) is the predicted value
- $x$ is the input feature
- $\theta_0$ is the **intercept** (also called bias): the predicted value when x = 0
- $\theta_1$ is the **slope** (also called weight): how much y changes for each unit increase in x

**Important Note**: We use $\hat{y}$ to denote predictions to distinguish them from actual values $y$.

### What Makes a "Best Fit" Line?

Consider three possible lines through our data points:

```
Line 1: y = 45 + 5x    (too steep)
Line 2: y = 48 + 5.5x  (just right)
Line 3: y = 50 + 4x    (not steep enough)
```

How do we determine which is "best"? We measure how far off each line's predictions are from the actual values. The line that minimizes these errors is the best fit.

## The Mathematics Behind the Line

### Understanding Errors (Residuals)

For each data point, we can calculate the **residual**, the difference between the actual value and the predicted value:

$$\text{residual}_i = y_i - \hat{y}_i = y_i - (\theta_0 + \theta_1 x_i)$$

If the residual is:
- **Positive**: Our prediction is too low
- **Negative**: Our prediction is too high
- **Zero**: Perfect prediction (rare in practice)

### Why Not Just Sum the Residuals?

You might think: "Why not just add up all the residuals and minimize that?" The problem is that positive and negative residuals would cancel out. A line that's way too high for some points and way too low for others could have a sum of zero!

**Example:**
```
Point 1: residual = +10 (predicted too low)
Point 2: residual = -10 (predicted too high)
Sum = 0 (looks perfect, but isn't!)
```

## Cost Function: Measuring Our Mistakes

### Mean Squared Error (MSE)

The solution is to **square** the residuals before summing them. This ensures all errors are positive and also penalizes larger errors more heavily (which is often desirable).

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 = \frac{1}{2m} \sum_{i=1}^{m} (y_i - (\theta_0 + \theta_1 x_i))^2$$

Where:
- $J$ is the cost function (also called loss function or objective function)
- $m$ is the number of training examples
- The $\frac{1}{2}$ in $\frac{1}{2m}$ is purely for mathematical convenience — when we take the derivative later, the exponent 2 comes down and cancels the $\frac{1}{2}$, giving us cleaner gradient formulas

### Why Square the Errors?

Squaring serves several purposes:

1. **Eliminates negative values**: All errors contribute positively
2. **Emphasizes larger errors**: An error of 10 contributes 100, while an error of 1 contributes only 1
3. **Mathematical properties**: Creates a smooth, differentiable function, that is very important for optimization
4. **Statistical foundation**: Under certain assumptions, minimizing MSE gives the maximum likelihood estimate

### Visualizing the Cost Function

If we plot $J(\theta_0, \theta_1)$ as a function of the parameters, we get a bowl-shaped surface in 3D (or a parabola in 2D if we fix one parameter). Our goal is to find the bottom of this bowl, the point where the cost is minimized. Because this bowl has a single lowest point (it is **convex**), gradient descent is guaranteed to find the global minimum for linear regression.

## Gradient Descent: Learning the Best Fit

### The Optimization Problem

We want to find the values of $\theta_0$ and $\theta_1$ that minimize the cost function $J$. This is an optimization problem. While there's an analytical solution (which we'll cover later), gradient descent provides a general approach that works for more complex models too.

### The Intuition: Hiking Down a Mountain

Imagine you're on a foggy mountain and want to reach the valley (minimum). You can't see the whole landscape, but you can feel the slope beneath your feet. So, what strategy will you follow?

- Figure out which direction is downhill (compute the gradient)
- Take a step in that direction
- Repeat until you reach the bottom

This is exactly what gradient descent does with our cost function.

### The Algorithm

Gradient descent updates the parameters iteratively:

$$\theta_0 := \theta_0 - \alpha \frac{\partial J}{\partial \theta_0}$$

$$\theta_1 := \theta_1 - \alpha \frac{\partial J}{\partial \theta_1}$$

Where:
- $\alpha$ is the **learning rate**: controls how big our steps are
- $\frac{\partial J}{\partial \theta_j}$ is the **partial derivative**: indicates the direction and magnitude of steepest ascent

### Computing the Gradients (Step-by-Step Derivation)

Let's carefully derive the partial derivatives using the chain rule. Starting from:

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - (\theta_0 + \theta_1 x_i))^2$$

**Derivative with respect to $\theta_0$:**

$$\frac{\partial J}{\partial \theta_0} = \frac{1}{2m} \sum_{i=1}^{m} 2(y_i - (\theta_0 + \theta_1 x_i)) \cdot \frac{\partial}{\partial \theta_0}(y_i - \theta_0 - \theta_1 x_i)$$

The inner derivative $\frac{\partial}{\partial \theta_0}(y_i - \theta_0 - \theta_1 x_i) = -1$, so:

$$\frac{\partial J}{\partial \theta_0} = \frac{1}{2m} \sum_{i=1}^{m} 2(y_i - \hat{y}_i)(-1) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)$$

Notice how the 2 from the power rule cancelled with the $\frac{1}{2}$ — that's exactly why we included it in the cost function.

**Derivative with respect to $\theta_1$:**

The inner derivative $\frac{\partial}{\partial \theta_1}(y_i - \theta_0 - \theta_1 x_i) = -x_i$, so:

$$\frac{\partial J}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \cdot x_i$$

**Intuition**:
- For $\theta_0$: The average error tells us how much to adjust the intercept
- For $\theta_1$: The average of (error × input) tells us how much to adjust the slope

### Numerical Walkthrough: Seeing Gradient Descent in Action

Let's trace through the first two iterations manually with a tiny dataset:

```
Data: (x=1, y=2), (x=2, y=4)
True relationship: y = 2x  (so θ₀=0, θ₁=2 is ideal)
Start: θ₀ = 0, θ₁ = 0,  α = 0.1
```

**Iteration 1:**

Predictions: $\hat{y}_1 = 0 + 0(1) = 0$, $\hat{y}_2 = 0 + 0(2) = 0$

Errors: $\hat{y}_1 - y_1 = 0 - 2 = -2$, $\hat{y}_2 - y_2 = 0 - 4 = -4$

Cost: $J = \frac{1}{2(2)}[(-2)^2 + (-4)^2] = \frac{1}{4}[4 + 16] = 5.0$

Gradients:
$$\frac{\partial J}{\partial \theta_0} = \frac{1}{2}[(-2) + (-4)] = -3.0$$
$$\frac{\partial J}{\partial \theta_1} = \frac{1}{2}[(-2)(1) + (-4)(2)] = \frac{1}{2}[-2 - 8] = -5.0$$

Update:
$$\theta_0 = 0 - 0.1(-3.0) = 0.3$$
$$\theta_1 = 0 - 0.1(-5.0) = 0.5$$

**Iteration 2:**

Predictions: $\hat{y}_1 = 0.3 + 0.5(1) = 0.8$, $\hat{y}_2 = 0.3 + 0.5(2) = 1.3$

Errors: $0.8 - 2 = -1.2$, $1.3 - 4 = -2.7$

Cost: $J = \frac{1}{4}[1.44 + 7.29] = 2.18$ ← **Cost dropped from 5.0 to 2.18!**

After many iterations, $\theta_0$ will approach 0 and $\theta_1$ will approach 2, converging to the true relationship.

### When Does Gradient Descent Stop? (Convergence)

Running for a fixed number of iterations works, but it's not always ideal. A principled stopping rule checks whether the cost has changed meaningfully:

$$|\, J(\text{iteration } t) - J(\text{iteration } t-1)\, | < \epsilon$$

If the cost changes by less than a tiny threshold $\epsilon$ (e.g., $10^{-6}$) between iterations, the algorithm has effectively converged and further iterations won't help.

You can also **detect problems** early by watching the cost:
- **Cost is decreasing smoothly** → gradient descent is working
- **Cost is decreasing very slowly** → learning rate is too small
- **Cost is increasing or oscillating** → learning rate is too large; reduce it

### The Learning Rate: A Balancing Act

The learning rate $\alpha$ is crucial:

- **Too small**: Learning is painfully slow; might need millions of iterations
- **Too large**: Might overshoot the minimum or even diverge (cost goes up instead of down)
- **Just right**: Converges efficiently to the minimum

Typical starting values: 0.001, 0.01, 0.1. A common strategy is to try 0.01 first, plot the cost history, and adjust based on what you see.

### Batch vs. Stochastic vs. Mini-Batch

Now that we have the gradient, we need to decide how to use it to update our parameters. There are three main approaches:

**Batch Gradient Descent** (what we described above):
- Uses all training examples in each iteration
- Stable convergence
- Can be slow for large datasets

**Stochastic Gradient Descent (SGD)**:
- Uses one random example per iteration
- Much faster per iteration
- More erratic path to minimum (can be beneficial for escaping local minima)

**Mini-Batch Gradient Descent**:
- Uses a small batch (e.g., 32, 64, 128 examples) per iteration
- Best of both worlds: faster than batch, more stable than SGD
- Most commonly used in practice

## Feature Scaling: Making Gradient Descent Work

### Why Feature Scaling Matters

This is one of the most important practical steps that beginners often skip, leading to gradient descent that is painfully slow or fails entirely.

Consider a house price model with two features:
- **Size**: ranges from 500 to 5,000 sq ft
- **Age**: ranges from 0 to 50 years

Because size values are 100× larger than age values, the cost function becomes a very elongated bowl rather than a circular one:

```
Without scaling (elongated bowl):       With scaling (circular bowl):
                                        
    θ₁ ^                                  θ₁ ^
       |   ___________                        |    ___
       |  /           \                       |   /   \
       | |             |                      |  |  *  |
       | |    *        |                      |   \___/
       |  \___________/                       |
       +---------------> θ₀                  +-----------> θ₀
       
  Gradient descent zigzags             Gradient descent goes
  and takes many steps                 straight to minimum
```

The gradient in the steep direction (size) dominates, causing gradient descent to zigzag back and forth and converge slowly.

### Two Common Scaling Methods

**Min-Max Normalization** (scales features to [0, 1]):

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

**Z-score Standardization** (scales features to mean=0, std=1):

$$x' = \frac{x - \mu}{\sigma}$$

Where $\mu$ is the mean and $\sigma$ is the standard deviation of the feature.

**Which to use?**
- Use **Z-score standardization** when your data has outliers or you don't know the range in advance. It's the default choice in most ML workflows.
- Use **Min-Max normalization** when you need values in a specific range (e.g., pixel values for images).

### Critical Rule: Scale Using Training Data Only

A common mistake is to scale the entire dataset before splitting into train and test sets. This "leaks" information from the test set into training. The correct procedure:

1. Split data into train and test sets first
2. Compute $\mu$ and $\sigma$ **from the training set only**
3. Apply those same values to scale the test set

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train data
X_test_scaled = scaler.transform(X_test)         # only transform test data
```

## Multiple Linear Regression: Beyond One Variable

### The Real World Has Many Variables

In reality, house prices don't just depend on size. They also depend on:

- Number of bedrooms
- Number of bathrooms
- Age of the house
- Neighborhood quality
- Distance to schools
- And many more...

Multiple linear regression extends our simple model to handle multiple input features.

### The Equation

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

Or in matrix notation (more compact and computationally efficient):

$$\hat{y} = X\theta$$

Where:
- $X$ is the feature matrix of shape $(m \times (n+1))$ — $m$ examples, $n$ features, plus a column of ones for the intercept
- $\theta = [\theta_0, \theta_1, \theta_2, \ldots, \theta_n]^T$ is the parameter vector

### The Column of Ones Trick

You might wonder: why do we prepend a column of ones to $X$? This is an elegant trick. Instead of writing the intercept separately, we treat it as just another weight multiplied by a feature that is always 1:

$$\hat{y} = \theta_0 \cdot 1 + \theta_1 x_1 + \theta_2 x_2 + \ldots$$

This means $\theta_0$ (the intercept) is just the weight for the constant feature $x_0 = 1$. Now all parameters are handled uniformly — the same update rule applies to all of them, including the intercept.

```python
# Without the trick: need separate handling for θ₀
# With the trick: all parameters updated the same way
X_b = np.c_[np.ones((m, 1)), X]  # prepend a column of 1s
predictions = X_b.dot(theta)      # θ₀ × 1 + θ₁ × x₁ + ...
```

### Example: House Price Prediction

$$\text{Price} = \theta_0 + \theta_1 \times \text{Size} + \theta_2 \times \text{Bedrooms} + \theta_3 \times \text{Age}$$

If we learn:

- $\theta_0 = 50,000$ (base price)
- $\theta_1 = 100$ (dollars per square foot)
- $\theta_2 = 5,000$ (dollars per bedroom)
- $\theta_3 = -2,000$ (negative because older houses are worth less)

Then a 2,000 sq ft house with 3 bedrooms that's 10 years old would be predicted at:

$$\text{Price} = 50,000 + 100(2000) + 5,000(3) + (-2,000)(10) = \$245,000$$

### Vectorized Implementation

For $m$ training examples and $n$ features, we organize our data as:

$$X = \begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\
1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)}
\end{bmatrix}, \quad
\theta = \begin{bmatrix}
\theta_0 \\
\theta_1 \\
\vdots \\
\theta_n
\end{bmatrix}$$

Then predictions for all examples are simply:

$$\hat{y} = X\theta \quad \text{(shape: } m \times 1\text{)}$$

### Cost Function for Multiple Variables

The cost function generalizes naturally:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 = \frac{1}{2m} (X\theta - y)^T(X\theta - y)$$

### Gradient Descent Update Rule

The vectorized gradient update becomes:

$$\theta := \theta - \alpha \frac{1}{m} X^T(X\theta - y)$$

This single line handles all parameters at once, including $\theta_0$, because we added the column of ones.

## The Normal Equation: Analytical Solution

### Closed-Form Solution

Instead of iteratively searching for the minimum with gradient descent, we can solve for the optimal parameters directly by setting the gradient to zero and solving algebraically.

The result is the **Normal Equation**:

$$\theta = (X^T X)^{-1} X^T y$$

### Derivation (Optional)

The cost function is:
$$J(\theta) = \frac{1}{2m}(X\theta - y)^T(X\theta - y)$$

Taking the gradient with respect to $\theta$ and setting it to zero:
$$\nabla_\theta J = \frac{1}{m}X^T(X\theta - y) = 0$$

Solving for $\theta$:
$$X^T X\theta = X^T y$$
$$\theta = (X^T X)^{-1} X^T y$$

### Gradient Descent vs. Normal Equation

| Aspect | Gradient Descent | Normal Equation |
|--------|------------------|-----------------|
| Speed for large n | Fast | Slow (computing $(X^TX)^{-1}$ is $O(n^3)$) |
| Need to choose $\alpha$ | Yes | No |
| Iterations needed | Many | None (one computation) |
| Works when $X^TX$ is singular | Yes | No |
| Works with large datasets | Yes (especially stochastic/mini-batch) | No (slow for m > 10,000) |

### When to Use Each

- **Normal Equation**: When n is small (< 10,000 features) and $(X^TX)$ is invertible
- **Gradient Descent**: When n is large, or for online learning scenarios

## Train/Test Split: Honest Evaluation

### Why You Must Split Your Data

A critical concept that beginners often miss: **you cannot evaluate your model on the same data you trained it on**. If you do, you're essentially asking "how well does the model memorize its own training data?" — not "how well does it generalize to new, unseen examples?"

Imagine you gave a student the exam questions in advance and they memorized the answers. They'd score 100%, but you'd have no idea if they actually understood the material.

**The correct workflow:**

```
All data (100%)
    │
    ├── Training set (80%) → used to fit the model (learn θ)
    │
    └── Test set (20%)    → used only at the end to evaluate generalization
```

### Overfitting: When the Model Memorizes Instead of Learns

A model is **overfitting** when it performs well on training data but poorly on test data. This happens when the model is too complex relative to the amount of training data — it learns the noise and quirks of the training set rather than the underlying pattern.

```
Training accuracy: 99%   ← looks amazing
Test accuracy:     72%   ← the model hasn't learned the real pattern
```

For linear regression, overfitting typically occurs when you have many features but few training examples. The model fits a hyperplane that passes through (or near) all training points but fails on new ones.

**Signs of overfitting:**
- Large gap between training and test error
- Test error is much higher than training error

**How to address it:**
- Get more training data
- Reduce the number of features
- Use regularization (Ridge or Lasso regression, covered in a future post)

### How to Split in Practice

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing
    random_state=42   # for reproducibility
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples:     {X_test.shape[0]}")
```

**Rule of thumb**: Use 80% training / 20% test. With very small datasets (< 500 examples), consider cross-validation instead.

## Assumptions of Linear Regression

Linear regression makes several important assumptions. Violating these can lead to unreliable results.

### 1. Linearity

**Assumption**: The relationship between X and y is linear.

**What it means**: The change in y for a unit change in x is constant across all values of x.

**How to check**:
- Scatter plots of each feature vs. the target
- Residual plots (should show no pattern)

**If violated**: Consider:
- Polynomial features (e.g., $x^2$, $x^3$)
- Logarithmic transformations
- Other non-linear models

### 2. Independence

**Assumption**: Observations are independent of each other.

**What it means**: The value of one observation doesn't depend on another.

**Common violations**:
- Time series data (observations close in time are often correlated)
- Clustered data (e.g., students from the same school)

**If violated**: Use specialized techniques like time series models or mixed-effects models.

### 3. Homoscedasticity

**Assumption**: The variance of residuals is constant across all levels of X.

**What it means**: The spread of residuals should be roughly the same whether x is small or large.

**How to check**: Residual plot (should show constant spread)

**If violated**:
- Transform the target variable (e.g., log transform)
- Use weighted least squares
- Use robust standard errors

### 4. Normality of Residuals

**Assumption**: Residuals are normally distributed.

**What it means**: When you plot the distribution of residuals, it should look like a bell curve.

**How to check**:
- Q-Q plot (quantile-quantile plot)
- Histogram of residuals
- Shapiro-Wilk test

**If violated**:
- May not be critical for prediction, but affects confidence intervals
- Consider transformations or non-parametric methods

### 5. No Multicollinearity

**Assumption**: Features are not highly correlated with each other.

**What it means**: Including height in both inches and centimeters would be perfect multicollinearity.

**How to check**:
- Correlation matrix
- Variance Inflation Factor (VIF)

**If violated**:
- Remove redundant features
- Use dimensionality reduction (PCA)
- Use regularization (Ridge, Lasso)

## Model Evaluation and Interpretation

### Evaluation Metrics

#### 1. Mean Squared Error (MSE)

$$MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

- **Interpretation**: Average squared difference between predictions and actual values
- **Units**: Squared units of the target variable
- **Lower is better**: 0 is perfect

#### 2. Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{MSE}$$

- **Interpretation**: Average magnitude of error, in the same units as the target
- **Easier to interpret than MSE**: If you're predicting house prices in dollars, RMSE is in dollars
- **Lower is better**: 0 is perfect

#### 3. Mean Absolute Error (MAE)

$$MAE = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|$$

- **Interpretation**: Average absolute difference
- **Less sensitive to outliers** than MSE/RMSE
- **Lower is better**: 0 is perfect

#### 4. R² Score (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum_{i=1}^{m}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{m}(y_i - \bar{y})^2}$$

Where $\bar{y}$ is the mean of actual values.

- **Interpretation**: Proportion of variance in y explained by the model
- **R² = 1.0**: Model explains all variance (perfect fit)
- **R² = 0.0**: Model is no better than always predicting the mean $\bar{y}$
- **R² < 0**: Model is worse than predicting the mean — something is very wrong

### Interpreting Coefficients

For the equation: $\text{Price} = 50,000 + 100 \times \text{Size} + 5,000 \times \text{Bedrooms}$

- **Intercept (50,000)**: Expected price when all features are 0 (often not meaningful in isolation)
- **Size coefficient (100)**: Each additional square foot increases price by $100, **holding bedrooms constant**
- **Bedrooms coefficient (5,000)**: Each additional bedroom increases price by $5,000, **holding size constant**

**Key point**: "Holding other variables constant" is crucial for interpretation in multiple regression.

**Important caveat when features are scaled**: If you used StandardScaler, the coefficients correspond to the scaled features, not the original units. To get back to original units, divide each coefficient by the corresponding feature's standard deviation.

### Residual Analysis

Examining residuals (errors) helps validate assumptions:

```python
residuals = y_test - predictions

# Plot 1: Residual plot — should show no pattern
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Plot 2: Distribution of residuals — should look like a bell curve
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
```

**What to look for**:
- Residuals scattered randomly around 0 (no pattern)
- Roughly normal distribution
- Constant spread across predicted values

## Conclusion

### Limitations of Linear Regression

While powerful, linear regression has limitations:

- **Assumes linearity**: Real relationships are often non-linear
- **Sensitive to outliers**: Squared errors heavily weight outliers
- **No automatic feature interactions**: Doesn't capture $x_1 \times x_2$ effects unless explicitly added
- **Can't extrapolate well**: Predictions outside training range are unreliable
- **Prone to overfitting**: With many features relative to samples

### When to Use Linear Regression

Linear regression is ideal when:

- You need **interpretability** (understanding feature importance)
- The relationship is **approximately linear**
- You have **continuous target** variables
- You want a **simple baseline** model
- **Sample size** is relatively small

For complex relationships, consider polynomial regression, regularization techniques (Ridge, Lasso), or more advanced models.

Now that you understand one of the most important algorithms in machine learning — linear regression may be simple, but it's the foundation for countless applications and more advanced techniques. In the next tutorial, we'll extend these concepts to **Logistic Regression** for classification problems.
