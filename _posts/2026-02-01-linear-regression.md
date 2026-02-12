---
layout: post
title: "Linear Regression: From Simple to Multiple Regression"
date: 2026-02-01
series: "Machine Learning Series"
series_author: "Mayank Sharma"
series_image: "/assets/images/2026-02-01-linear-regression/linear-regression.png"
excerpt: "Master linear regression from scratch, understanding simple and multiple regression, cost functions, and gradient descent."
---

# Linear Regression: From Simple to Multiple Regression

Continuing in our Machine Learning Series, today we dive deep into linear regression, the cornerstone of predictive modeling. Imagine you're a real estate agent trying to estimate house prices. You notice that larger houses generally sell for more money. Intuitively, you understand there's a relationship between size and price, but can you quantify it? Can you predict the price of a 2,000 square foot house based on past sales data? This is exactly what linear regression does, it finds the mathematical relationship between variables and uses it to make predictions.

## Table of Contents

1. [Introduction: The Power of Linear Relationships](#introduction-the-power-of-linear-relationships)
2. [Simple Linear Regression: The Foundation](#simple-linear-regression-the-foundation)
3. [The Mathematics Behind the Line](#the-mathematics-behind-the-line)
4. [Cost Function: Measuring Our Mistakes](#cost-function-measuring-our-mistakes)
5. [Gradient Descent: Learning the Best Fit](#gradient-descent-learning-the-best-fit)
6. [Multiple Linear Regression: Beyond One Variable](#multiple-linear-regression-beyond-one-variable)
7. [The Normal Equation: Analytical Solution](#the-normal-equation-analytical-solution)
8. [Assumptions of Linear Regression](#assumptions-of-linear-regression)
9. [Implementation from Scratch](#implementation-from-scratch)
10. [Model Evaluation and Interpretation](#model-evaluation-and-interpretation)
11. [Conclusion](#conclusion)
12. [Jupyter Notebook](#jupyter-notebook)

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
- $\frac{1}{2m}$ is for mathematical convenience (the 1/2 makes derivatives cleaner)

### Why Square the Errors?

Squaring serves several purposes:

1. **Eliminates negative values**: All errors contribute positively
2. **Emphasizes larger errors**: An error of 10 contributes 100, while an error of 1 contributes only 1
3. **Mathematical properties**: Creates a smooth, differentiable function, that is very important for optimization
4. **Statistical foundation**: Under certain assumptions, minimizing MSE gives the maximum likelihood estimate

### Visualizing the Cost Function

If we plot $J(\theta_0, \theta_1)$ as a function of the parameters, we get a bowl-shaped surface in 3D (or a parabola in 2D if we fix one parameter). Our goal is to find the bottom of this bowl, the point where the cost is minimized.

## Gradient Descent: Learning the Best Fit

### The Optimization Problem

We want to find the values of $\theta_0$ and $\theta_1$ that minimize the cost function $J$. This is an optimization problem. While there's an analytical solution (which we'll cover later), gradient descent provides a general approach that works for more complex models too.

### The Intuition: Hiking Down a Mountain

Imagine you're on a foggy mountain and want to reach the valley (minimum). You can't see the whole landscape, but you can feel the slope beneath your feet. The strategy:

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

### Computing the Gradients

Taking the partial derivatives of the cost function:

$$\frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) = \frac{1}{m} \sum_{i=1}^{m} ((\theta_0 + \theta_1 x_i) - y_i)$$

$$\frac{\partial J}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \cdot x_i = \frac{1}{m} \sum_{i=1}^{m} ((\theta_0 + \theta_1 x_i) - y_i) \cdot x_i$$

**Intuition**:
- For $\theta_0$: The average error tells us how much to adjust the intercept
- For $\theta_1$: The average of (error × input) tells us how much to adjust the slope

### The Learning Rate: A Balancing Act

The learning rate $\alpha$ is crucial:

- **Too small**: Learning is painfully slow; might need millions of iterations
- **Too large**: Might overshoot the minimum or even diverge
- **Just right**: Converges efficiently to the minimum

Typical values: 0.001, 0.01, 0.1 (depends on data scaling)

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

$$\hat{y} = \theta^T X$$

Where:

- $X = [1, x_1, x_2, \ldots, x_n]^T$ is the feature vector (we add a 1 for the intercept term)
- $\theta = [\theta_0, \theta_1, \theta_2, \ldots, \theta_n]^T$ is the parameter vector

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

For m training examples and n features, we organize our data as:

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

$$\hat{y} = X\theta$$

### Cost Function for Multiple Variables

The cost function generalizes naturally:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 = \frac{1}{2m} (X\theta - y)^T(X\theta - y)$$

### Gradient Descent Update Rule

The vectorized gradient update becomes:

$$\theta := \theta - \alpha \frac{1}{m} X^T(X\theta - y)$$

This single line replaces the need for separate updates for each parameter!

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

## Implementation from Scratch

### Using NumPy: Complete Implementation

Let's build linear regression from scratch to deeply understand the mechanics:

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using gradient descent.

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

    def fit(self, X, y):
        """
        Fit linear regression model using gradient descent.

        Parameters:
        -----------
        X : array-like, shape (m, n)
            Training features
        y : array-like, shape (m,)
            Target values
        """
        # Add intercept term (column of ones)
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]  # X_b has shape (m, n+1)

        # Initialize parameters randomly
        self.theta = np.random.randn(n + 1, 1)

        # Gradient descent
        for iteration in range(self.n_iterations):
            # Compute predictions
            predictions = X_b.dot(self.theta)

            # Compute errors
            errors = predictions - y.reshape(-1, 1)

            # Compute cost (for tracking)
            cost = (1 / (2 * m)) * np.sum(errors ** 2)
            self.cost_history.append(cost)

            # Compute gradients
            gradients = (1 / m) * X_b.T.dot(errors)

            # Update parameters
            self.theta -= self.learning_rate * gradients

            # Print progress every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.4f}")

    def predict(self, X):
        """
        Make predictions using the learned parameters.

        Parameters:
        -----------
        X : array-like, shape (m, n)
            Features to predict on

        Returns:
        --------
        predictions : array, shape (m,)
            Predicted values
        """
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        return X_b.dot(self.theta).flatten()

    def plot_cost_history(self):
        """Plot the cost function over iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost J(θ)')
        plt.title('Cost Function Over Iterations')
        plt.grid(True)
        plt.show()


class LinearRegressionNormalEq:
    """
    Linear Regression using the Normal Equation (closed-form solution).
    """

    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        """
        Fit linear regression using the normal equation.

        Parameters:
        -----------
        X : array-like, shape (m, n)
            Training features
        y : array-like, shape (m,)
            Target values
        """
        # Add intercept term
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]

        # Normal equation: θ = (X^T X)^(-1) X^T y
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        """
        Make predictions using the learned parameters.

        Parameters:
        -----------
        X : array-like, shape (m, n)
            Features to predict on

        Returns:
        --------
        predictions : array, shape (m,)
            Predicted values
        """
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        return X_b.dot(self.theta)


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    m = 100  # number of examples
    X = 2 * np.random.rand(m, 1)  # random values between 0 and 2
    y = 4 + 3 * X + np.random.randn(m, 1)  # y = 4 + 3x + noise

    # Flatten y to 1D
    y = y.flatten()

    print("=" * 50)
    print("Training with Gradient Descent")
    print("=" * 50)

    # Train using gradient descent
    model_gd = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model_gd.fit(X, y)

    print("\nLearned parameters (Gradient Descent):")
    print(f"θ₀ (intercept) = {model_gd.theta[0][0]:.4f}")
    print(f"θ₁ (slope) = {model_gd.theta[1][0]:.4f}")

    # Train using normal equation
    print("\n" + "=" * 50)
    print("Training with Normal Equation")
    print("=" * 50)

    model_ne = LinearRegressionNormalEq()
    model_ne.fit(X, y)

    print("\nLearned parameters (Normal Equation):")
    print(f"θ₀ (intercept) = {model_ne.theta[0]:.4f}")
    print(f"θ₁ (slope) = {model_ne.theta[1]:.4f}")

    # Make predictions
    X_test = np.array([[0], [2]])
    predictions_gd = model_gd.predict(X_test)
    predictions_ne = model_ne.predict(X_test)

    # Visualize
    plt.figure(figsize=(12, 5))

    # Plot 1: Data and regression line
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, label='Training data')
    plt.plot(X_test, predictions_gd, 'r-', linewidth=2, label='Gradient Descent')
    plt.plot(X_test, predictions_ne, 'g--', linewidth=2, label='Normal Equation')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression: Predictions')
    plt.legend()
    plt.grid(True)

    # Plot 2: Cost history
    plt.subplot(1, 2, 2)
    plt.plot(model_gd.cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost J(θ)')
    plt.title('Cost Function Convergence')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
```

### Using Scikit-Learn

For production use, we use scikit-learn which provides an optimized implementation:

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficients: {model.coef_}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
```

## Model Evaluation and Interpretation

### Evaluation Metrics

#### 1. Mean Squared Error (MSE)

$$MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

- **Interpretation**: Average squared difference between predictions and actual values
- **Units**: Squared units of the target variable
- **Lower is better**: 0 is perfect

#### 2. Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{MSE}$$

- **Interpretation**: Average magnitude of error
- **Units**: Same as target variable (easier to interpret than MSE)
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
- **Range**: 0 to 1 (can be negative for very poor models)
- **1.0 is perfect**: Model explains all variance
- **0.0 means**: Model is no better than just predicting the mean

### Interpreting Coefficients

For the equation: $\text{Price} = 50,000 + 100 \times \text{Size} + 5,000 \times \text{Bedrooms}$

- **Intercept (50,000)**: Expected price when all features are 0 (often not meaningful)
- **Size coefficient (100)**: Each additional square foot increases price by $100, **holding bedrooms constant**
- **Bedrooms coefficient (5,000)**: Each additional bedroom increases price by $5,000, **holding size constant**

**Key point**: "Holding other variables constant" is crucial for interpretation in multiple regression.

### Residual Analysis

Examining residuals (errors) helps validate assumptions:

```python
# Calculate residuals
residuals = y_test - predictions

# Plot 1: Residual plot
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Plot 2: Distribution of residuals
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
```

**What to look for**:
- Residuals scattered randomly around 0 (no pattern)
- Roughly normal distribution
- Constant spread across predicted values

## Conclusion:

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

For complex relationships, consider polynomial regression, regularization techniques, or more advanced models.

Now that you understand one of the most important algorithms in machine learning. Linear regression may be simple, but it's the foundation for countless applications and more advanced techniques. In the next tutorial, we'll extend these concepts to **Logistic Regression** for classification problems.

## Jupyter Notebook

For hands-on practice, check out the companion notebooks -  [Linear Regression Tutorial](https://drive.google.com/file/d/1nl8lb0FZRNFG9L_3LmhMVLxdgbv-vvVh/view?usp=sharing)