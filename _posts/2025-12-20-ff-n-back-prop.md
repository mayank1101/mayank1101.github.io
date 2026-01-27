---
layout: post
title: "Understanding Feed-Forward and Backpropagation: The Heart of Neural Networks"
date: 2025-12-20
series: "Deep Learning Series"
series_author: "Mayank Sharma"
series_image: "/assets/images/2025-12-20-understanding-feedforward-backpropogation/forward-backpropogation.png"
excerpt: "Learn how neural networks learn through feed-forward and backpropagation from scratch to truly understand these concepts."
---

Continuing in our Deep Learning Series, let's dive deeper into the heart of neural networks. Imagine you are learning to throw darts at a dartboard. At first, your throws are random. You miss left, then right, then too high, then too low. But each throw teaches you something. You adjust your aim slightly based on where the dart landed. That cycle **try → observe → adjust** is exactly how neural networks learn.

In deep learning, this learning happens through two tightly connected processes:

- **Feed-forward**: make a prediction
- **Backpropagation**: learn from the mistake

If you understand these two ideas deeply, *everything else in neural networks becomes easier*.

---

## Table of Contents

1. [Introduction: How Neural Networks Learn](#introduction-how-neural-networks-learn)
2. [Feed-Forward Pass: Making Predictions](#feed-forward-pass-making-predictions)
3. [Backpropagation: Learning from Mistakes](#backpropagation-learning-from-mistakes)
4. [The Mathematics Behind Backpropagation](#the-mathematics-behind-backpropagation)
5. [Building a Neural Network from Scratch](#building-a-neural-network-from-scratch)
6. [Conclusion](#conclusion)
7. [Jupyter Notebook](#jupyter-notebook)

## Introduction: How Neural Networks Learn

### The Big Picture

At it's core, a neural network is simply a **function approximator**. It learns a mapping `inputs -> outputs` by adjusting millions or even billions of small numbers called **weights** and **biases**.

In neural networks, learning happens through a repeated loop:

1. **Feed-forward pass**  
   In each iteration, in a feed-forward pass, we use current parameters to make a prediction.

2. **Backpropagation**  
   In each iteration, in backpropagation, we measure the error and adjust parameters to reduce it.

This loop runs thousands or millions of times until predictions become accurate. This is not magic. It is systematic trial, error, and correction done efficiently at scale.

### Why This Matters ?

Understanding feed-forward and backpropagation is crucial because:

- **Foundation of Deep Learning**: Every neural network from simple ones to GPT-4 uses these mechanisms
- **Debugging Intuition**: Knowing how networks learn helps you diagnose and fix training problems
- **Architecture Design**: Understanding gradients helps you design better neural architectures
- **Research & Innovation**: Many breakthroughs in AI come from improving these fundamental processes

## Feed-Forward Pass: Making Predictions

### What is Feed-Forward?

The feed-forward pass is the process of taking input data and passing it through the network's layers to produce an output. It's called `feed-forward` because information flows in one direction: from `input → hidden layers → output`. No corrections happen here. The network simply answers: *“Given what I know right now, what’s my prediction?”*

Think of it like an assembly line: raw input enters, transformations happen layer by layer, and a final product comes out.

Now, let's break down a simple neural network with one hidden layer:

```
Input Layer (3 neurons)
        ↓
    [Weights W1]
        ↓
    [Biases b1]
        ↓
Hidden Layer (4 neurons) → [Activation Function: ReLU]
        ↓
    [Weights W2]
        ↓
    [Biases b2]
        ↓
Output Layer (2 neurons) → [Activation Function: Softmax]
        ↓
    Prediction
```

Each arrow represents a matrix multiplication plus a bias.

---

### Let's Understand the Step-by-Step Process: Feed-Forward Process

#### Step 1: Input Layer

The process begins with your input data. For a network classifying handwritten digits, this might be a 28×28 pixel image flattened into a 784 dimensional vector.

```python
# Example input: a simple 3-dimensional vector
input_data = [0.5, 0.2, 0.8]
```

#### Step 2: First Linear Transformation

The input is first transformed using weights and biases. Each neuron computes a weighted sum:

$$z^{[1]} = W^{[1]} \cdot x + b^{[1]}$$

Where:

- $W^{[1]}$ is the weight matrix connecting input to hidden layer
- $x$ is the input vector
- $b^{[1]}$ is the bias vector
- $z^{[1]}$ is the pre-activation output

**What's happening here?** Each neuron in the hidden layer computes a weighted sum of all input features. The weights determine how much each input contributes to each hidden neuron. The biases add a constant offset to each hidden neuron.

This answers:
“How important is each input feature to this neuron?”

```python
# If input has 3 features and hidden layer has 4 neurons:
# W1 shape: (4, 3)
# b1 shape: (4,)
# z1 = W1 @ input + b1 → shape: (4,)
```

#### Step 3: Apply Activation Function

So, activation functions introduce non-linearity. Without them, stacking layers would still behave like a single linear model.

$$a^{[1]} = g(z^{[1]})$$

Let's look at some common activation functions:

- **ReLU**: $f(x) = \max(0, x)$
- **Sigmoid**: $f(x) = \frac{1}{1 + e^{-x}}$
- **Tanh**: $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

#### Step 4: Subsequent Layers

This process repeats for each layer:

$$z^{[2]} = W^{[2]} \cdot a^{[1]} + b^{[2]}$$
$$a^{[2]} = g(z^{[2]})$$

For the output layer with classification tasks, we typically use:

- **Softmax** for multi-class classification (outputs probabilities that sum to 1)
- **Sigmoid** for binary classification
- **Linear** (no activation) for regression

#### Step 5: Final Prediction

The output layer produces the network's prediction:

```python
# For a 2-class classification with softmax
output = [0.8, 0.2]  # 80% confidence for class 0, 20% for class 1
prediction = 0  # Predicted class
```

### The Feed-Forward Formula (Complete)

Now let's take an example of a 3-layer network. For a 3-layer network `(input → hidden → output)`:

$$
\begin{align}
z^{[1]} &= W^{[1]} \cdot x + b^{[1]} \\
a^{[1]} &= \text{ReLU}(z^{[1]}) \\
z^{[2]} &= W^{[2]} \cdot a^{[1]} + b^{[2]} \\
\hat{y} &= \text{softmax}(z^{[2]})
\end{align}
$$

Intuitively, you can think of each layer as asking increasingly sophisticated questions:

- **Input Layer**: "What are the raw features?" (pixel values, word embeddings, etc.)
- **Hidden Layers**: "What patterns exist?" (edges, shapes, concepts)
- **Output Layer**: "What's the final answer?" (cat, dog, bird)

Each neuron is like a feature detector that becomes activated when it sees certain patterns in the previous layer.

## Backpropagation: Learning from Mistakes

### The Learning Problem

Once the feed-forward pass is complete, we have a prediction. But how do we improve the network's performance? This is where backpropagation (short for "backward propagation of errors") comes in. **The Goal** is to adjust every weight and bias in the network so that we can minimize the difference between predictions and actual values.

### Loss Functions: Measuring Error

Before we can improve, we need to measure how wrong we are, this is where loss function comes in. A loss function quantifies the error:

#### Mean Squared Error (MSE) - For Regression

$$L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

#### Cross-Entropy Loss - For Classification

$$L = -\sum_{i=1}^{C} y_i \cdot \log(\hat{y}_i)$$

Where:

- $y_i$ is the true value
- $\hat{y}_i$ is the predicted value
- $N$ is the number of samples
- $C$ is the number of classes

**Key Point**: **Lower loss = Better performance**

### What's the Intuition Behind Backpropagation ?

Imagine you're hiking in foggy mountains and want to reach the valley (minimum loss). You can't see far, but you can feel the slope beneath your feet. Backpropagation is like feeling which direction is downhill and taking a step in that direction.

Mathematically, backpropagation computes the **gradient** of the loss with respect to each parameter, essentially, it tells us "if I change this weight slightly, how much will the loss change?"

### The Chain Rule: The Mathematical Engine

Backpropagation is fundamentally an application of the chain rule from calculus. If we have nested functions:

$$y = f(g(h(x)))$$

The derivative is:
$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}$$

In neural networks, this becomes a cascade of derivatives flowing backward through the layers.

## The Mathematics Behind Backpropagation

### Setting Up the Problem

Let's work through backpropagation for a simple 2-layer network:

**Network Structure**:

- Input: $x$ (3 features)
- Hidden layer: 4 neurons with ReLU
- Output: 2 neurons with softmax
- Loss: Cross-entropy

**Forward Pass Summary**:

$$
\begin{align}
z^{[1]} &= W^{[1]} \cdot x + b^{[1]} \quad &\text{(Hidden pre-activation)} \\
a^{[1]} &= \text{ReLU}(z^{[1]}) \quad &\text{(Hidden activation)} \\
z^{[2]} &= W^{[2]} \cdot a^{[1]} + b^{[2]} \quad &\text{(Output pre-activation)} \\
\hat{y} &= \text{softmax}(z^{[2]}) \quad &\text{(Final prediction)} \\
L &= -\sum y \cdot \log(\hat{y}) \quad &\text{(Loss)}
\end{align}
$$

### Backward Pass: Computing Gradients

Now we compute how the loss changes with respect to each parameter, starting from the output and working backward.

#### Step 1: Gradient at Output Layer

For cross-entropy loss with softmax, the gradient simplifies beautifully:

$$\frac{\partial L}{\partial z^{[2]}} = \hat{y} - y$$

This is remarkably elegant: the gradient is simply the difference between prediction and truth!

```python
# If prediction is [0.7, 0.3] and truth is [1, 0]:
dz2 = [0.7, 0.3] - [1, 0] = [-0.3, 0.3]
```

#### Step 2: Gradients for Output Layer Parameters

**Weight gradient**:
$$\frac{\partial L}{\partial W^{[2]}} = \frac{\partial L}{\partial z^{[2]}} \cdot (a^{[1]})^T$$

**Bias gradient**:
$$\frac{\partial L}{\partial b^{[2]}} = \frac{\partial L}{\partial z^{[2]}}$$

```python
# Computing gradients
dW2 = dz2 @ a1.T  # Outer product
db2 = dz2.sum(axis=0)  # Sum over batch dimension
```

#### Step 3: Gradient Flowing to Hidden Layer

The gradient "flows back" through the weights:

$$\frac{\partial L}{\partial a^{[1]}} = (W^{[2]})^T \cdot \frac{\partial L}{\partial z^{[2]}}$$

This is where we see the chain rule in action, wherethe gradient from the output layer is propagated backward through the weight matrix.

#### Step 4: Gradient Through ReLU

ReLU derivative is simple:
$$\frac{d}{dz}\text{ReLU}(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

```python
# Element-wise multiplication with ReLU derivative
dz1 = da1 * (z1 > 0)  # Derivative is 0 where z1 was negative
```

#### Step 5: Gradients for Hidden Layer Parameters

$$\frac{\partial L}{\partial W^{[1]}} = \frac{\partial L}{\partial z^{[1]}} \cdot x^T$$
$$\frac{\partial L}{\partial b^{[1]}} = \frac{\partial L}{\partial z^{[1]}}$$

### The Complete Backpropagation Algorithm

**Forward Pass** (compute and cache):

- $z^{[1]} = W^{[1]} \cdot x + b^{[1]}$
- $a^{[1]} = \text{ReLU}(z^{[1]})$
- $z^{[2]} = W^{[2]} \cdot a^{[1]} + b^{[2]}$
- $\hat{y} = \text{softmax}(z^{[2]})$
- $L = -\sum y \log(\hat{y})$

**Backward Pass** (compute gradients):

- $\frac{\partial L}{\partial z^{[2]}} = \hat{y} - y$
- $\frac{\partial L}{\partial W^{[2]}} = \frac{\partial L}{\partial z^{[2]}} \cdot (a^{[1]})^T$
- $\frac{\partial L}{\partial b^{[2]}} = \frac{\partial L}{\partial z^{[2]}}$
- $\frac{\partial L}{\partial a^{[1]}} = (W^{[2]})^T \cdot \frac{\partial L}{\partial z^{[2]}}$
- $\frac{\partial L}{\partial z^{[1]}} = \frac{\partial L}{\partial a^{[1]}} \odot (z^{[1]} > 0)$
- $\frac{\partial L}{\partial W^{[1]}} = \frac{\partial L}{\partial z^{[1]}} \cdot x^T$
- $\frac{\partial L}{\partial b^{[1]}} = \frac{\partial L}{\partial z^{[1]}}$

**Parameter Update** (gradient descent):

- $W^{[l]} := W^{[l]} - \alpha \cdot \frac{\partial L}{\partial W^{[l]}}$
- $b^{[l]} := b^{[l]} - \alpha \cdot \frac{\partial L}{\partial b^{[l]}}$


Where $\alpha$ is the learning rate.

### Why Backpropagation is Efficient

Computing gradients naively would require one forward pass for each parameter (millions of passes!). Backpropagation is brilliant because:

1. **One backward pass computes all gradients** simultaneously
2. **Reuses computations** from the forward pass
3. **Uses the chain rule** to decompose complex derivatives into simple ones

This makes training even massive networks computationally feasible.

## Building a Neural Network from Scratch

Let's implement a complete neural network with feed-forward and backpropagation from scratch to truly understand these concepts.

### Problem: Binary Classification

We'll build a network to classify points as belonging to one of two classes based on their coordinates.

### Complete Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

class NeuralNetwork:
    """
    A simple feedforward neural network with one hidden layer.

    Architecture:
    - Input layer: n_input neurons
    - Hidden layer: n_hidden neurons with ReLU activation
    - Output layer: 1 neuron with sigmoid activation

    This network performs binary classification.
    """

    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.01):
        """
        Initialize the neural network with random weights.

        Args:
            n_input: Number of input features
            n_hidden: Number of neurons in hidden layer
            n_output: Number of output neurons (1 for binary classification)
            learning_rate: Learning rate for gradient descent
        """
        self.lr = learning_rate

        # Initialize weights with small random values (Xavier initialization)
        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2. / n_input)
        self.b1 = np.zeros((1, n_hidden))

        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2. / n_hidden)
        self.b2 = np.zeros((1, n_output))

        # Cache for backward pass
        self.cache = {}

    def relu(self, Z):
        """ReLU activation function."""
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        """Derivative of ReLU function."""
        return (Z > 0).astype(float)

    def sigmoid(self, Z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # Clip to prevent overflow

    def sigmoid_derivative(self, A):
        """Derivative of sigmoid function."""
        return A * (1 - A)

    def forward(self, X):
        """
        Perform forward propagation.

        Args:
            X: Input data of shape (batch_size, n_input)

        Returns:
            A2: Output predictions of shape (batch_size, n_output)
        """
        # Hidden layer
        Z1 = X @ self.W1 + self.b1  # Linear transformation
        A1 = self.relu(Z1)           # ReLU activation

        # Output layer
        Z2 = A1 @ self.W2 + self.b2  # Linear transformation
        A2 = self.sigmoid(Z2)         # Sigmoid activation

        # Cache values needed for backward pass
        self.cache = {
            'X': X,
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2,
            'A2': A2
        }

        return A2

    def compute_loss(self, Y, Y_pred):
        """
        Compute binary cross-entropy loss.

        Args:
            Y: True labels of shape (batch_size, 1)
            Y_pred: Predicted labels of shape (batch_size, 1)

        Returns:
            loss: Scalar loss value
        """
        m = Y.shape[0]

        # Binary cross-entropy loss
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        loss = -np.mean(Y * np.log(Y_pred + epsilon) +
                       (1 - Y) * np.log(1 - Y_pred + epsilon))

        return loss

    def backward(self, Y):
        """
        Perform backpropagation to compute gradients.

        Args:
            Y: True labels of shape (batch_size, 1)
        """
        m = Y.shape[0]  # Batch size

        # Retrieve cached values from forward pass
        X = self.cache['X']
        Z1 = self.cache['Z1']
        A1 = self.cache['A1']
        A2 = self.cache['A2']

        # Output layer gradients
        # For binary cross-entropy with sigmoid, the gradient simplifies to:
        dZ2 = A2 - Y  # Shape: (m, 1)

        # Gradients for W2 and b2
        dW2 = (1/m) * (A1.T @ dZ2)  # Shape: (n_hidden, 1)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)  # Shape: (1, 1)

        # Hidden layer gradients
        # Gradient flowing back from output layer
        dA1 = dZ2 @ self.W2.T  # Shape: (m, n_hidden)

        # Apply ReLU derivative
        dZ1 = dA1 * self.relu_derivative(Z1)  # Shape: (m, n_hidden)

        # Gradients for W1 and b1
        dW1 = (1/m) * (X.T @ dZ1)  # Shape: (n_input, n_hidden)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)  # Shape: (1, n_hidden)

        # Update parameters
        self.W2 -= self.lr * dW2  
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, Y, epochs=1000, verbose=True):
        """
        Train the neural network.

        Args:
            X: Training data of shape (m, n_input)
            Y: Training labels of shape (m, 1)
            epochs: Number of training epochs
            verbose: Whether to print training progress

        Returns:
            loss_history: List of loss values over epochs
        """
        loss_history = []

        for epoch in range(epochs):
            # Forward pass
            Y_pred = self.forward(X)

            # Compute loss
            loss = self.compute_loss(Y, Y_pred)
            loss_history.append(loss)

            # Backward pass (compute gradients and update parameters)
            self.backward(Y)

            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                accuracy = self.evaluate(X, Y)
                print(f'Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}')

        return loss_history

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Input data of shape (m, n_input)

        Returns:
            predictions: Binary predictions (0 or 1)
        """
        Y_pred = self.forward(X)
        return (Y_pred > 0.5).astype(int)

    def evaluate(self, X, Y):
        """
        Evaluate accuracy on a dataset.

        Args:
            X: Input data of shape (m, n_input)
            Y: True labels of shape (m, 1)

        Returns:
            accuracy: Accuracy score between 0 and 1
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == Y)
        return accuracy


def plot_decision_boundary(model, X, Y, title="Decision Boundary"):
    """
    Plot the decision boundary learned by the model.
    """
    # Set min and max values with padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Generate a grid of points
    h = 0.01  # Step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict for all points in the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap='RdYlBu',
                edgecolors='black', s=100, linewidths=1.5)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(label='Predicted Class')
    plt.grid(True, alpha=0.3)


def visualize_network_learning(loss_history):
    """
    Visualize the learning process by plotting loss over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to see convergence better


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("FEEDFORWARD AND BACKPROPAGATION TUTORIAL")
    print("=" * 60)

    # 1. Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    print("-" * 60)

    # Create a non-linear dataset (two moons)
    X, Y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    Y = Y.reshape(-1, 1)  # Reshape to column vector

    # Split into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Output classes: {len(np.unique(Y))}")

    # Visualize the dataset
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train.ravel(),
                cmap='RdYlBu', edgecolors='black', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Training Dataset: Two Moons')
    plt.colorbar(label='Class')
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2. Initialize neural network
    print("\n2. Initializing neural network...")
    print("-" * 60)

    n_input = X_train.shape[1]  # 2 features
    n_hidden = 10               # 10 hidden neurons
    n_output = 1                # 1 output (binary classification)
    learning_rate = 0.1

    model = NeuralNetwork(
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_output,
        learning_rate=learning_rate
    )

    print(f"Network architecture: {n_input} → {n_hidden} → {n_output}")
    print(f"Total parameters: {(n_input * n_hidden + n_hidden) + (n_hidden * n_output + n_output)}")
    print(f"Learning rate: {learning_rate}")

    # 3. Train the network
    print("\n3. Training the network...")
    print("-" * 60)

    loss_history = model.train(
        X_train, Y_train,
        epochs=1000,
        verbose=True
    )

    # 4. Evaluate on test set
    print("\n4. Evaluating on test set...")
    print("-" * 60)

    train_accuracy = model.evaluate(X_train, Y_train)
    test_accuracy = model.evaluate(X_test, Y_test)

    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # 5. Visualize results
    print("\n5. Visualizing results...")
    print("-" * 60)

    # Plot learning curve
    visualize_network_learning(loss_history)
    plt.show()

    # Plot decision boundary on training data
    plot_decision_boundary(model, X_train, Y_train,
                          title="Decision Boundary on Training Data")
    plt.show()

    # Plot decision boundary on test data
    plot_decision_boundary(model, X_test, Y_test,
                          title="Decision Boundary on Test Data")
    plt.show()

    # 6. Test on specific examples
    print("\n6. Testing on specific examples...")
    print("-" * 60)

    test_points = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [-1.0, 0.5],
        [0.5, -0.5]
    ])

    predictions = model.predict(test_points)
    probabilities = model.forward(test_points)

    for i, point in enumerate(test_points):
        print(f"Point {point}: Predicted class = {predictions[i][0]}, "
              f"Probability = {probabilities[i][0]:.4f}")

    print("\n" + "=" * 60)
    print("TUTORIAL COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Forward pass: Input → Hidden (ReLU) → Output (Sigmoid)")
    print("2. Backward pass: Compute gradients using chain rule")
    print("3. Update parameters: W := W - α * dW")
    print("4. Repeat until convergence!")
```

### Code Walkthrough

#### 1. Network Architecture

Our `NeuralNetwork` class implements:

- **Input layer**: 2 features (x, y coordinates)
- **Hidden layer**: 10 neurons with ReLU activation
- **Output layer**: 1 neuron with sigmoid activation

#### 2. Weight Initialization

We use Xavier initialization to set initial weights:

```python
W = np.random.randn(n_in, n_out) * np.sqrt(2. / n_in)
```

This helps prevent vanishing/exploding gradients at the start of training.

#### 3. Forward Pass Implementation

```python
Z1 = X @ W1 + b1      # Linear transformation
A1 = relu(Z1)          # Non-linear activation
Z2 = A1 @ W2 + b2     # Linear transformation
A2 = sigmoid(Z2)       # Final activation
```

#### 4. Backward Pass Implementation

```python
# Output layer
dZ2 = A2 - Y                    # Gradient at output
dW2 = (1/m) * (A1.T @ dZ2)     # Weight gradient
db2 = (1/m) * sum(dZ2)         # Bias gradient

# Hidden layer
dA1 = dZ2 @ W2.T               # Gradient flowing back
dZ1 = dA1 * relu'(Z1)          # Through activation
dW1 = (1/m) * (X.T @ dZ1)      # Weight gradient
db1 = (1/m) * sum(dZ1)         # Bias gradient
```

#### 5. Parameter Update

```python
W -= learning_rate * dW
b -= learning_rate * db
```

## Conclusion

So, now that you've mastered the fundamental mechanisms that power all neural networks from the simplest to the most sophisticated AI systems in use today. You've seen how:

- **Feed-forward pass** transforms input through layers to produce predictions
- **Backpropagation** efficiently computes gradients using the chain rule
- **Gradient descent** uses these gradients to incrementally improve parameters
- **The training loop** repeats forward → compute loss → backward → update thousands of times

What's remarkable is that these simple principles of weight initialization, matrix multiplication, activation functions, and gradient descent can scale to create systems that can:

- Generate human-like text (GPT-5)
- Recognize objects in images (CNNs)
- Translate languages (Transformers)
- Play complex games (AlphaGo)

## Jupyter Notebook

For hands-on practice, check out this companion notebook - [Feedforward and Backpropogation From Scratch](https://colab.research.google.com/drive/1rQ45fgE9QgEI8CBxNcILCWOrD6BL20_x?usp=sharing)