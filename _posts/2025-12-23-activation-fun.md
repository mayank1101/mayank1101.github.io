---
layout: post
title: "Understanding Activation Functions: The Non-Linear Magic in Neural Networks"
date: 2025-12-23
series: "Deep Learning Series"
series_author: "Mayank Sharma"
series_image: "/assets/images/2025-12-23-activation-fun/activation-comparison.png"
excerpt: "Master activation functions from Sigmoid to GELU, understand why neural networks need non-linearity and how to choose the right activation for your model."
---

Continuing in our series on deep learning, let's dive into the fascinating world of activation functions. So, imagine a neural network as a team of decision-makers, without activation functions, every neuron would simply be a yes-man—taking inputs, multiplying them by weights, adding them up, and passing them along unchanged. The result? No matter how many layers you stack, the entire network would collapse into a single linear transformation. Activation functions are what give neurons their personality, the ability to say "yes," "no," or "maybe" in complex, non-linear ways.

## Table of Contents

1. [Introduction: Why Non-Linearity Matters](#introduction-why-non-linearity-matters)
2. [The Mathematics of Activation Functions](#the-mathematics-of-activation-functions)
3. [Classic Activations: Sigmoid and Tanh](#classic-activations-sigmoid-and-tanh)
4. [The ReLU Revolution](#the-relu-revolution)
5. [ReLU Variants: Fixing the Dying ReLU Problem](#relu-variants-fixing-the-dying-relu-problem)
6. [Modern Activations: GELU, Swish, and Mish](#modern-activations-gelu-swish-and-mish)
7. [Output Activations: Softmax and It's Friends](#output-activations-softmax-and-its-friends)
8. [Specialized Activations](#specialized-activations)
9. [Choosing the Right Activation Function](#choosing-the-right-activation-function)
10. [Implementation Guide](#implementation-guide)
11. [Conclusion](#conclusion)
12. [Further Reading and Resources](#further-reading-and-resources)

## Introduction: Why Non-Linearity Matters

### The Linearity Problem

Let us consider a neural network without activation functions. Each layer performs a linear transformation:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

Without an activation function, $\mathbf{a}^{[l]} = \mathbf{z}^{[l]}$, and composing multiple layers gives:

$$\mathbf{z}^{[L]} = \mathbf{W}^{[L]} \mathbf{W}^{[L-1]} \cdots \mathbf{W}^{[1]} \mathbf{x} + \text{bias terms}$$

This is just one big matrix multiplication! No matter how deep your network, it's equivalent to a single linear layer. You cannot learn complex patterns like:

- Recognizing faces in images
- Understanding sarcasm in text
- Playing chess or Go

### The Universal Approximation Theorem

The universal approximation theorem states that a neural network with at least one hidden layer and a non-linear activation function can approximate any continuous function to arbitrary precision (given enough neurons).

The key thing to remember here is **non-linearity enables complexity**. Activation functions introduce the curves, bends, and twists that allow networks to model the intricate patterns in real-world data.

### What Makes a Good Activation Function?

Generally speaking, an ideal activation function should have:

1. **Non-linearity**: The whole point—enables learning complex functions
2. **Differentiability**: Required for backpropagation (or at least subdifferentiable)
3. **Computational efficiency**: Called billions of times during training
4. **Non-vanishing gradients**: Gradients should flow during backpropagation
5. **Zero-centered output**: Helps with optimization (not strictly required)
6. **Bounded or unbounded**: Depending on use case

## The Mathematics of Activation Functions

### Formal Definition

Mathematically, an activation function $\sigma: \mathbb{R} \to \mathbb{R}$ takes a scalar input and produces a scalar output. In a neural network layer:

$$\mathbf{a}^{[l]} = \sigma(\mathbf{z}^{[l]}) = \sigma(\mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]})$$

The function is applied element-wise to each component of the pre-activation vector $\mathbf{z}^{[l]}$.

### Key Properties to Analyze

For each activation function, we need to examine:

1. **Formula**: The mathematical definition
2. **Range**: Output bounds (e.g., $(0, 1)$, $(-1, 1)$, $(-\infty, \infty)$)
3. **Derivative**: For backpropagation
4. **Saturation**: Where gradients become very small
5. **Zero-centered**: Whether outputs are centered around zero

### The Gradient Flow Problem (Important!)

During backpropagation, gradients flow backward through the network:

$$\frac{\partial L}{\partial \mathbf{z}^{[l]}} = \frac{\partial L}{\partial \mathbf{a}^{[l]}} \odot \sigma'(\mathbf{z}^{[l]})$$

If $\sigma'(\mathbf{z}^{[l]})$ is very small (saturated activation), the gradients will vanish. If it's very large, gradients will explode. The derivative of the activation function directly impacts how well gradients flow.

## Classic Activations: Sigmoid and Tanh

### Sigmoid Function

The sigmoid function, also called the logistic function, was the workhorse of early neural networks.

**Formula**:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Derivative**:
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Range**: $(0, 1)$

**Properties**:
- Smooth, differentiable everywhere
- Outputs can be interpreted as probabilities
- Squashes any input to $(0, 1)$

**The Problems with Sigmoid**:

1. **Vanishing Gradients**: For large $|x|$, $\sigma'(x) \approx 0$. In deep networks, gradients multiply through layers, shrinking exponentially.

2. **Not Zero-Centered**: Outputs are always positive $(0, 1)$. This causes zig-zagging during gradient descent because all gradients for weights have the same sign.

3. **Computationally Expensive**: Exponential computation is slower than simple operations.

**Example**:
```python
import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid."""
    s = sigmoid(x)
    return s * (1 - s)

# Example values
x = np.array([-5, -2, 0, 2, 5])
print(f"x:        {x}")
print(f"sigmoid:  {sigmoid(x).round(4)}")
print(f"gradient: {sigmoid_derivative(x).round(4)}")

# Output:
# x:        [-5 -2  0  2  5]
# sigmoid:  [0.0067 0.1192 0.5    0.8808 0.9933]
# gradient: [0.0066 0.105  0.25   0.105  0.0066]
```

Notice how the gradient at $x = \pm 5$ is only 0.0066, it's almost like no learning happens for saturated neurons.

### Tanh (Hyperbolic Tangent)

Tanh addressed sigmoid's centering problem while keeping the smooth, bounded properties.

**Formula**:
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

**Derivative**:
$$\tanh'(x) = 1 - \tanh^2(x)$$

**Range**: $(-1, 1)$

**Relationship to Sigmoid**:
$$\tanh(x) = 2\sigma(2x) - 1$$

So we can say that Tanh is just a scaled and shifted sigmoid!

**Advantages over Sigmoid**:

- **Zero-centered**: Outputs range from $-1$ to $1$, centered at $0$
- **Stronger gradients**: Maximum gradient is 1.0 (vs 0.25 for sigmoid)

**Still has problems**:

- Vanishing gradients for large $|x|$
- Computationally expensive (exponentials)

**When to Use Tanh**:
- RNN/LSTM hidden states (historically)
- When you need bounded, zero-centered outputs
- Rarely in modern feedforward networks

**Example**:
```python
def tanh(x):
    """Hyperbolic tangent activation."""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh."""
    return 1 - np.tanh(x) ** 2

# Compare sigmoid and tanh gradients
x = np.array([0, 1, 2, 3])
print(f"x:              {x}")
print(f"sigmoid grad:   {sigmoid_derivative(x).round(4)}")
print(f"tanh grad:      {tanh_derivative(x).round(4)}")

# Output:
# x:              [0 1 2 3]
# sigmoid grad:   [0.25   0.1966 0.105  0.0452]
# tanh grad:      [1.     0.4200 0.0707 0.0099]
```

At $x = 0$, tanh gradient is 1.0 vs sigmoid's 0.25, this is four times stronger!

## The ReLU Revolution

### Rectified Linear Unit (ReLU)

In 2010, ReLU changed everything. This simple function enabled training of truly deep networks.

**Formula**:
$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**Derivative**:
$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**Range**: $[0, \infty)$

### Why ReLU Changed Everything

1. **No Vanishing Gradient (for positive inputs)**: Gradient is exactly 1 for $x > 0$. Gradients flow unchanged through positive activations.

2. **Computational Efficiency**: It's just a comparison and selection, no exponentials, no divisions. Orders of magnitude faster than sigmoid/tanh.

3. **Sparse Activations**: For typical inputs, about 50% of neurons output zero. This sparsity has regularization benefits and computational savings.

4. **Biological Plausibility**: Neurons in the brain have firing thresholds, and ReLU mimics this behavior.

### The Dying ReLU Problem

Now, ReLU has one critical flaw: **dying neurons** problem.

If a neuron's input becomes negative for all training examples, its gradient is zero, and it stops learning forever. This generally happens when:

- Learning rate is too high (weights become very negative)
- Bad initialization pushes neurons into negative territory
- Unlucky data distribution

**Example of Dying ReLU**:
```python
def relu(x):
    """Rectified Linear Unit."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU."""
    return (x > 0).astype(float)

# Dying ReLU scenario
weights = -5.0  # Very negative weight
inputs = np.array([1, 2, 3, 4, 5])
pre_activation = weights * inputs  # All negative!

print(f"Inputs: {inputs}")
print(f"Pre-activation: {pre_activation}")
print(f"ReLU output: {relu(pre_activation)}")  # All zeros!
print(f"Gradients: {relu_derivative(pre_activation)}")  # All zeros!
# This neuron is dead—it will never learn again.
```

### Let's Look at Some ReLU Best Practices

1. **Use proper initialization**: He initialization (variance $= 2/n$) is designed for ReLU
2. **Moderate learning rates**: Don't let weights become too negative
3. **Batch normalization**: Keeps activations in a good range
4. **Monitor dead neurons**: Track percentage of zeros in activations

## ReLU Variants: Fixing the Dying ReLU Problem

### Leaky ReLU

It came with the simplest fix: allow a small gradient for negative inputs.

**Formula**:
$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

Where $\alpha$ is a small constant, typically 0.01.

**Derivative**:
$$\text{LeakyReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \leq 0 \end{cases}$$

**Advantages**:
- No dead neurons (gradient is always $\alpha$ for negative inputs)
- Still computationally efficient
- Often works as well or better than ReLU

**Disadvantages**:
- Introduces a hyperparameter $\alpha$
- Results can be inconsistent

### Parametric ReLU (PReLU)

Make $\alpha$ learnable instead of fixed.

**Formula**:
$$\text{PReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

Where $\alpha$ is a **learnable parameter**.

**Advantages**:
- Network learns the optimal slope
- Can adapt per-channel or per-neuron

**Disadvantages**:
- Adds parameters to learn
- Risk of overfitting on small datasets

### Exponential Linear Unit (ELU)

ELU provides smooth negative values with saturation.

**Formula**:
$$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

**Derivative**:
$$\text{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha e^x = \text{ELU}(x) + \alpha & \text{if } x \leq 0 \end{cases}$$

**Advantages**:
- Smooth curve (no kink at zero)
- Negative values push mean towards zero
- Saturates for large negative values (noise robustness)

**Disadvantages**:
- Exponential computation for negative inputs
- Slower than ReLU

### Scaled Exponential Linear Unit (SELU)

SELU is designed for self-normalizing neural networks.

**Formula**:
$$\text{SELU}(x) = \lambda \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

Where $\lambda \approx 1.0507$ and $\alpha \approx 1.6733$ are carefully chosen constants.

**The Magic of SELU**:

SELU is designed to automatically normalize the activations. With proper initialization (LeCun Normal) and without batch normalization, activations in a deep SELU network will automatically converge to zero mean and unit variance. This "self-normalization" property makes SELU particularly powerful for deep networks.

**Requirements for SELU to work**:

- LeCun Normal initialization
- No batch normalization
- Fully connected layers (not conv layers)
- Alpha dropout instead of regular dropout

**Advantages**:

- Self-normalizing (no BatchNorm needed)
- Strong performance on certain tasks

**Disadvantages**:

- Strict requirements
- Doesn't work well with CNNs
- Exponential computation

### Comparison of ReLU Variants

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu(x):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

x = np.linspace(-3, 3, 100)

fig, axes = plt.subplots(1, 4, figsize=(16, 3))
functions = [
    ('ReLU', relu(x)),
    ('Leaky ReLU (α=0.1)', leaky_relu(x, 0.1)),
    ('ELU (α=1)', elu(x, 1.0)),
    ('SELU', selu(x))
]

for ax, (name, y) in zip(axes, functions):
    ax.plot(x, y, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_title(name)
    ax.set_xlabel('x')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Modern Activations: GELU, Swish, and Mish

### GELU (Gaussian Error Linear Unit)

GELU is the activation function of choice for Transformers, BERT, GPT, and most modern language models.

**Formula**:
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

Where $\Phi(x)$ is the CDF of the standard normal distribution.

**Approximation** (commonly used):
$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)$$

**Intuition**:

GELU can be thought of as a smooth, probabilistic version of ReLU. Instead of a hard threshold:

- ReLU: "If positive, pass through; otherwise, zero"
- GELU: "Pass through with probability proportional to how positive the input is"

For large positive $x$: $\text{GELU}(x) \approx x$ (like ReLU)
For large negative $x$: $\text{GELU}(x) \approx 0$ (like ReLU)
Around zero: Smooth transition (unlike ReLU's kink)

**Derivative**:
$$\text{GELU}'(x) = \Phi(x) + x \cdot \phi(x)$$

Where $\phi(x)$ is the PDF of the standard normal.

**Advantages**:

- Smooth, non-monotonic (has a small negative region)
- Works exceptionally well in Transformers
- Better gradient flow than ReLU

**Disadvantages**:

- More computationally expensive than ReLU
- Uses approximation in practice

### Swish / SiLU (Sigmoid Linear Unit)

Discovered by Google Brain's neural architecture search in 2017.

**Formula**:
$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**Derivative**:
$$\text{Swish}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \text{Swish}(x) + \sigma(x)(1 - \text{Swish}(x))$$

**Properties**:

- Smooth and non-monotonic
- Unbounded above, bounded below (approaches 0)
- Self-gated: the sigmoid acts as a gate on the input

**Swish-β (Parameterized)**:
$$\text{Swish}_\beta(x) = x \cdot \sigma(\beta x)$$

- When $\beta = 1$: Standard Swish
- When $\beta \to \infty$: Approaches ReLU
- When $\beta \to 0$: Approaches linear function $f(x) = x/2$

**Advantages**:
- Often outperforms ReLU on deep networks
- Smooth gradient flow
- Non-monotonic (allows for negative values)

**Disadvantages**:
- Sigmoid computation is expensive
- Not always better than ReLU

### Mish

Mish was proposed in 2019 and showed improvements in computer vision tasks.

**Formula**:
$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$$

**Properties**:
- Smooth and non-monotonic
- Self-regularizing (bounded below at ≈ -0.31)
- Preserves small negative gradients

**Advantages**:
- Strong performance in CNNs (YOLOv4 uses Mish)
- Smooth gradient landscape
- Better than ReLU and Swish on some vision tasks

**Disadvantages**:
- Expensive to compute (exp, log, tanh)
- Benefits vary by task

### Comparison: GELU vs Swish vs Mish

```python
def gelu(x):
    """GELU activation using tanh approximation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def swish(x):
    """Swish/SiLU activation."""
    return x * sigmoid(x)

def mish(x):
    """Mish activation."""
    return x * np.tanh(np.log(1 + np.exp(x)))

x = np.linspace(-4, 4, 100)

# All three look similar but have subtle differences
print("At x = -1:")
print(f"  GELU:  {gelu(-1):.4f}")
print(f"  Swish: {swish(-1):.4f}")
print(f"  Mish:  {mish(-1):.4f}")

print("\nAt x = -0.5:")
print(f"  GELU:  {gelu(-0.5):.4f}")
print(f"  Swish: {swish(-0.5):.4f}")
print(f"  Mish:  {mish(-0.5):.4f}")
```

**When to Use Each**:
- **GELU**: Transformers, NLP, default for modern architectures
- **Swish**: Deep CNNs, when ReLU underperforms
- **Mish**: Computer vision, object detection (YOLOv4)

## Output Activations: Softmax and It's Friends

### Softmax

So, softmax converts a vector of raw scores (logits) into a probability distribution.

**Formula**:
$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Properties**:

- Input logits (unbounded real numbers)
- Outputs sum to 1
- All outputs are positive
- Preserves relative ordering (largest input → largest probability)
- Numerically unstable without trick

**Numerical Stability**:

Computing $e^{z_i}$ can overflow for large $z_i$. So, the trick is to subtract the maximum:

$$\text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{K} e^{z_j - \max(z)}}$$

This is mathematically equivalent but numerically stable.

**Derivative (Jacobian)**:

The softmax Jacobian:
$$\frac{\partial \text{Softmax}_i}{\partial z_j} = \text{Softmax}_i(\delta_{ij} - \text{Softmax}_j)$$

When combined with cross-entropy loss, this simplifies beautifully to:
$$\frac{\partial L}{\partial z_i} = \text{Softmax}_i - y_i$$

**Implementation**:
```python
def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Example
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Logits: {logits}")
print(f"Probabilities: {probs.round(4)}")
print(f"Sum: {probs.sum():.4f}")

# Output:
# Logits: [2.  1.  0.1]
# Probabilities: [0.6590 0.2424 0.0986]
# Sum: 1.0000
```

### Temperature in Softmax

Temperature $T$ controls the "sharpness" of the distribution. "Sharpness" means how concentrated the probability mass is around the maximum value.:

$$\text{Softmax}_T(z_i) = \frac{e^{z_i/T}}{\sum_{j=1}^{K} e^{z_j/T}}$$

- **$T = 1$**: Standard softmax
- **$T \to 0$**: Approaches one-hot (argmax)
- **$T \to \infty$**: Approaches uniform distribution

**Applications**:
- Knowledge distillation (soft targets)
- Controlling randomness in text generation
- Label smoothing

### Sigmoid for Multi-Label Classification

For multi-label problems (each sample can have multiple labels), we use sigmoid on each output independently:

$$P(\text{label}_i = 1) = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}$$

Unlike softmax, sigmoid outputs don't need to sum to 1.

### Linear Output (No Activation)

For regression tasks, the output layer typically has no activation:

$$\hat{y} = \mathbf{w}^T \mathbf{h} + b$$

This allows the network to output any real value.

## Specialized Activations

### Hardswish and Hardsigmoid

It's a smooth approximation of ReLU, but with hardware-optimized versions that avoid expensive exponentials.

**Hardsigmoid**:
$$\text{Hardsigmoid}(x) = \max(0, \min(1, \frac{x + 3}{6}))$$

**Hardswish**:
$$\text{Hardswish}(x) = x \cdot \text{Hardsigmoid}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}$$

Used in MobileNetV3 and EfficientNet for mobile deployment.

### Maxout

Maxout takes the maximum over multiple linear transformations:

$$\text{Maxout}(x) = \max(w_1^T x + b_1, w_2^T x + b_2, ..., w_k^T x + b_k)$$

**Properties**:
- Can approximate any convex function
- No saturation (gradients flow through the max)
- Doubles parameters (or more)

### Softplus

It's a smooth approximation of ReLU:

$$\text{Softplus}(x) = \ln(1 + e^x)$$

**Derivative**:
$$\text{Softplus}'(x) = \sigma(x) = \frac{1}{1 + e^{-x}}$$

The derivative of softplus is sigmoid! This connects these functions beautifully.

### GLU (Gated Linear Unit)

GLU is a gating mechanism that combines linear and sigmoid transformations, allowing the network to learn what information to let through:

$$\text{GLU}(x, W, V, b, c) = (xW + b) \otimes \sigma(xV + c)$$

The sigmoid acts as a gate, controlling information flow.

## Choosing the Right Activation Function

### Decision Guide by Architecture

```
Architecture Type?
│
├── Feedforward Networks (MLP)
│   ├── Standard tasks → ReLU or Leaky ReLU
│   ├── Deep networks → GELU or Swish
│   └── Self-normalizing → SELU (with proper setup)
│
├── Convolutional Networks (CNN)
│   ├── Standard → ReLU
│   ├── Better performance → Mish or Swish
│   ├── Mobile/efficient → Hardswish
│   └── Deep ResNets → ReLU with proper init
│
├── Transformers / Attention
│   ├── Encoder → GELU (default)
│   ├── Decoder → GELU (default)
│   └── FFN layers → GELU or Swish
│
├── Recurrent Networks (RNN/LSTM)
│   ├── Gates → Sigmoid
│   ├── State → Tanh
│   └── Modern RNNs → Consider GELU
│
└── Output Layer
    ├── Binary classification → Sigmoid
    ├── Multi-class → Softmax
    ├── Multi-label → Sigmoid (per label)
    └── Regression → Linear (no activation)
```

### Comparison Table

| Activation | Range | Gradient | Speed | Use Case |
|-----------|-------|----------|-------|----------|
| **Sigmoid** | (0, 1) | Vanishing | Slow | Output (binary), gates |
| **Tanh** | (-1, 1) | Vanishing | Slow | RNN states, bounded output |
| **ReLU** | [0, ∞) | Dead neurons | Fast | CNNs, default hidden |
| **Leaky ReLU** | (-∞, ∞) | Good | Fast | When ReLU has dead neurons |
| **ELU** | (-α, ∞) | Good | Medium | Better than ReLU sometimes |
| **SELU** | Self-norm | Good | Medium | Self-normalizing networks |
| **GELU** | ≈(-0.17, ∞) | Excellent | Medium | Transformers, NLP |
| **Swish** | ≈(-0.28, ∞) | Excellent | Medium | Deep CNNs |
| **Mish** | ≈(-0.31, ∞) | Excellent | Slow | Vision, YOLO |
| **Softmax** | (0, 1), sum=1 | Special | Medium | Multi-class output |

### Rules of Thumb

1. **Start with ReLU**: It's fast and works for most cases
2. **Transformers**: Use GELU (it's what BERT/GPT use)
3. **Deep CNNs**: Try Mish or Swish if ReLU underperforms
4. **Mobile deployment**: Use Hardswish
5. **Dead neuron issues**: Switch to Leaky ReLU or ELU
6. **Output layers**: Match to your task (softmax, sigmoid, linear)

## Implementation Guide

### PyTorch Activation Functions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Built-in activations
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
elu = nn.ELU(alpha=1.0)
selu = nn.SELU()
gelu = nn.GELU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=-1)

# Functional API (often preferred in forward())
x = torch.randn(10, 5)
y1 = F.relu(x)
y2 = F.leaky_relu(x, negative_slope=0.01)
y3 = F.gelu(x)
y4 = F.silu(x)  # Swish = SiLU
y5 = F.mish(x)

# In a model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Choose activation
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'swish':
            self.act = nn.SiLU()
        elif activation == 'mish':
            self.act = nn.Mish()
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)  # No activation on output
        return x
```

### Custom Activation Functions

```python
class CustomActivation(nn.Module):
    """Template for custom activation functions."""

    def __init__(self, param=1.0):
        super().__init__()
        self.param = param

    def forward(self, x):
        # Your custom activation here
        return x * torch.sigmoid(self.param * x)

# Using autograd function for efficiency
class CustomActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return torch.where(x > 0, x, alpha * x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = torch.where(x > 0,
                            torch.ones_like(x),
                            torch.full_like(x, ctx.alpha))
        return grad_output * grad_x, None

# Learnable activation (like PReLU)
class LearnableActivation(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_features) * 0.25)

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * x)
```

### NumPy Implementations (From Scratch)

```python
import numpy as np

class Activations:
    """Collection of activation functions and their derivatives."""

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_derivative(x):
        s = Activations.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def elu_derivative(x, alpha=1.0):
        return np.where(x > 0, 1, Activations.elu(x, alpha) + alpha)

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def swish(x):
        return x * Activations.sigmoid(x)

    @staticmethod
    def softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

## Conclusion

Activation functions are still an active area of research, as architectures evolve, so will activation functions. This trends toward more sophisticated activation functions that offer:

- Smoother functions (better gradients)
- Non-monotonic functions (more expressivity)
- Hardware-optimized variants (faster computation)

---

### Further Reading and Resources

**Papers**:

1. ["Understanding difficulty of training deep feedforward networks"](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
2. ["Rectified Linear Units Improve Restricted Boltzmann Machines"](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)
3. ["Fast and Accurate Deep Networks with ELUs"](https://arxiv.org/pdf/1511.07289)
4. ["Searching for Activation Functions"](https://arxiv.org/pdf/1710.05941)
5. ["Mish: A Self Regularized Non-Monotonic Activation Function"](https://arxiv.org/pdf/1908.08681)