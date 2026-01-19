---
layout: post
title: "Understanding Optimizers: How Neural Networks Actually Learn"
date: 2026-01-19
series: "Deep Learning Series"
series_author: "Mayank Sharma"
series_image: "/assets/images/2026-01-19-understanding-optimizers/optimizer-landscape.png"
excerpt: "Master optimization algorithms from SGD to Adam and beyond. Understand how neural networks navigate the loss landscape to find optimal solutions."
---

Continuing in our Deep Learning Series, we now turn our attention to the most critical component of neural network training: optimizers. Imagine you're blindfolded on a mountain range, trying to find the lowest valley. You can't see the landscape, but you can feel the slope beneath your feet. Each step you take is guided by the gradient, the direction of steepest descent. This is exactly how neural networks learn: optimizers are the algorithms that decide how to take each step based on the gradient information available.

## Table of Contents

1. [Introduction: The Art of Finding Minima](#introduction-the-art-of-finding-minima)
2. [Gradient Descent: The Foundation](#gradient-descent-the-foundation)
3. [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
4. [Momentum: Learning from the Past](#momentum-learning-from-the-past)
5. [Nesterov Accelerated Gradient](#nesterov-accelerated-gradient)
6. [AdaGrad: Adaptive Learning Rates](#adagrad-adaptive-learning-rates)
7. [RMSprop: Solving AdaGrad's Problems](#rmsprop-solving-adagrads-problems)
8. [Adam: The Best of Both Worlds](#adam-the-best-of-both-worlds)
9. [AdamW: Decoupled Weight Decay](#adamw-decoupled-weight-decay)
10. [Advanced Optimizers: LAMB, LARS, and Beyond](#advanced-optimizers-lamb-lars-and-beyond)
11. [Learning Rate Scheduling](#learning-rate-scheduling)
12. [Choosing the Right Optimizer](#choosing-the-right-optimizer)
13. [Practical Implementation Guide](#practical-implementation-guide)
14. [Conclusion: Mastering Optimization](#conclusion-mastering-optimization)

## Introduction: The Art of Finding Minima

### Why Optimizers Matter

Training a neural network is fundamentally an optimization problem. We have a loss function $L(\theta)$ that measures how wrong our model is, and we want to find parameters $\theta^*$ that minimize this loss:

$$\theta^* = \arg\min_{\theta} L(\theta)$$

The optimizer is the algorithm that finds this minimum. It's like a navigation system for the loss landscape, basically a high-dimensional surface where every point represents a different set of model parameters.

### The Loss Landscape

Consider the loss landscape of a neural network. For a simple network with just two parameters, we could visualize this as a 3D surface, you know like a mountain range where elevation represents loss. Real neural networks have millions of parameters, creating a surface in millions of dimensions that we cannot visualize but must navigate.

This landscape has several challenging features:

1. **Local Minima**: Valleys that aren't the lowest point
2. **Saddle Points**: Flat regions that are neither maxima nor minima
3. **Plateaus**: Extended flat regions where gradients vanish
4. **Ravines**: Narrow valleys with steep walls

Good optimizers must handle all these challenges efficiently. Now, let's explore each major optimizer and understand how it works.

## Gradient Descent: The Foundation

### The Core Idea

Gradient descent is the simplest optimization algorithm. The key insight is that the gradient $\nabla L(\theta)$ points in the direction of steepest ascent. To minimize the loss, we move in the opposite direction:

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

Where:
- $\theta_t$ are the parameters at step $t$
- $\eta$ is the learning rate (step size)
- $\nabla L(\theta_t)$ is the gradient of the loss

### A Concrete Example With Vanilla Gradient Descent

Let's say we're training a simple linear regression $y = wx + b$ with MSE loss:

$$L(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (wx_i + b))^2$$

The gradients are:

$$\frac{\partial L}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} x_i(y_i - (wx_i + b))$$

$$\frac{\partial L}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - (wx_i + b))$$

Starting from random values, we update:

$$w_{t+1} = w_t - \eta \frac{\partial L}{\partial w}$$
$$b_{t+1} = b_t - \eta \frac{\partial L}{\partial b}$$

### Now Comes The Learning Rate Dilemma

The learning rate $\eta$ is critical: if it's **Too large**, the algorithm overshoots, oscillates, or diverges and if it's **Too small**, training is extremely slow. So, what we need is **Just right**, somethign that ensures smooth convergence (but what's "just right"?)

This is called the `"learning rate scheduling problem"`, and it's one of the most challenging aspects of deep learning. Modern deep learning and modern optimizers address it in sophisticated ways.

### Limitations of Vanilla Gradient Descent

1. **Requires full dataset**: Must compute gradients over all training examples
2. **Same learning rate for all parameters**: Some parameters may need faster or slower updates
3. **Gets stuck in local minima**: No mechanism to escape poor solutions
4. **Oscillates in ravines**: Zig-zags instead of going directly to the minimum

## Stochastic Gradient Descent (SGD)

Unlike Vanilla Gradient Descent, which uses the entire dataset to compute gradients, SGD uses a single randomly chosen sample (or mini-batch):

$$\theta_{t+1} = \theta_t - \eta \nabla L_i(\theta_t)$$

Where $L_i$ is the loss for sample (or mini-batch) $i$.

Important thing here is, to note that the expected value of the stochastic gradient equals the true gradient:

$$\mathbb{E}[\nabla L_i(\theta)] = \nabla L(\theta)$$

This means SGD will converge to the same solution (on average), but each step is much faster to compute.

### The Benefits of Noise

The stochastic nature of SGD has unexpected benefits:

1. **Escapes local minima**: Random fluctuations can push the optimizer out of poor solutions
2. **Better generalization**: The noise acts as implicit regularization
3. **Faster iterations**: Each step is cheap to compute

### Mini-Batch SGD

In practice, we use mini-batches (typically 32-256 samples):

$$\theta_{t+1} = \theta_t - \eta \frac{1}{|B|} \sum_{i \in B} \nabla L_i(\theta_t)$$

This balances three important factors:

- **Gradient variance**: Larger batches → more stable gradients
- **Computation efficiency**: Modern hardware is optimized for batch operations
- **Generalization**: Some noise is beneficial

### Implementation

```python
import numpy as np

class SGD:
    """Vanilla Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        """
        Update parameters using gradients.

        Args:
            params: Dictionary of parameters {name: value}
            grads: Dictionary of gradients {name: gradient}

        Returns:
            Updated parameters
        """
        for name in params:
            params[name] -= self.learning_rate * grads[name]
        return params

# Example usage
params = {'W': np.random.randn(10, 5), 'b': np.zeros(5)}
grads = {'W': np.random.randn(10, 5) * 0.1, 'b': np.random.randn(5) * 0.1}

optimizer = SGD(learning_rate=0.01)
params = optimizer.update(params, grads)
```

## Momentum: Learning from the Past

### The Problem with SGD

SGD often tend to oscillate wildly in ravines, narrow valleys where the gradient is steep in one direction but shallow in another. The algorithm zig-zags, making slow progress toward the minimum. Imagine a ball rolling down a hill. It doesn't just follow the gradient, it builds up momentum. If it's been moving in one direction consistently, it should continue in that direction even if the current gradient is small.

### Mathematical Formulation

Momentum introduces a "velocity" term that accumulates past gradients:

$$v_t = \beta v_{t-1} + \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

Where:

- $v_t$ is the velocity at step $t$
- $\beta$ is the momentum coefficient (typically 0.9)

### Exponential Moving Average

The velocity is an exponentially weighted moving average of gradients. Expanding:

$$v_t = \nabla L(\theta_t) + \beta \nabla L(\theta_{t-1}) + \beta^2 \nabla L(\theta_{t-2}) + ...$$

Recent gradients have more influence, but past gradients still contribute. This smooths out oscillations.

### Why It Works

The momentum term has three key effects:

- **Consistent direction**: If gradients point the same way, velocity builds up
- **Oscillating direction**: Opposing gradients cancel out in the velocity
- **Faster convergence**: Especially in ravines and on plateaus

### Implementation

```python
class MomentumSGD:
    """SGD with Momentum."""

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        """Update parameters using momentum."""
        for name in params:
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(params[name])

            # Update velocity
            self.velocity[name] = (self.momentum * self.velocity[name] +
                                   grads[name])

            # Update parameters
            params[name] -= self.learning_rate * self.velocity[name]

        return params
```

### Choosing the Momentum Coefficient

- **$\beta = 0$**: No momentum (vanilla SGD)
- **$\beta = 0.9$**: Standard choice, good balance
- **$\beta = 0.99$**: Stronger momentum, useful for noisy gradients
- **$\beta \to 1$**: Very strong momentum, risk of overshooting

## Nesterov Accelerated Gradient

### The Problem with Standard Momentum

So, the standard momentum can overshoot the minimum because it keeps moving even when it should slow down. It's like a ball that builds up too much speed and rolls past the valley bottom.

### Nesterov's Insight

Nesterov realized we could do better. Instead of computing the gradient at the current position, compute it at the "lookahead" position—where momentum would take you:

$$v_t = \beta v_{t-1} + \nabla L(\theta_t - \eta \beta v_{t-1})$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

Nesterov momentum says that "Before computing the gradient, peek ahead to see where momentum is taking me. If I'm about to overshoot, I'll know and can correct."

This provides a form of "anticipatory" update that leads to better convergence.

### Comparison

| Aspect | Standard Momentum | Nesterov Momentum |
|--------|------------------|-------------------|
| Gradient computed at | Current position | Lookahead position |
| Correction | After overshooting | Before overshooting |
| Convergence | Good | Better (theoretically optimal for convex) |

### Implementation

```python
class NesterovMomentum:
    """Nesterov Accelerated Gradient optimizer."""

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        """
        Update using Nesterov momentum.

        Note: In practice, we use a reformulation that doesn't
        require computing gradients at the lookahead position.
        """
        for name in params:
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(params[name])

            v_prev = self.velocity[name].copy()

            # Update velocity
            self.velocity[name] = (self.momentum * self.velocity[name] -
                                   self.learning_rate * grads[name])

            # Nesterov update: use velocity correction
            params[name] += (-self.momentum * v_prev +
                            (1 + self.momentum) * self.velocity[name])

        return params
```

## AdaGrad: Adaptive Learning Rates

AdaGrad adapts the learning rate for each parameter based on the historical gradient information. The key idea is that different parameters may need different learning rates:

- **Frequent features**: Need smaller learning rates (already well-tuned)
- **Rare features**: Need larger learning rates (haven't been updated much)

AdaGrad adapts the learning rate for each parameter based on its history.

### Mathematical Formulation

AdaGrad accumulates squared gradients:

$$G_t = G_{t-1} + (\nabla L(\theta_t))^2$$

And divides the learning rate by the square root:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla L(\theta_t)$$

Where:
- $G_t$ is the accumulated squared gradient
- $\epsilon$ is a small constant for numerical stability (typically $10^{-8}$)

This means:

- **Large accumulated gradient**: Learning rate decreases (frequent updates)
- **Small accumulated gradient**: Learning rate stays larger (rare updates)

This is especially useful for sparse features (e.g., NLP with rare words).

### The Problem

The accumulated gradient $G_t$ only grows, never shrinks. Eventually, the effective learning rate becomes so small that learning stops entirely. This is called "premature learning rate decay."

### Implementation

```python
class AdaGrad:
    """AdaGrad optimizer with per-parameter learning rates."""

    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.accumulated_grads = {}

    def update(self, params, grads):
        """Update parameters with adaptive learning rates."""
        for name in params:
            if name not in self.accumulated_grads:
                self.accumulated_grads[name] = np.zeros_like(params[name])

            # Accumulate squared gradients
            self.accumulated_grads[name] += grads[name] ** 2

            # Adaptive update
            params[name] -= (self.learning_rate * grads[name] /
                            (np.sqrt(self.accumulated_grads[name]) +
                             self.epsilon))

        return params
```

## RMSprop: Solving AdaGrad's Problems

RMSprop (proposed by Geoff Hinton in a Coursera lecture) uses an exponentially decaying average instead of accumulating all squared gradients:

$$E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) (\nabla L(\theta_t))^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \nabla L(\theta_t)$$

Where $\gamma$ is the decay rate (typically 0.9).

### Why It Works

- **Recent gradients matter more**: The exponential decay forgets old gradients
- **Learning rate adapts dynamically**: Can increase or decrease based on recent history
- **No premature decay**: The denominator doesn't grow unboundedly

### Implementation

```python
class RMSprop:
    """RMSprop optimizer with exponentially decaying gradient average."""

    def __init__(self, learning_rate=0.001, decay=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.moving_avg_sq = {}

    def update(self, params, grads):
        """Update parameters using RMSprop."""
        for name in params:
            if name not in self.moving_avg_sq:
                self.moving_avg_sq[name] = np.zeros_like(params[name])

            # Update moving average of squared gradients
            self.moving_avg_sq[name] = (self.decay * self.moving_avg_sq[name] +
                                        (1 - self.decay) * grads[name] ** 2)

            # Update parameters
            params[name] -= (self.learning_rate * grads[name] /
                            (np.sqrt(self.moving_avg_sq[name]) + self.epsilon))

        return params
```

## Adam: The Best of Both Worlds

Adam (Adaptive Moment Estimation) was introduced by Diederik Kingma and Jimmy Ba in 2014 and combines the advantages of momentum and RMSprop. It maintains two moving averages:
1. **Momentum**: Exponential moving average of gradients (first moment)
2. **RMSprop**: Exponential moving average of squared gradients (second moment)

Plus bias correction to account for initialization at zero.

### Mathematical Formulation

**First moment (momentum-like)**:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t)$$

**Second moment (RMSprop-like)**:
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(\theta_t))^2$$

**Bias correction**:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Update**:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### Understanding Bias Correction

At $t=0$, $m_0 = v_0 = 0$. In early steps, the moving averages are biased toward zero. Bias correction compensates:

- At $t=1$: $\hat{m}_1 = \frac{m_1}{1 - 0.9} = 10 \times m_1$ (corrects for initialization)
- As $t \to \infty$: $\hat{m}_t \to m_t$ (correction vanishes)

### Default Hyperparameters

The original Adam paper recommends:
- $\beta_1 = 0.9$: First moment decay
- $\beta_2 = 0.999$: Second moment decay
- $\epsilon = 10^{-8}$: Numerical stability
- $\eta = 0.001$: Learning rate

These work well for most problems without tuning.

### Implementation

```python
class Adam:
    """Adam optimizer combining momentum and adaptive learning rates."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step

    def update(self, params, grads):
        """Update parameters using Adam."""
        self.t += 1

        for name in params:
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])

            # Update first moment (momentum)
            self.m[name] = (self.beta1 * self.m[name] +
                           (1 - self.beta1) * grads[name])

            # Update second moment (RMSprop-like)
            self.v[name] = (self.beta2 * self.v[name] +
                           (1 - self.beta2) * grads[name] ** 2)

            # Bias correction
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Update parameters
            params[name] -= (self.learning_rate * m_hat /
                            (np.sqrt(v_hat) + self.epsilon))

        return params
```

### When Adam Struggles

Despite it's popularity, Adam has known issues:

1. **Poor generalization**: Sometimes converges to sharper minima than SGD with momentum
2. **Non-convergence**: In some pathological cases, doesn't converge (though rare in practice)
3. **Weight decay interaction**: L2 regularization doesn't work as expected

## AdamW: Decoupled Weight Decay

AdamW is a variant of Adam that fixes the weight decay issue. The key insight is that Adam's weight decay implementation is mathematically incorrect.

### The Problem with L2 Regularization in Adam

In standard SGD, L2 regularization and weight decay are equivalent:

**L2 Regularization**: Add $\frac{\lambda}{2}||\theta||^2$ to loss
$$\nabla L_{reg} = \nabla L + \lambda \theta$$

**Weight Decay**: Shrink weights after each update
$$\theta_{t+1} = \theta_t - \eta \nabla L - \eta \lambda \theta_t$$

In Adam, these are NOT equivalent because the gradient is divided by $\sqrt{v_t}$:

**L2 in Adam**: The regularization gradient is also scaled by $\frac{1}{\sqrt{v_t}}$, which is wrong. We want consistent regularization regardless of the gradient history.

### The Fix: Decoupled Weight Decay

AdamW applies weight decay directly to the weights, not through the gradient:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t - \eta \lambda \theta_t$$

### Implementation

```python
class AdamW:
    """AdamW optimizer with decoupled weight decay."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.01):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        """Update parameters using AdamW."""
        self.t += 1

        for name in params:
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])

            # Update moments
            self.m[name] = (self.beta1 * self.m[name] +
                           (1 - self.beta1) * grads[name])
            self.v[name] = (self.beta2 * self.v[name] +
                           (1 - self.beta2) * grads[name] ** 2)

            # Bias correction
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Adam update
            params[name] -= (self.learning_rate * m_hat /
                            (np.sqrt(v_hat) + self.epsilon))

            # Decoupled weight decay
            params[name] -= self.learning_rate * self.weight_decay * params[name]

        return params
```

### Why AdamW Matters

AdamW is now the default optimizer for many state-of-the-art models:
- **BERT**: Uses AdamW with specific hyperparameters
- **GPT**: Uses AdamW
- **Vision Transformers**: Uses AdamW

The "W" in AdamW stands for "weight decay", a small change with big implications.

## Advanced Optimizers: LAMB, LARS, and Beyond

### The Large Batch Challenge

As we train larger models, we want to use larger batches to speed up training (more parallelism). But large batches often hurt generalization, the model converges to sharper, less generalizable minima.

### LARS (Layer-wise Adaptive Rate Scaling)

LARS scales the learning rate differently for each layer based on the ratio of weight norm to gradient norm:

$$\lambda_l = \frac{||\theta_l||}{||\nabla L(\theta_l)||}$$

$$\theta_l^{t+1} = \theta_l^t - \eta \lambda_l \frac{\nabla L(\theta_l)}{||\nabla L(\theta_l)||}$$

This prevents layers with small gradients from getting left behind.

### LAMB (Layer-wise Adaptive Moments for Batch training)

LAMB combines LARS with Adam, enabling large batch training with adaptive learning rates:

$$r_t = \frac{m_t}{\sqrt{v_t} + \epsilon}$$

$$\theta_{t+1} = \theta_t - \eta \frac{||\theta_t||}{||r_t + \lambda \theta_t||} (r_t + \lambda \theta_t)$$

LAMB was used to train BERT in 76 minutes (vs 3+ days with standard Adam).

### Other Modern Optimizers

These are some of the most popular optimizers used in modern deep learning, out of scope for this series. However, they are worth mentioning and if you are curious then you can explore further on your own:

- **RAdam (Rectified Adam)**: Addresses variance issues in early training
- **Lookahead**: Wraps any optimizer with a "slow weights" mechanism
- **Ranger**: Combines RAdam with Lookahead
- **Lion**: New optimizer from Google (2023) using sign of momentum

## Learning Rate Scheduling

### Why Schedule?

A fixed learning rate has limitations:
- **Too high initially**: Training is unstable
- **Too high later**: Overshoots the minimum
- **Too low always**: Training is too slow

Learning rate scheduling adjusts $\eta$ during training.

### Common Schedules

**Step Decay**:
$$\eta_t = \eta_0 \times \gamma^{\lfloor t / s \rfloor}$$

Drop the learning rate by $\gamma$ every $s$ epochs.

**Exponential Decay**:
$$\eta_t = \eta_0 \times e^{-kt}$$

Continuous exponential decrease.

**Cosine Annealing**:
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t\pi}{T}))$$

Smooth cosine decay from $\eta_{max}$ to $\eta_{min}$.

**Warmup + Decay**:
Start with a low learning rate, increase linearly for warmup steps, then decay:

$$\eta_t = \begin{cases}
\eta_{max} \times \frac{t}{T_{warmup}} & t < T_{warmup} \\
\text{decay schedule} & t \geq T_{warmup}
\end{cases}$$

### Implementation

```python
class LearningRateScheduler:
    """Learning rate schedulers for training."""

    @staticmethod
    def step_decay(initial_lr, epoch, drop_rate=0.5, epochs_drop=10):
        """Step decay: drop lr by drop_rate every epochs_drop epochs."""
        return initial_lr * (drop_rate ** (epoch // epochs_drop))

    @staticmethod
    def exponential_decay(initial_lr, epoch, decay_rate=0.95):
        """Exponential decay."""
        return initial_lr * (decay_rate ** epoch)

    @staticmethod
    def cosine_annealing(initial_lr, epoch, total_epochs, min_lr=0):
        """Cosine annealing from initial_lr to min_lr."""
        import math
        return min_lr + 0.5 * (initial_lr - min_lr) * (
            1 + math.cos(math.pi * epoch / total_epochs)
        )

    @staticmethod
    def warmup_cosine(initial_lr, epoch, total_epochs,
                      warmup_epochs=5, min_lr=0):
        """Linear warmup followed by cosine decay."""
        import math
        if epoch < warmup_epochs:
            return initial_lr * epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr + 0.5 * (initial_lr - min_lr) * (
                1 + math.cos(math.pi * progress)
            )
```

### One Cycle Policy

The one cycle policy (by Leslie Smith) uses:
1. Warmup from low LR to high LR
2. Annealing from high LR to very low LR
3. Often combined with momentum cycling (high momentum when LR is low)

This often achieves faster convergence and better results.

## Choosing the Right Optimizer

### Decision Framework

```
What type of problem?
├── Computer Vision (CNNs)
│   ├── Training from scratch → SGD + Momentum + LR Schedule
│   ├── Fine-tuning → AdamW with low LR
│   └── Large batch → LARS or LAMB
│
├── NLP / Transformers
│   ├── Training from scratch → AdamW + Warmup + Cosine Decay
│   ├── Fine-tuning → AdamW, LR = 1e-5 to 5e-5
│   └── Large models (GPT-3 scale) → AdamW with careful tuning
│
├── Reinforcement Learning
│   ├── Policy gradients → Adam
│   └── Value functions → RMSprop or Adam
│
└── General / Unsure
    ├── Start with AdamW (safest default)
    ├── If poor generalization → Try SGD + Momentum
    └── If sparse features → AdaGrad or Adam
```

### Optimizer Comparison Table

| Optimizer | Learning Rate | Momentum | Adaptive LR | Weight Decay | Best For |
|-----------|--------------|----------|-------------|--------------|----------|
| SGD | 0.01-0.1 | No | No | L2 | Simple problems |
| SGD+Momentum | 0.01-0.1 | Yes | No | L2 | CNNs, CV |
| Nesterov | 0.01-0.1 | Yes (better) | No | L2 | Convex problems |
| AdaGrad | 0.01 | No | Yes | L2 | Sparse features |
| RMSprop | 0.001 | No | Yes | L2 | RNNs, RL |
| Adam | 0.001 | Yes | Yes | L2 (issues) | General default |
| AdamW | 0.001 | Yes | Yes | Decoupled | Transformers, BERT |
| LAMB | 0.001 | Yes | Yes | Decoupled | Large batch training |

### Hyperparameter Tuning Guidelines

**SGD with Momentum**:
- Learning rate: 0.01-0.1 (higher than Adam)
- Momentum: 0.9
- Weight decay: 1e-4 to 1e-5

**Adam/AdamW**:
- Learning rate: 0.001 (or 3e-4)
- Beta1: 0.9
- Beta2: 0.999
- Epsilon: 1e-8
- Weight decay (AdamW): 0.01

**Fine-tuning**:
- Use 10-100x smaller learning rate than training from scratch
- Warmup is often important

## Practical Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Create model
model = SimpleNet(input_dim=784, hidden_dim=256, output_dim=10)

# Option 1: SGD with Momentum
optimizer_sgd = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

# Option 2: Adam
optimizer_adam = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)

# Option 3: AdamW (recommended for most cases)
optimizer_adamw = optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Learning rate scheduling
scheduler_cosine = CosineAnnealingLR(optimizer_adamw, T_max=100)
scheduler_onecycle = OneCycleLR(
    optimizer_adamw,
    max_lr=0.01,
    epochs=100,
    steps_per_epoch=1000
)

# Training loop
def train_epoch(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in dataloader:
        # Forward pass
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optional: Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update parameters
        optimizer.step()

        # Update learning rate (for OneCycleLR, update per step)
        # scheduler.step()  # Uncomment for per-step schedulers

        total_loss += loss.item()

    # Update learning rate (for epoch-based schedulers)
    scheduler.step()

    return total_loss / len(dataloader)

# Print current learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

print(f"Current LR: {get_lr(optimizer_adamw)}")
```

### Gradient Clipping

Gradient clipping prevents exploding gradients, especially important for RNNs:

```python
# Clip by norm (most common)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### Common Training Recipes

**Vision Transformer (ViT)**:
```python
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=300)
# Warmup for first 10 epochs
```

**BERT Fine-tuning**:
```python
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=1000, num_training_steps=10000
)
```

**ResNet Training**:
```python
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])
```

## Conclusion:

We've traveled from simple gradient descent to sophisticated adaptive methods. Each optimizer represents a different strategy for navigating the loss landscape:

- **SGD**: Follow the gradient, step by step
- **Momentum**: Build up speed in consistent directions
- **Adam**: Adapt to each parameter's needs
- **AdamW**: Regularize properly in the adaptive setting

The best optimizer is the one that works for your specific problem. Experiment, monitor, and adapt.

### Further Reading

**Classic Papers**:
- [Robbins & Monro (1951): "A Stochastic Approximation Method"](https://projecteuclid.org/journalArticle/Download?urlId=10.1214%2Faoms%2F1177729586)
- [Polyak (1964): "Some methods of speeding up the convergence of iteration methods"](https://www.mathnet.ru/links/7c4c06162008a5226a4fae13e0ac554f/zvmmf7713.pdf)
- [Kingma & Ba (2014): "Adam: A Method for Stochastic Optimization"](https://arxiv.org/pdf/1412.6980)
- [Loshchilov & Hutter (2017): "Decoupled Weight Decay Regularization"](https://arxiv.org/pdf/1711.05101)

**Online Resources**:
- [PyTorch Optimizers Documentation](https://pytorch.org/docs/stable/optim.html)
- [Sebastian Ruder's "An overview of gradient descent optimization algorithms"](https://ruder.io/optimizing-gradient-descent/)
- [Distill.pub: "Why Momentum Really Works"](https://distill.pub/2017/momentum/)