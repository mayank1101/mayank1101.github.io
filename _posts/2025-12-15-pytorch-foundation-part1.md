---
layout: post
title: "Part 1: The PyTorch Foundation"
date: 2025-12-15
series: "NLP Mastery Series"
series_author: "Mayank Sharma"
series_image: "/assets/images/2025-12-15-pytorch-foundation-part1/pytorch-foundation-part1.png"
excerpt: "Master PyTorch fundamentals—tensors, autograd, and gradient descent. Learn how dynamic computation graphs work, how GPUs accelerate training, and how to build learning systems from scratch."
---

## Introduction: Why PyTorch?

If you work with modern deep learning, especially NLP, vision, or generative AI, you will almost certainly encounter **PyTorch**.

Over the past few years, PyTorch has become the *default* framework for AI research and is rapidly becoming just as important in production systems. But this dominance didn’t happen by accident. PyTorch succeeds because it feels natural to humans.

You write normal Python code.  
You run it line by line.  
And what you write is *exactly* what the model executes.

That design choice shapes everything we’ll learn in this article.

---

## The Dynamic Graph Advantage

PyTorch uses **dynamic computation graphs**, often described as *define-by-run*. This means the computation graph is built **as your code runs**, not ahead of time.

Why does this matter?

- **Intuitive** – You write standard Python. No sessions. No placeholders.
- **Debuggable** – `print()`, `pdb`, stack traces all work normally.
- **Flexible** – You can change control flow, loops, and model structure on the fly.

This is why PyTorch feels more like programming and less like configuration.

### Why researchers prefer PyTorch

- The majority of papers at NeurIPS, ICML, and ICLR use PyTorch.
- It integrates seamlessly with NumPy and the Python ecosystem.
- Hugging Face Transformers are built on PyTorch.
- Research ideas move quickly from concept to code.

### PyTorch in production

PyTorch is no longer “just for research”:

- **PyTorch 2.x** introduces `torch.compile` for major performance gains.
- **TorchServe** enables scalable model serving.
- **ONNX export** allows deployment across platforms.
- Companies like Meta, Microsoft, and Tesla use PyTorch at scale.

With that context, let’s start from the absolute foundation.

---

## 1. Tensors: The Core Data Structure

### What is a Tensor?

At its core, a **tensor** is a multi-dimensional array. If you’ve used NumPy arrays, you already understand the idea. Think of tensors as containers for numbers, arranged in different dimensions:

- **Scalar (Rank 0)** – a single number  
- **Vector (Rank 1)** – a list of numbers  
- **Matrix (Rank 2)** – a table of numbers  
- **Higher-rank tensors** – stacks of matrices and beyond  

In deep learning, tensors represent *everything*:

- Images → `(batch, channels, height, width)`
- Text → `(batch, sequence_length, embedding_dim)`
- Model parameters → weights and biases

If you understand tensors, you understand PyTorch.

---

### Creating Tensors

```python
import torch
import numpy as np

# From Python lists
t1 = torch.tensor([[1, 2], [3, 4]])

# From NumPy arrays (shares memory)
arr = np.array([[1.0, 2.0], [3.0, 4.0]])
t2 = torch.from_numpy(arr)

# Common initializations
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
rand_uniform = torch.rand(2, 2)
rand_normal = torch.randn(3, 3)

# Shape-aware creation
x = torch.tensor([1, 2, 3])
y = torch.zeros_like(x)

# Linearly spaced values
linear = torch.linspace(0, 10, steps=5)
```

These simple creation patterns appear everywhere in real models.

---

### Tensor Attributes You Must Know

Every tensor carries metadata that controls how it behaves.

```python
x = torch.randn(3, 4)

x.shape          # Dimensions
x.dtype          # Numeric precision
x.device         # CPU or GPU
x.requires_grad  # Gradient tracking
x.is_contiguous()
```

#### Common data types

* `float32` → default for deep learning
* `float64` → higher precision, slower
* `int64` → indices, token IDs
* `bool` → masks

Understanding `dtype` and device early will save you countless bugs later.

---

## Essential Tensor Operations

### Element-wise Operations

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

a + b
a * b
a ** 2
```

Operations happen element by element unless explicitly stated otherwise.

In-place operations end with `_`:

```python
a.add_(1)
```

Use them carefully as they modify memory directly.

---

### Matrix Operations: The Backbone of Neural Networks

```python
A = torch.randn(3, 4)
B = torch.randn(4, 2)

C = A @ B
```

Matrix multiplication is the **single most important operation** in deep learning. `Batch` operations matter even more:

```python
batch_A = torch.randn(10, 3, 4)
batch_B = torch.randn(10, 4, 2)
batch_C = torch.bmm(batch_A, batch_B)
```

This is how entire batches flow through neural networks efficiently.

---

### Broadcasting (Powerful, but Dangerous)

Broadcasting allows tensors of different shapes to interact:

```python
a = torch.randn(3, 4)
b = torch.randn(4)

a + b
```

This is convenient—but it can silently hide bugs. Always verify shapes when debugging unexpected results.

---

### Reshaping and Indexing

```python
x = torch.randn(2, 3, 4)

x.view(6, 4)
x.reshape(2, 12)

x.squeeze()
x.unsqueeze(0)
```

Indexing works exactly like NumPy:

```python
tensor = torch.arange(12).reshape(3, 4)
tensor[:, 1]
tensor[0:2, 2:4]
```

Master this once—it pays dividends forever.

---

## Device Management: CPU vs GPU

GPUs are what make deep learning practical at scale.

### Detecting available devices

```python
torch.cuda.is_available()
torch.backends.mps.is_available()
```

### Device-agnostic best practice

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000).to(device)

z = x @ y
```

All tensors involved in an operation **must live on the same device**.

---

## 2. Autograd: How PyTorch Learns

Autograd is PyTorch’s automatic differentiation engine. It tracks operations and computes gradients for you.

### The Computational Graph

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
```

Every operation builds a graph behind the scenes.

When you call:

```python
y.backward()
```

PyTorch applies the chain rule backward through that graph.

---

### Gradients Accumulate (This Matters)

```python
x.grad.zero_()
```

Gradients **do not reset automatically**. Forgetting this is one of the most common beginner mistakes.

---

### Disabling Gradients

For inference and evaluation:

```python
with torch.no_grad():
    y = model(x)
```

This saves memory and speeds up computation.

---

## 3. Gradient Descent from Scratch

Before using `torch.nn`, let’s train a model manually.

We’ll learn:

* Forward pass
* Loss computation
* Backward pass
* Parameter updates

```python
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

The training loop:

```python
for epoch in range(epochs):
    y_pred = w * X + b
    loss = ((y_pred - y_true) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()
```

This pattern underlies *every* deep learning system.

---

## Key Takeaways

* Tensors are the foundation of PyTorch.
* Devices matter, we write portable, device-agnostic code.
* Autograd handles differentiation automatically.
* Gradients accumulate unless cleared.
* Training is just forward → loss → backward → update.

Once these ideas click, PyTorch becomes intuitive.

---

## What Comes Next?

In the next part of this series, we’ll move from raw tensors to:

* `torch.nn.Module`
* Layers, parameters, and optimizers
* Clean, scalable training loops

This is where PyTorch starts to feel *powerful*.

---

## Jupyter Notebook

Practice alongside this article using the companion notebook:

**[Part 1: PyTorch Foundation – Colab](https://colab.research.google.com/drive/1Eh6A3ENCzDnkEr4m5dSNQbe_b489ZUK2?usp=drive_link)**