---
layout: post
title: "Part 1: The PyTorch Foundation"
date: 2025-12-15
series: "Deep Learning Series"
series_author: "Mayank Sharma"
series_image: "/assets/images/2025-12-15-pytorch-foundation-part1/pytorch-foundation-part1.png"
excerpt: "Master PyTorch fundamentals - tensors, autograd, and gradient descent. Learn dynamic computation graphs, GPU acceleration, and build your first neural network from scratch."
---

## Introduction: Why PyTorch?

If you ever decided to work with modern deep learning, especially NLP, vision, or generative AI, you will almost certainly encounter **PyTorch**. Over the past few years, PyTorch has become the *default* framework for AI research and is rapidly becoming just as important in production systems. But this dominance didn’t happen by accident.

PyTorch succeeds because it feels natural to humans.

You write normal Python code.  
You run it line by line.  
And what you write is *exactly* what the model executes.

That design choice shapes everything we’ll learn in this article.

## The Dynamic Graph Advantage

PyTorch uses **dynamic computation graphs**, often described as *define-by-run*. This means the computation graph is built **as your code runs**, not ahead of time.

Why does this matter?

- **Intuitive** – You write standard Python. No sessions. No placeholders
- **Debuggable** – `print()`, `pdb`, stack traces all work normally
- **Flexible** – You can change control flow, loops, and model structure on the fly

This is why PyTorch feels more like programming and less like configuration.

### Why researchers prefer PyTorch

- The majority of papers at NeurIPS, ICML, and ICLR use PyTorch
- It integrates seamlessly with NumPy and the Python ecosystem
- Hugging Face Transformers are built on PyTorch
- Research ideas move quickly from concept to code

### PyTorch in production

PyTorch is no longer “just for research”:

- **PyTorch 2.x** introduces `torch.compile` for major performance gains
- **TorchServe** enables scalable model serving
- **ONNX export** allows deployment across platforms
- Companies like Meta, Microsoft, and Tesla use PyTorch at scale

With that context, let’s start from the absolute foundation.

---

## 1. Tensors: The Core Data Structure

### What is a Tensor?

At its core, a **tensor** is a multi-dimensional array. If you’ve used NumPy arrays, you already understand the idea. Think of tensors as containers for numbers, arranged in different dimensions:

**Tensor Hierarchy:**

- **Rank 0** (Scalar): `5.0` → Single number.
- **Rank 1** (Vector): `[1, 2, 3]` → 1D array.
- **Rank 2** (Matrix): `[[1, 2], [3, 4]]` → 2D array.
- **Rank 3+** (Tensor): `[[[...]]]` → 3D+ arrays.

In deep learning, tensors represents *everything*:

- **Images**: (batch_size, channels, height, width) → Rank 4.
- **Text sequences**: (batch_size, sequence_length, embedding_dim) → Rank 3.
- **Model parameters**: Weights and biases of arbitrary shape.

If you understand tensors, you understand PyTorch.

### Creating Tensors

PyTorch provides multiple ways to create tensors:

```python
import torch
import numpy as np

# From Python lists
tensor_from_list = torch.tensor([[1, 2], [3, 4]])
print(tensor_from_list)
# tensor([[1, 2],
#         [3, 4]])

# From NumPy arrays (shares memory by default)
numpy_array = np.array([[1.0, 2.0], [3.0, 4.0]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(tensor_from_numpy)
# tensor([[1., 2.],
#         [3., 4.]], dtype=torch.float64)

# Common initialization patterns
zeros = torch.zeros(3, 4)  # 3x4 matrix of zeros
ones = torch.ones(2, 3)    # 2x3 matrix of ones
random_uniform = torch.rand(2, 2)  # Uniform [0, 1)
random_normal = torch.randn(3, 3)  # Normal N(0, 1)

# Create tensor with same shape as another
x = torch.tensor([1, 2, 3])
y = torch.zeros_like(x)
print(y)  # tensor([0, 0, 0])

# Linearly spaced values
linear = torch.linspace(0, 10, steps=5)
print(linear)  # tensor([ 0.0, 2.5, 5.0, 7.5, 10.0])
```

### Understanding Tensor Attributes

Every tensor carries `metadata` that controls how it behaves.

```python
x = torch.randn(3, 4)

# Shape: dimensions of the tensor
print(x.shape)        # torch.Size([3, 4])
print(x.size())       # Equivalent to .shape

# Data type: precision and numeric format
print(x.dtype)        # torch.float32 (default)

# Device: where tensor lives (CPU or GPU)
print(x.device)       # cpu

# Gradient tracking: whether autograd records operations
print(x.requires_grad)  # False (default)

# Memory layout
print(x.is_contiguous())  # True (C-contiguous memory)
```

**Common data types:**

```python
# Integer types
int_tensor = torch.tensor([1, 2], dtype=torch.int32)    # 32-bit integer
long_tensor = torch.tensor([1, 2], dtype=torch.int64)   # 64-bit integer

# Floating-point types
float_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)  # Single precision
double_tensor = torch.tensor([1.0, 2.0], dtype=torch.float64) # Double precision

# Boolean and complex types
bool_tensor = torch.tensor([True, False], dtype=torch.bool)
complex_tensor = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
```

Common data types

- `float32` → default for deep learning
- `float64` → higher precision, slower
- `int64` → indices, token IDs
- `bool` → masks

Understanding `dtype` and device early will save you countless bugs later.

### Essential Tensor Operations

#### Element-wise Operations

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Basic arithmetic (element-wise)
print(a + b)          # tensor([5., 7., 9.])
print(a - b)          # tensor([-3., -3., -3.])
print(a * b)          # tensor([4., 10., 18.])
print(a / b)          # tensor([0.25, 0.40, 0.50])
print(a ** 2)         # tensor([1., 4., 9.])

# In-place operations (append underscore)
a.add_(1)             # a becomes [2., 3., 4.]
print(a)

# Mathematical functions
print(torch.sqrt(a))  # Element-wise square root
print(torch.exp(a))   # Element-wise exponential
print(torch.log(a))   # Element-wise natural log
```

#### Matrix Operations: The Heart of Deep Learning

```python
# Matrix multiplication: THE most important operation
A = torch.randn(3, 4)  # 3 rows, 4 columns
B = torch.randn(4, 2)  # 4 rows, 2 columns

# Three equivalent ways to multiply matrices
C1 = torch.matmul(A, B)    # Explicit function
C2 = A @ B                  # Python 3.5+ operator (RECOMMENDED)
C3 = A.mm(B)                # Method version

print(C1.shape)  # torch.Size([3, 2])

# Batch matrix multiplication (crucial for neural networks)
batch_A = torch.randn(10, 3, 4)  # 10 matrices of shape 3×4
batch_B = torch.randn(10, 4, 2)  # 10 matrices of shape 4×2
batch_C = torch.bmm(batch_A, batch_B)  # 10 matrices of shape 3×2
print(batch_C.shape)  # torch.Size([10, 3, 2])

# Dot product (1D tensors)
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])
dot_product = torch.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32.0
print(dot_product)

# Transpose operations
matrix = torch.randn(3, 5)
print(matrix.T.shape)              # torch.Size([5, 3])
print(matrix.transpose(0, 1).shape)  # Same as .T

# Advanced: Einstein summation (powerful but complex)
# Matrix multiplication using einsum
result = torch.einsum('ij,jk->ik', A, B)
```

#### Broadcasting: NumPy-style Array Operations

Broadcasting allows operations on tensors of different shapes:

```python
# Broadcasting rules:
# 1. Dimensions are compared from right to left
# 2. Dimensions must be equal, one of them is 1, or one doesn't exist

a = torch.randn(3, 4)
b = torch.randn(4)      # Can broadcast to (3, 4)

result = a + b          # b is broadcast to each row of a
print(result.shape)     # torch.Size([3, 4])

# Common patterns
matrix = torch.randn(5, 3)
column_vector = torch.randn(5, 1)
row_vector = torch.randn(1, 3)

print((matrix + column_vector).shape)  # (5, 3) - add to each column
print((matrix + row_vector).shape)     # (5, 3) - add to each row

# WARNING: Broadcasting can hide bugs!
x = torch.randn(3, 4)
y = torch.randn(3, 1)  # Intended: (3, 4)?
z = x + y  # Silently broadcasts instead of erroring
```

#### Reshaping and Indexing

```python
x = torch.randn(2, 3, 4)

# View: returns a new tensor sharing the same data
y = x.view(6, 4)        # Reshape to 6×4 (must preserve total elements)
print(y.shape)          # torch.Size([6, 4])

# Reshape: like view but copies data if necessary
z = x.reshape(2, 12)    # More flexible than view
print(z.shape)          # torch.Size([2, 12])

# Squeeze and unsqueeze: remove/add dimensions of size 1
a = torch.randn(1, 3, 1, 4)
b = a.squeeze()         # Remove all dimensions of size 1
print(b.shape)          # torch.Size([3, 4])

c = torch.randn(3, 4)
d = c.unsqueeze(0)      # Add dimension at position 0
print(d.shape)          # torch.Size([1, 3, 4])

# Advanced indexing
tensor = torch.arange(12).reshape(3, 4)
print(tensor)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

print(tensor[0, :])     # First row: tensor([0, 1, 2, 3])
print(tensor[:, 1])     # Second column: tensor([1, 5, 9])
print(tensor[0:2, 2:4]) # Submatrix slice

# Boolean indexing
mask = tensor > 5
print(tensor[mask])     # tensor([ 6,  7,  8,  9, 10, 11])

# Fancy indexing
indices = torch.tensor([0, 2])
print(tensor[indices])  # Rows 0 and 2
```

### Device Management: CPU vs. GPU

GPU acceleration is PyTorch's killer feature for deep learning. Here we will look at how to use GPUs in PyTorch.

#### Checking GPU Availability

```python
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("No GPU available, using CPU")

# Check if MPS (Apple Silicon GPU) is available
if torch.backends.mps.is_available():
    print("Apple Silicon GPU available")
```

#### Moving Tensors Between Devices

```python
# Create tensor on CPU (default)
cpu_tensor = torch.randn(3, 4)
print(cpu_tensor.device)  # cpu

# Move to GPU
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to('cuda')
    # Alternative: gpu_tensor = cpu_tensor.cuda()
    print(gpu_tensor.device)  # cuda:0

    # Move back to CPU
    back_to_cpu = gpu_tensor.to('cpu')
    # Alternative: back_to_cpu = gpu_tensor.cpu()
    print(back_to_cpu.device)  # cpu

# For Apple Silicon
if torch.backends.mps.is_available():
    mps_tensor = cpu_tensor.to('mps')
    print(mps_tensor.device)  # mps:0
```

#### Device-Agnostic Code Pattern (BEST PRACTICE)

```python
# Set device dynamically
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create tensors directly on the target device
x = torch.randn(1000, 1000, device=device)

# Move existing tensors
y = torch.randn(1000, 1000)
y = y.to(device)

# Operations must be on the same device
z = x @ y  # Works: both on same device

# This would ERROR:
# a = torch.randn(10, device='cpu')
# b = torch.randn(10, device='cuda')
# c = a + b  # RuntimeError: Expected all tensors on the same device
```

#### Performance Considerations

```python
import time

# CPU computation
cpu_x = torch.randn(5000, 5000)
cpu_y = torch.randn(5000, 5000)

start = time.time()
cpu_z = cpu_x @ cpu_y
print(f"CPU time: {time.time() - start:.4f} seconds")

# GPU computation (with proper synchronization)
if torch.cuda.is_available():
    gpu_x = cpu_x.to('cuda')
    gpu_y = cpu_y.to('cuda')

    # Warm-up run (GPU kernel compilation)
    _ = gpu_x @ gpu_y

    torch.cuda.synchronize()  # Wait for GPU to finish
    start = time.time()
    gpu_z = gpu_x @ gpu_y
    torch.cuda.synchronize()
    print(f"GPU time: {time.time() - start:.4f} seconds")

    # Typical speedup: 10-100x for large matrices
```

**Key takeaways:**

- Always check `torch.cuda.is_available()` before using GPU
- Use device-agnostic patterns for portable code
- Operations require tensors on the same device
- GPU shines for large-scale operations; overhead hurts small operations

---

## 2. Autograd: The Engine of Learning

PyTorch's automatic differentiation engine, **autograd**, is what makes training neural networks practical. Instead of manually deriving and coding gradients, autograd computes them automatically.

### The Computational Graph

When you perform operations on tensors with `requires_grad=True`, PyTorch builds a **directed acyclic graph (DAG)** tracking the computation:

- **Nodes**: Tensors (data)
- **Edges**: Operations (functions)

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward pass: build computational graph
z = x * y + y ** 2
print(z)  # tensor(15., grad_fn=<AddBackward0>)

# The graph looks like:
#     x (2.0)    y (3.0)
#        \\      / \\
#         \\    /   \\
#          \\  /     \\
#          mul      pow
#            \\      /
#             \\    /
#              \\  /
#               add
#                |
#              z (15.0)
```

So what is the `grad_fn`? It's the function that created the tensor. For example, `z` has `grad_fn=<AddBackward0>` because it was created by an addition operation. This is how PyTorch knows how to compute gradients during backpropagation. Each tensor resulting from an operation stores its `grad_fn`, a reference to the function that created it. This enables backpropagation.

### Computing Gradients with `.backward()`

The magic happens when we call `.backward()`:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

# Compute gradients
y.backward()

# Access gradient: dy/dx = 2x + 3
print(x.grad)  # tensor(7.) = 2(2) + 3
```

**How it works:**

1. When you apply `y.backward()`, PyTorch traverses the graph in reverse (topological order)
2. Applies chain rule at each node
3. Accumulates gradients in `.grad` attribute of leaf tensors

#### Multi-variable Example

```python
a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)

# Function: f(a, b) = a^2 * b + b^3
f = a ** 2 * b + b ** 3

f.backward()

# Gradients:
# df/da = 2a * b = 2(3)(4) = 24
# df/db = a^2 + 3b^2 = 9 + 3(16) = 57
print(a.grad)  # tensor(24.)
print(b.grad)  # tensor(57.)
```

### Vector-Valued Functions

For scalar outputs, `.backward()` is straightforward. But, for vector outputs, you must specify the gradient:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# y is a vector, so backward needs a gradient vector
# This represents the Jacobian-vector product
gradient = torch.tensor([1.0, 1.0, 1.0])
y.backward(gradient)

print(x.grad)  # tensor([2., 4., 6.]) = 2x evaluated at each element
```

### Gradient Accumulation and Zeroing

**CRITICAL**: It's important to note that Gradients **accumulate** by default. You `must zero` them between iterations.

```python
x = torch.tensor(2.0, requires_grad=True)

# First computation
y1 = x ** 2
y1.backward()
print(x.grad)  # tensor(4.)

# Second computation WITHOUT zeroing
y2 = x ** 3
y2.backward()
print(x.grad)  # tensor(16.) = 4 + 12 (ACCUMULATED!)

# Proper way: zero gradients
x.grad.zero_()
y3 = x ** 3
y3.backward()
print(x.grad)  # tensor(12.) (correct)
```

**When gradient accumulation is useful:**

- When you are simulating larger batch sizes with limited memory
- When you want to accumulate gradients across multiple loss terms

### Detaching from the Graph

Sometimes you want to use a tensor's value without tracking gradients:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

# Detach y from computational graph
y_detached = y.detach()
print(y_detached.requires_grad)  # False

# No gradients will flow through y_detached
z = y_detached * 3
z.backward()  # ERROR: z is not part of graph
```

**Use cases:**

- Implementing certain loss functions
- Debugging: isolate problematic computations
- Performance: avoid unnecessary gradient computation

### Context Manager: `torch.no_grad()`

For `inference` or `evaluation`, you must disable gradient tracking entirely:

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)

# Normal operation (gradients tracked)
y = x ** 2

# Disable gradient tracking
with torch.no_grad():
    y_no_grad = x ** 2
    print(y_no_grad.requires_grad)  # False

# Also useful for updating parameters without tracking
with torch.no_grad():
    x -= 0.1 * x.grad
```

**Benefits:**

- It reduces memory consumption
- It speeds up computation (no need to store intermediate values)
- It is essential for inference and evaluation

---

## 3. Putting It Together: Gradient Descent from Scratch

Now that we have a solid understanding of PyTorch's basic core concepts. Let's implement `linear regression` using only tensors and autograd, without using `torch.nn` yet.

### Problem Setup

We want to learn a linear function: `y = w * x + b`

```python
import torch
import matplotlib.pyplot as plt

# Generate synthetic data: y = 3x + 2 + noise
torch.manual_seed(42)
X = torch.randn(100, 1)
y_true = 3 * X + 2 + 0.5 * torch.randn(100, 1)

# Initialize parameters (random initialization)
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print(f"Initial w: {w.item():.4f}, b: {b.item():.4f}")
```

### Training Loop

```python
learning_rate = 0.01
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    # Forward pass: compute predictions
    y_pred = w * X + b

    # Compute loss (Mean Squared Error)
    loss = ((y_pred - y_true) ** 2).mean()
    losses.append(loss.item())

    # Backward pass: compute gradients
    loss.backward()

    # Update parameters (gradient descent)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # CRITICAL: Zero gradients for next iteration
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print(f"\\nFinal parameters: w={w.item():.4f}, b={b.item():.4f}")
print(f"True parameters:  w=3.0000, b=2.0000")
```

### Visualizing Results

```python
# Plot loss curve
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# Plot predictions vs. true values
plt.subplot(1, 2, 2)
with torch.no_grad():
    y_final = w * X + b

plt.scatter(X.numpy(), y_true.numpy(), alpha=0.5, label='True data')
plt.scatter(X.numpy(), y_final.numpy(), alpha=0.5, label='Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Results')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('linear_regression_results.png')
print("Visualization saved to 'linear_regression_results.png'")
```

### Important Points to Remember

1. **Forward Pass**: Compute predictions using current parameters.
2. **Loss Calculation**: Quantify prediction error.
3. **Backward Pass**: Compute gradients via `loss.backward()`.
4. **Parameter Update**: Move in direction of negative gradient.
5. **Gradient Zeroing**: Essential to prevent accumulation.

This pattern forms the foundation of ALL deep learning training in PyTorch.

---

### Conclusion

By implementing linear regression from scratch, you've learned:

- **Tensors** are the fundamental data structure—understand creation, operations, and attributes.
- **Device management** is crucial—write device-agnostic code for portability.
- **Autograd** automatically computes gradients through computational graphs.
- **`.backward()`** propagates gradients; always zero them between iterations.
- **Gradient descent** can be implemented from scratch using only tensors and autograd.

---

## Jupyter Notebook

For hands-on practice, check out the companion notebooks - [Part1: PyTorch Foundation](https://colab.research.google.com/drive/1Eh6A3ENCzDnkEr4m5dSNQbe_b489ZUK2?usp=drive_link)