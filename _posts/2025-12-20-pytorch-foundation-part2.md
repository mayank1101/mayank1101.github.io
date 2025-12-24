---
layout: post
title: "Part 2: Deep Learning with PyTorch"
date: 2025-12-20
series: "NLP Mastery Series"
series_author: "Mayank Sharma"
series_image: "/assets/images/2025-12-20-pytorch-foundation-part2/pytorch-foundation-part2.png"
excerpt: "Build production-ready neural networks using torch.nn, modern optimizers, and efficient data pipelines. Train a complete CNN for image classification following industry best practices."
---

## Building Production-Ready Neural Networks

## Introduction

In **Part 1**, we built learning systems from scratch using tensors and autograd. That exercise was important—it showed *how* learning really works under the hood.

But no one builds real-world deep learning systems that way.

Imagine manually writing every layer, activation function, optimizer, and training loop detail for a 100-layer ResNet or a large transformer. It would be slow, error-prone, and impossible to maintain.

This is where PyTorch’s **high-level abstractions** come in.

In this article, we move from *learning mechanics* to *engineering practice*.

You’ll learn how PyTorch helps you:
- Build complex models cleanly using `torch.nn`
- Train them efficiently with `torch.optim`
- Feed data at scale using `Dataset` and `DataLoader`
- Structure training and validation loops the way real teams do

By the end, you’ll train a complete CNN for image classification using industry-standard patterns.

---

## 1. The `torch.nn` Module

### Why `nn.Module` Exists

Every neural network in PyTorch inherits from `nn.Module`.

This base class is not just a formality—it solves several hard problems for you:

- Automatically tracks learnable parameters
- Allows clean composition of layers and submodules
- Handles device transfers (`.to(device)`)
- Switches between training and evaluation behavior
- Enables saving and loading model state

In short, `nn.Module` turns raw tensor code into *maintainable software*.

---

### A Minimal `nn.Module`

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

model = SimpleModel()
x = torch.randn(5, 10)
y = model(x)

print(y.shape)  # torch.Size([5, 1])
````

**Key ideas to internalize:**

* Define layers in `__init__`
* Define computation in `forward`
* Calling `model(x)` automatically runs `forward(x)`
* All parameters are tracked automatically—no manual bookkeeping

This pattern never changes, even in very large models.

---

## 2. Essential Building Blocks

### Fully Connected (Linear) Layers

```python
linear = nn.Linear(128, 64)

x = torch.randn(32, 128)
y = linear(x)

print(y.shape)  # [32, 64]
```

A linear layer performs:

[
y = xW^T + b
]

**Where they are used:**

* Classification heads
* MLPs
* Projection layers in transformers

---

### Convolutional Layers

```python
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

x = torch.randn(8, 3, 32, 32)
y = conv(x)

print(y.shape)  # [8, 16, 32, 32]
```

Convolutions extract **local spatial patterns**—edges, textures, shapes.

The padding here preserves spatial size, which simplifies model design.

---

### Pooling Layers

Pooling reduces spatial resolution while keeping important features.

```python
maxpool = nn.MaxPool2d(2, 2)
avgpool = nn.AvgPool2d(2, 2)
adaptive = nn.AdaptiveAvgPool2d((1, 1))
```

Pooling helps:

* Reduce computation
* Increase receptive field
* Add translation invariance

---

### Batch Normalization

Batch normalization stabilizes training by normalizing activations.

```python
bn = nn.BatchNorm2d(16)
```

**Why it matters:**

* Enables higher learning rates
* Makes training less sensitive to initialization
* Improves convergence speed

⚠️ **Important:** BatchNorm behaves differently during training and evaluation.

```python
model.train()
model.eval()
```

Always switch modes correctly.

---

### Dropout

Dropout randomly disables neurons during training.

```python
dropout = nn.Dropout(p=0.5)
```

It helps prevent overfitting by discouraging reliance on any single feature.

Dropout is **automatically disabled during evaluation**.

---

## 3. Activation Functions

Non-linearities give neural networks their expressive power.

```python
nn.ReLU()
nn.LeakyReLU(0.01)
nn.GELU()
nn.Sigmoid()
nn.Tanh()
```

**Practical guidance:**

* ReLU → default for CNNs and MLPs
* Leaky ReLU → if many neurons die
* GELU → transformers and modern architectures
* Sigmoid → binary outputs
* Tanh → bounded, zero-centered outputs

---

## 4. Loss Functions

Loss functions tell the model *how wrong* it is.

```python
criterion = nn.CrossEntropyLoss()
```

Key rules:

* Use **raw logits** (no softmax) with `CrossEntropyLoss`
* Use `BCEWithLogitsLoss` for binary classification
* Use `MSELoss` for regression

Choosing the correct loss is just as important as choosing the model.

---

## 5. Model Construction Patterns

### Sequential (Quick, but Limited)

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

Good for prototypes, but not scalable.

---

### Custom Modules (Recommended)

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)
```

---

### Modular Composition (Best Practice)

Reusable blocks lead to clean, extensible architectures.

```python
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
```

This pattern scales naturally to large CNNs and transformers.

---

## 6. Data Handling with `Dataset` and `DataLoader`

### Why Data Pipelines Matter

A slow data pipeline can waste a powerful GPU.

PyTorch separates:

* **Dataset** → how data is stored
* **DataLoader** → how data is fed

---

### Custom Dataset

```python
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
```

---

### DataLoader

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

Key settings:

* `shuffle=True` for training
* `num_workers` > 0 for speed
* `pin_memory=True` for GPUs

---

## 7. Optimization with `torch.optim`

Optimizers decide *how* the model updates its parameters.

```python
optim.SGD(...)
optim.Adam(...)
optim.AdamW(...)
```

**Practical defaults:**

* Start with **AdamW**
* Tune learning rate before anything else
* Add schedulers for long training runs

---

### Learning Rate Scheduling

Learning rate should change over time.

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

Schedulers often make a bigger difference than changing models.

---

## 8. Training and Validation Loops

### Training Loop

```python
model.train()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Validation Loop

```python
model.eval()
with torch.no_grad():
    ...
```

These two modes must never be mixed.

---

## 9. Case Study: CNN on CIFAR-10

You now have everything needed to build a real image classifier.

* 60,000 RGB images
* 10 classes
* Standard benchmark for CNNs

The architecture combines:

* Convolution blocks
* Batch normalization
* Dropout
* A clean classification head

This mirrors how real production models are structured.

---

## Key Takeaways

* `nn.Module` is the backbone of PyTorch models
* Modular design beats monolithic code
* Data pipelines are just as important as models
* Optimizers and schedulers control learning dynamics
* Training loops follow consistent, repeatable patterns

If Part 1 taught you *how learning works*, Part 2 taught you *how deep learning is built in practice*.

---

## Jupyter Notebook

Practice alongside this article:

**[Part 2: PyTorch Foundation – Colab](https://colab.research.google.com/drive/1CTMO_KYfnfIMpkzNcRVDvs455HM8pX50?usp=sharing)**
