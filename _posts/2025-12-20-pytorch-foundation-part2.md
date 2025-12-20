---
layout: post
title: "Part 2: Deep Learning with PyTorch"
date: 2025-12-20
---

<!-- # Part 2: Deep Learning with PyTorch -->

## Building Production-Ready Neural Networks

## Introduction

In Part 1, we built everything from scratch using raw tensors and autograd. While educational, this approach doesn't scale to real-world deep learning projects. Imagine manually implementing every layer, activation function, and optimizer for a 100-layer ResNet!

**Part 2 introduces the high-level abstractions** that make PyTorch the framework of choice for both researchers and practitioners:

- `torch.nn`: Building blocks for neural networks
- `torch.optim`: Advanced optimization algorithms
- `torch.utils.data`: Efficient data pipelines
- Training loops and best practices

By the end of this article, you'll build and train a complete Convolutional Neural Network (CNN) for image classification, following industry-standard patterns.

---

## 1. The `torch.nn` Module

### Understanding `nn.Module`

Every neural network in PyTorch inherits from `nn.Module`. This base class provides:
- Automatic parameter tracking
- Easy model composition (nesting modules)
- GPU transfer with `.to(device)`
- Training/eval mode switching
- State saving/loading

**The minimal example:**

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()  # Always call parent constructor
        # Define layers here
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 1)

    def forward(self, x):
        # Define forward pass
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

# Instantiate and use
model = SimpleModel()
input_data = torch.randn(5, 10)  # Batch of 5 samples
output = model(input_data)  # Calls forward()
print(output.shape)  # torch.Size([5, 1])
```

**Key concepts:**
- `__init__`: Define all layers and parameters
- `forward()`: Define the computation graph
- Calling `model(x)` automatically invokes `forward(x)`
- All `nn.Module` parameters are tracked automatically

### Essential Layers

#### Fully Connected (Linear) Layers

```python
# nn.Linear(in_features, out_features, bias=True)
linear = nn.Linear(128, 64)

# What it does: y = xW^T + b
x = torch.randn(32, 128)  # Batch size 32
y = linear(x)
print(y.shape)  # torch.Size([32, 64])

# Access parameters
print(f"Weight shape: {linear.weight.shape}")  # [64, 128]
print(f"Bias shape: {linear.bias.shape}")      # [64]
```

**Use cases:**
- Classification heads
- Fully connected layers in MLPs
- Projection layers in transformers

#### Convolutional Layers

```python
# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

# Input: (batch, channels, height, width)
x = torch.randn(8, 3, 32, 32)  # 8 RGB images of 32×32
y = conv(x)
print(y.shape)  # torch.Size([8, 16, 32, 32])

# Key parameters explained:
# - in_channels: Input depth (3 for RGB)
# - out_channels: Number of filters
# - kernel_size: Filter size (3 = 3×3)
# - stride: Step size (default 1)
# - padding: Zero-padding (1 maintains spatial dims)
```

**Spatial dimension formula:**
```python
output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
```

**Example:** Input 32, kernel 3, padding 1, stride 1
```python
output = floor((32 + 2*1 - 3) / 1) + 1 = 32
```

#### Pooling Layers

```python
# Max pooling: takes maximum in each window
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.randn(8, 16, 32, 32)
y = maxpool(x)
print(y.shape)  # torch.Size([8, 16, 16, 16]) - halved spatial dims

# Average pooling: takes average
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
y_avg = avgpool(x)
print(y_avg.shape)  # torch.Size([8, 16, 16, 16])

# Adaptive pooling: output size independent of input size
adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output always 1×1
y_adaptive = adaptive_pool(x)
print(y_adaptive.shape)  # torch.Size([8, 16, 1, 1])
```

**Use cases:**
- Downsampling feature maps
- Reducing spatial dimensions
- Translation invariance

#### Batch Normalization

Normalizes activations to have mean 0 and variance 1, improving training stability.

```python
# For 2D data (MLPs)
bn1d = nn.BatchNorm1d(num_features=128)
x = torch.randn(32, 128)
y = bn1d(x)

# For images (CNNs)
bn2d = nn.BatchNorm2d(num_features=16)
x = torch.randn(8, 16, 32, 32)
y = bn2d(x)

# How it works:
# 1. Compute mean and std across batch
# 2. Normalize: (x - mean) / sqrt(var + eps)
# 3. Apply learned scale (gamma) and shift (beta)
```

**Benefits:**
- Accelerates training (enables higher learning rates)
- Reduces sensitivity to initialization
- Acts as regularization (slight noise from batch statistics)

**Important:** Behaves differently in training vs. eval mode!

```python
model.train()  # Updates running statistics
model.eval()   # Uses fixed statistics
```

#### Dropout

Randomly sets activations to zero during training for regularization.

```python
dropout = nn.Dropout(p=0.5)  # Drop 50% of activations

x = torch.randn(32, 128)
model.train()
y_train = dropout(x)  # Some elements zeroed out

model.eval()
y_eval = dropout(x)   # No dropout during inference
```

**Key points:**
- Only active during training (`model.train()`)
- Automatically scaled by 1/(1-p) to maintain expected sum
- Prevents overfitting by reducing co-adaptation

### Activation Functions

Non-linear activations are essential for neural networks to learn complex patterns.

```python
# ReLU: Most common activation
relu = nn.ReLU()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(relu(x))  # tensor([0., 0., 0., 1., 2.])

# Leaky ReLU: Addresses dying ReLU problem
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
print(leaky_relu(x))  # tensor([-0.02, -0.01, 0., 1., 2.])

# GELU: Smooth activation used in transformers
gelu = nn.GELU()
print(gelu(x))

# Sigmoid: Maps to (0, 1)
sigmoid = nn.Sigmoid()
print(sigmoid(x))

# Tanh: Maps to (-1, 1)
tanh = nn.Tanh()
print(tanh(x))
```

**When to use:**
- **ReLU**: Default choice for hidden layers
- **Leaky ReLU**: If experiencing dying ReLU (many zero activations)
- **GELU**: Modern transformers and vision models
- **Sigmoid**: Binary classification output
- **Tanh**: When outputs should be centered around zero

### Loss Functions

```python
# Cross-Entropy Loss: Multi-class classification
# Combines nn.LogSoftmax and nn.NLLLoss
criterion = nn.CrossEntropyLoss()

# Predictions: raw logits (before softmax)
logits = torch.randn(32, 10)  # 32 samples, 10 classes
targets = torch.randint(0, 10, (32,))  # Class indices

loss = criterion(logits, targets)
print(loss)

# Binary Cross-Entropy: Binary classification
# Combines sigmoid and BCE loss
bce_loss = nn.BCEWithLogitsLoss()
logits = torch.randn(32, 1)
targets = torch.randint(0, 2, (32, 1)).float()
loss = bce_loss(logits, targets)

# Mean Squared Error: Regression
mse_loss = nn.MSELoss()
predictions = torch.randn(32, 1)
targets = torch.randn(32, 1)
loss = mse_loss(predictions, targets)
```

### Building Complex Models

**Pattern 1: Sequential models**

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)

# Simple but inflexible
x = torch.randn(32, 784)
output = model(x)
```

**Pattern 2: Custom modules (RECOMMENDED)**

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = MLP(input_size=784, hidden_size=256, num_classes=10)
```

**Pattern 3: Modular composition**

```python
class ConvBlock(nn.Module):
    """Reusable Conv-BN-ReLU block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

---

## 2. Data Handling with `DataLoader`

Efficient data loading is crucial for training performance.

### The Dataset Abstraction

Custom datasets implement two methods:
- `__len__()`: Return number of samples
- `__getitem__(idx)`: Return sample at index

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target

# Example usage
data = torch.randn(1000, 28, 28)  # 1000 images
targets = torch.randint(0, 10, (1000,))  # Labels

dataset = CustomDataset(data, targets)
print(len(dataset))  # 1000
sample, label = dataset[0]
print(sample.shape, label)
```

### DataLoader: Batching and Shuffling

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,      # Shuffle at every epoch
    num_workers=4,     # Parallel data loading
    pin_memory=True,   # Faster GPU transfer
    drop_last=False    # Drop incomplete last batch?
)

# Iterate over batches
for batch_idx, (data, targets) in enumerate(dataloader):
    print(f"Batch {batch_idx}: data shape = {data.shape}, targets shape = {targets.shape}")
    # data: [32, 28, 28], targets: [32]
```

**Key parameters:**
- `batch_size`: Number of samples per batch
- `shuffle`: Randomize order (True for training)
- `num_workers`: Parallel processes for data loading (0 = main process)
- `pin_memory`: Use pinned memory for faster GPU transfer
- `drop_last`: Drop last incomplete batch (useful for batch norm)

### Working with Image Data

```python
from torchvision import datasets, transforms

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor [0, 1]
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load MNIST dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

# Inspect batch
images, labels = next(iter(train_loader))
print(f"Images shape: {images.shape}")  # [64, 1, 28, 28]
print(f"Labels shape: {labels.shape}")  # [64]
```

### Data Augmentation

Critical for improving generalization on image tasks.

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# No augmentation for validation/test
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

---

## 3. Optimization with `torch.optim`

### Optimizer Fundamentals

Optimizers update model parameters based on gradients.

```python
import torch.optim as optim

model = SimpleModel()

# Stochastic Gradient Descent (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD with momentum (smoother updates)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam: Adaptive learning rates (most popular)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# AdamW: Adam with decoupled weight decay
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**Choosing an optimizer:**
- **SGD with momentum**: Simple, well-understood, good for final fine-tuning
- **Adam**: Default choice, adaptive learning rates, fast convergence
- **AdamW**: Modern best practice, fixes weight decay in Adam

### Learning Rate Scheduling

Learning rate decay improves convergence.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step decay: Reduce LR every N epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Exponential decay
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Cosine annealing: Smooth decay
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Reduce on plateau: Adaptive based on validation loss
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10
)

# Usage in training loop
for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)

    scheduler.step()  # For Step, Exponential, Cosine
    # OR
    scheduler.step(val_loss)  # For ReduceLROnPlateau
```

---

## 4. The Training Loop

The standard training pattern in PyTorch.

### Basic Training Loop

```python
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # Set to training mode
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data to device
        data, targets = data.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss
```

### Validation Loop

```python
def validate(model, val_loader, criterion, device):
    model.eval()  # Set to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy
```

### Complete Training Pipeline

```python
def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")

    return history
```

---

## 5. Case Study: CNN for CIFAR-10

Let's build a complete image classifier.

### The CIFAR-10 Dataset

- 60,000 32×32 color images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training, 10,000 test images

### Model Architecture

```python
class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Dropout(0.2),

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Dropout(0.3),

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
            nn.Dropout(0.4),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Model summary
model = CIFAR10_CNN()
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

### Data Preparation

```python
from torchvision import datasets, transforms

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
```

### Training

```python
# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CIFAR10_CNN().to(device)

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=100,
    steps_per_epoch=len(train_loader)
)

# Train
history = train_model(model, train_loader, test_loader, num_epochs=100)
```

---

## Jupyter Notebook 

For hands-on practice, check out the companion notebooks:

1. [Part2: PyTorch Foundation](https://github.com/mayank1101/mayank1101.github.io/blob/main/notebooks/2025-12-20-pytorch-foundation-part2/pytorch-foundation-tutorial-part2.ipynb)

## Next Steps

In **Part 3: FeedForward and Backward propagation**, we'll explore:
- Feedforward and backward propagation from scratch
- Understanding the math behind backpropagation in detail

**Before moving forward**, ensure you can:
- Build custom `nn.Module` classes
- Create `Dataset` and `DataLoader` pipelines
- Implement complete training loops
- Train a CNN on CIFAR-10

---

*Part of the NLP Mastery series by TensorTales*
