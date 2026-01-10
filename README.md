# ResNet-50-Dual

## Project Overview
This project provides a PyTorch implementation of the ResNet-50 (Residual Network) algorithm. The implementation strictly follows the original ResNet paper, including the network structure design, convolution layer parameters, stride settings, and batch normalization positions.

## Features
- **PyTorch Implementation**: Uses PyTorch framework, includes complete ResNet-50 architecture
- **Data Preprocessing**: Built-in data preprocessing pipeline for ImageNet dataset
- **Training & Validation**: Complete training and validation workflows
- **Support for ImageNet**: Designed for ImageNet dataset input size (224×224×3)
- **MIT License**: Open source under the MIT License

## Algorithm Principle

### Residual Learning
The core idea of ResNet is residual learning, which addresses the degradation problem in deep neural networks. Instead of directly fitting the desired underlying mapping  H(x) , residual networks fit a residual mapping  F(x) = H(x) - x . The original mapping is then expressed as  H(x) = F(x) + x .

### Skip Connections
Skip connections (also known as shortcut connections) enable the gradient to flow directly through the network during backpropagation, preventing vanishing gradient problems. They add the input of a layer directly to its output, allowing the network to learn residual functions with reference to the layer inputs.

### Bottleneck Structure
ResNet-50 uses bottleneck residual blocks, which consist of three convolutional layers: a 1×1 convolution (for dimension reduction), a 3×3 convolution (for feature extraction), and another 1×1 convolution (for dimension restoration). This design reduces computational complexity while maintaining representational power.

### Degradation Problem Solution
Deep networks often suffer from degradation, where accuracy saturates and then degrades rapidly with increasing depth. ResNet solves this by allowing the network to learn identity mappings through skip connections, ensuring that adding more layers cannot degrade performance compared to shallower networks.

## Python Implementation

### Requirements
- Python 3.7+
- PyTorch 1.8+
- torchvision

### Usage
```python
import torch
from ResNet-50 import resnet50

# Create model
model = resnet50()

# Forward pass
input = torch.randn(1, 3, 224, 224)
output = model(input)
print(output.shape)
```

### Training
```bash
python ResNet-50.py
```

## Network Architecture
The ResNet-50 architecture consists of:
- 1 initial convolutional layer (7×7, stride 2)
- 1 max pooling layer (3×3, stride 2)
- 4 residual blocks with bottleneck structures:
  - Block 1: 3 bottleneck layers, 64 output channels
  - Block 2: 4 bottleneck layers, 128 output channels
  - Block 3: 6 bottleneck layers, 256 output channels
  - Block 4: 3 bottleneck layers, 512 output channels
- 1 adaptive average pooling layer
- 1 fully connected layer (1000 classes for ImageNet)

## Future Plans
- **C++ Implementation**: Will introduce PyTorch C++ API (libtorch) implementation for improved performance in production environments
- **Model Optimization**: Add quantization and pruning support for deployment
- **More ResNet Variants**: Support ResNet-18, ResNet-34, ResNet-101, and ResNet-152 architectures
- **Pretrained Models**: Provide pretrained weights for various datasets

## License
MIT License

---

# PyTorch-ResNet50

## 项目概述
本项目提供了ResNet-50（残差网络）算法的PyTorch实现。实现严格遵循ResNet原论文，包括网络结构设计、卷积层参数、步长设置和批归一化位置等关键细节。

## 功能特性
- **PyTorch实现**：使用PyTorch框架，包含完整的ResNet-50架构
- **数据预处理**：内置ImageNet数据集的数据预处理流程
- **训练与验证**：完整的训练和验证工作流程
- **支持ImageNet**：针对ImageNet数据集输入尺寸（224×224×3）设计
- **MIT许可证**：基于MIT许可证开源

## 算法原理

### 残差学习
ResNet的核心思想是残差学习，用于解决深度神经网络中的退化问题。残差网络不直接拟合期望的底层映射  H(x) ，而是拟合残差映射  F(x) = H(x) - x 。原始映射则表示为  H(x) = F(x) + x 。

### 跳跃连接
跳跃连接（也称为 shortcut 连接）使梯度在反向传播过程中能够直接流过网络，防止梯度消失问题。它们将层的输入直接添加到其输出，允许网络学习相对于层输入的残差函数。

### 瓶颈结构
ResNet-50使用瓶颈残差块，由三个卷积层组成：1×1卷积（用于降维）、3×3卷积（用于特征提取）和另一个1×1卷积（用于恢复维度）。这种设计在保持表示能力的同时降低了计算复杂度。

### 退化问题解决方案
深度网络通常会遇到退化问题，即随着深度增加，精度达到饱和然后迅速下降。ResNet通过跳跃连接允许网络学习恒等映射，确保添加更多层不会导致性能比浅层网络更差。

## Python实现

### 依赖要求
- Python 3.7+
- PyTorch 1.8+
- torchvision

### 使用方法
```python
import torch
from ResNet-50 import resnet50

# 创建模型
model = resnet50()

# 前向传播
input = torch.randn(1, 3, 224, 224)
output = model(input)
print(output.shape)
```

### 训练
```bash
python ResNet-50.py
```

## 网络架构
ResNet-50架构包含：
- 1个初始卷积层（7×7，步长2）
- 1个最大池化层（3×3，步长2）
- 4个带有瓶颈结构的残差块：
  - 块1：3个瓶颈层，64个输出通道
  - 块2：4个瓶颈层，128个输出通道
  - 块3：6个瓶颈层，256个输出通道
  - 块4：3个瓶颈层，512个输出通道
- 1个自适应平均池化层
- 1个全连接层（ImageNet数据集1000个类别）

## 未来规划
- **C++实现**: 将引入PyTorch C++ API (libtorch)实现，以提高生产环境中的性能
- **模型优化**: 添加量化和剪枝支持，便于部署
- **更多ResNet变体**: 支持ResNet-18、ResNet-34、ResNet-101和ResNet-152架构
- **预训练模型**: 提供针对各种数据集的预训练权重

## 许可证
MIT许可证
