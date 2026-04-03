# 环境搭建指南

## 概述

本项目包含三个独立环境：训练环境 (GPU)、推理环境 (NPU)、服务环境 (gRPC)。它们可以部署在不同的机器上。

## 训练环境 (GPU 服务器)

### 系统要求

- Python 3.9+
- NVIDIA GPU (推荐 RTX 3090 或更高，显存 ≥ 8GB)
- CUDA 11.8+
- cuDNN 8+

### 安装步骤

```bash
# 创建虚拟环境
python -m venv .venv-training
source .venv-training/bin/activate  # Linux/macOS
# 或 .venv-training\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements/training.txt
```

### 验证安装

```python
import torch
print(f"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}")
# 应输出 CUDA True

from ultralytics import YOLO
print(f"Ultralytics {YOLO.__module__.split('.')[0]}")
```

## 推理环境 (昇腾 NPU 服务器)

### 系统要求

- Python 3.9+
- 昇腾 310B/310P 加速卡
- CANN 6.3+ (Ascend Computing Language)
- Atlas 200I DK A2 / Atlas 300I Pro 等昇腾硬件

### 安装步骤

```bash
# 创建虚拟环境
python -m venv .venv-inference
source .venv-inference/bin/activate

# 安装 CANN (需在昇腾服务器上从华为官网下载)
# 参考：https://www.hiascend.com/software/cann

# 安装推理依赖
pip install -r requirements/inference.txt
```

### 验证安装

```python
import acl
print("ACL 模块导入成功")

# 检查 NPU 设备
ret = acl.init()
print(f"ACL 初始化返回值: {ret}")  # 应为 0
```

## 服务环境 (gRPC 服务器)

### 系统要求

- Python 3.9+
- 与推理环境共享同一台昇腾服务器 (因为需要加载 OM 模型)

### 安装步骤

```bash
# 创建虚拟环境 (可与推理环境共用)
python -m venv .venv-service
source .venv-service/bin/activate

# 安装依赖
pip install -r requirements/service.txt
```

### 生成 Protobuf 代码

```bash
python -m grpc_tools.protoc \
    -I./service/proto \
    --python_out=./service/proto \
    --grpc_python_out=./service/proto \
    ./service/proto/detection.proto
```

### 验证安装

```bash
# 测试 gRPC 连接 (需先启动服务)
python service/grpc_client.py --image ./data/test.jpg --server localhost:50051
```

## 项目依赖说明

### requirements/training.txt
- `torch` — PyTorch 深度学习框架
- `ultralytics` — YOLO 实现 (需支持 OBB 版本)
- `opencv-python` — 图像处理
- `numpy` — 数值计算
- `pyyaml` — 配置文件解析
- `albumentations` — 数据增强

### requirements/inference.txt
- `numpy` — 数值计算
- `opencv-python` — 图像处理
- `pyyaml` — 配置文件解析
- `ascend` — 昇腾 ACL Python API (CANN 提供)

### requirements/service.txt
- `grpcio` — gRPC 框架
- `grpcio-tools` — Protobuf 代码生成工具
- `protobuf` — Protobuf 运行时
- `numpy` — 数值计算
- `opencv-python` — 图像处理
- `pyyaml` — 配置文件解析

## 常见问题

### 1. CUDA 版本不匹配
确保安装的 PyTorch 版本与系统 CUDA 版本匹配：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. CANN 安装失败
- 确认操作系统版本在 CANN 支持列表中
- 安装前需要安装驱动固件
- 参考华为官方文档：https://www.hiascend.com/document

### 3. Protobuf 生成失败
确保 `grpcio-tools` 已安装：
```bash
pip install grpcio-tools
python -m grpc_tools.protoc --version
```

### 4. 端口被占用
gRPC 服务默认使用 50051 端口，可通过命令行修改：
```bash
python service/grpc_server.py --port 50052
```
