# 训练指南

## 概述

使用 YOLO-OBB-v8n 模型在 DOTA v1.0 数据集上训练旋转目标检测模型。

## 数据准备

### 1. 下载 DOTA 数据集

```bash
python data/scripts/download_dota.py
```

脚本会自动下载 DOTA v1.0 的训练集、验证集和测试集到 `data/datasets/DOTA/` 目录。

### 2. 预处理数据

```bash
python data/scripts/prepare_dota.py
```

预处理步骤：
- 将原始大图 (通常 10000+ 像素) 切分为 1024×1024 的子图
- 处理旋转边界框标签 (.obb.txt 格式)
- 划分训练/验证集

预处理后的目录结构：
```
data/datasets/DOTA/
├── train/
│   ├── images/  # .png 子图
│   └── labels/  # .txt 标签文件
└── val/
    ├── images/
    └── labels/
```

### 标签格式

每行一个目标，格式为：
```
class_id cx cy w h angle
```
- `class_id`: 0-14 (15 类 DOTA 目标)
- `cx, cy`: 边界框中心点 (归一化到 0-1)
- `w, h`: 边界框宽高 (归一化到 0-1)
- `angle`: 旋转角度 (弧度制)

## 训练模型

### 方式一：直接运行训练脚本

```bash
python models/training/train.py
```

### 方式二：使用快捷脚本

```bash
bash scripts/run_training.sh
```

### 方式三：自定义参数

```bash
python models/training/train.py \
    --epochs 200 \
    --batch-size 8 \
    --imgsz 1024 \
    --lr 0.01 \
    --device 0
```

## 配置说明

训练参数在 `configs/training_config.yaml` 中配置：

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `imgsz` | 1024 | 输入图像尺寸，DOTA 推荐 1024 |
| `batch_size` | 16 | 批次大小，根据显存调整 |
| `epochs` | 100 | 训练轮数 |
| `optimizer` | SGD | 优化器 |
| `lr0` | 0.01 | 初始学习率 |
| `lrf` | 0.1 | 最终学习率系数 (cosine decay) |
| `momentum` | 0.937 | SGD 动量 |
| `weight_decay` | 0.0005 | 权重衰减 |
| `warmup_epochs` | 3 | 预热轮数 |

### 数据增强配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hsv_h` | 0.015 | 色调抖动 |
| `hsv_s` | 0.7 | 饱和度增强 |
| `hsv_v` | 0.4 | 亮度增强 |
| `rotation` | 90 | 随机旋转角度 |
| `scale` | 0.5 | 缩放比例 |
| `mosaic` | 1.0 | Mosaic 增强概率 |

## 监控训练

### 日志输出

训练日志输出到 `models/training/runs/DOTA_OBB/train/` 目录：

- `results.csv` — 每轮的指标数据
- `args.yaml` — 训练参数
- `weights/best.pt` — 最佳权重
- `weights/last.pt` — 最新权重

### 指标解释

- **mAP@50**: IoU=0.5 时的平均精度均值，目标 ≥ 0.80
- **Precision**: 精确率，目标 ≥ 0.90
- **Recall**: 召回率
- **Box Loss**: 边界框回归损失
- **Cls Loss**: 分类损失

## 常见问题

### 显存不足 (OOM)

减小 batch_size 或启用梯度累积：
```bash
python models/training/train.py --batch-size 8
```

### 训练不收敛

1. 降低初始学习率 (`lr0: 0.005`)
2. 增加预热轮数 (`warmup_epochs: 5`)
3. 检查数据标签是否正确

### 恢复训练

```bash
python models/training/train.py --resume
```

## 导出模型

训练完成后，导出为 ONNX 格式：

```bash
python models/export/export_onnx.py \
    --weights models/training/runs/DOTA_OBB/train/weights/best.pt \
    --output models/exported/best.onnx
```

或使用快捷脚本导出 ONNX → OM：

```bash
bash scripts/export_model.sh
```
