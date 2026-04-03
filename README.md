## 项目使用说明

### 1. 项目概述

本项目是一个基于YOLO-OBB的目标检测模型的训练和部署工具链。它支持从数据准备、模型训练、模型转换到在昇腾NPU上进行推理的全流程。

### 2. 环境要求

- **硬件**:
  - 昇腾NPU (310B/P)
  - 支持的硬件平台：Atlas 200I DK A2 或 Atlas 300I Pro
- **软件**:
  - Python 3.7+
  - PyTorch 1.8+
  - ONNX 1.9.0+
  - 升腾CANN（Compute Architecture for Neural Networks）
  - 其他依赖项请参考`requirements.txt`

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 数据准备
下载DOTA数据集并解压到`data/dota`目录下：
```bash
python data/scripts/download_dota.py
```

### 5. 模型训练
使用以下命令启动训练：
```bash
python models/train.py --data data/dota --weights yolov8n_obb.pt --epochs 100
```

### 6. 模型导出
将训练好的PyTorch模型导出为ONNX格式：
```bash
python models/export/export_onnx.py \
    --weights models/training/runs/DOTA_OBB/train/weights/best.pt \
    --output models/exported/best.onnx \
    --imgsz 1024 \
    --dynamic \
    --simplify
```

### 7. 模型转换
使用昇腾ATC工具将ONNX模型转换为OM格式：
```bash
atc \
    --model=models/exported/best.onnx \
    --framework=5 \
    --output=models/om_models/yolov8n_obb \
    --input_format=NCHW \
    --input_shape="images:1,3,1024,1024" \
    --soc_version=Ascend310B \
    --op_select_implmode=high_performance
```

### 8. 推理
使用ACL推理引擎进行推理：
```python
from inference.acl_runtime.acl_inference import AclInference

engine = AclInference(
    om_path='./models/om_models/yolov8n_obb.om',
    device_id=0,
    conf_thres=0.25,
    iou_thres=0.7,
    max_det=300
)

import cv2

image = cv2.imread('test.jpg')
result = engine(image)

print(f"检测到 {len(result.boxes)} 个目标")
for i in range(len(result.boxes)):
    box = result.boxes[i]
    score = result.scores[i]
    class_id = result.class_ids[i]
    print(f"  类别 {class_id}, 置信度 {score:.2f}, 框 {box}")
```

### 移植到昇腾平台

#### 1. 环境配置
确保您的环境已经安装了昇腾CANN及相关驱动。您可以通过以下命令检查CANN版本：
```bash
npusiminfo
```

#### 2. 模型导出与转换
按照上述步骤将训练好的PyTorch模型导出为ONNX格式，然后使用ATC工具将其转换为OM格式。

#### 3. 配置AIPP
如果需要使用AIPP进行预处理加速，请在`configs/npu_inference_config.yaml`中进行配置，并在ATC转换时指定输入格式。

#### 4. 性能优化
- **算子融合**：在ATC转换时使用`high_performance`模式。
- **Batch推理**：调整batch size以提高吞吐量。
- **多设备并行**：使用多个NPU设备进行并行推理。
- **量化技术**：通过训练后量化或量化感知训练来提高模型推理速度和降低内存占用。

#### 5. 性能基准测试
使用以下命令进行性能基准测试：
```bash
python inference/benchmark/benchmark_latency.py \
    --config configs/npu_inference_config.yaml \
    --warmup 10 \
    --iters 100 \
    --imgsz 1024
```

#### 6. 常见问题及解决方法
- **ATC转换失败**：确认ONNX模型格式正确，CANN版本与昇腾驱动匹配，检查`--soc_version`是否与实际硬件一致。
- **ACL初始化失败**：确保CANN环境变量已正确配置，驱动版本匹配，权限足够。
- **推理结果异常**：检查AIPP配置是否与训练时的预处理一致，确认输入图像格式（RGB vs BGR），验证NMS参数（`conf_thres`, `iou_thres`）。

### 项目结构
```
remote-sensing/
├── data/
│   ├── dota/
│   └── scripts/
│       └── download_dota.py
├── models/
│   ├── training/
│   │   └── runs/
│   │       └── DOTA_OBB/
│   │           └── train/
│   │               └── weights/
│   ├── export/
│   │   └── export_onnx.py
│   └── om_models/
├── inference/
│   ├── acl_runtime/
│   │   └── acl_inference.py
│   └── benchmark/
│       └── benchmark_latency.py
├── configs/
│   └── npu_inference_config.yaml
├── README.md
└── requirements.txt
```

### 联系方式
如果您有任何问题或需要进一步的帮助，请联系 [您的联系方式]。
