# API 参考

## 概述

gRPC 检测服务提供以下 RPC 接口。服务默认监听 `0.0.0.0:50051`。

## Protobuf 定义

定义文件：`service/proto/detection.proto`

## 服务接口

### DetectionService

#### Detect (单次检测)

```protobuf
rpc Detect(DetectionRequest) returns (DetectionResponse);
```

单次图像检测。

**请求 (DetectionRequest):**

| 字段 | 类型 | 说明 |
|------|------|------|
| `image_data` | bytes | base64 编码的 JPEG/PNG 图像数据 |
| `image_path` | string | 图像文件路径 (服务端需能访问) |
| `confidence_threshold` | float | 置信度阈值 (默认 0.25) |
| `iou_threshold` | float | NMS IoU 阈值 (默认 0.7) |
| `max_detections` | int32 | 最大检测框数 (默认 300) |

`image_data` 和 `image_path` 二选一。

**响应 (DetectionResponse):**

| 字段 | 类型 | 说明 |
|------|------|------|
| `detections` | repeated Detection | 检测结果列表 |
| `latency_ms` | float | 处理时延 (毫秒) |
| `error_message` | string | 错误信息，空表示成功 |
| `image_width` | int32 | 图像宽度 |
| `image_height` | int32 | 图像高度 |

**Detection (单个检测结果):**

| 字段 | 类型 | 说明 |
|------|------|------|
| `box` | repeated float | 旋转框 `[cx, cy, w, h, angle]` |
| `confidence` | float | 置信度 |
| `class_id` | int32 | 类别 ID (0-14) |
| `class_name` | string | 类别名称 |

**示例 (Python 客户端):**

```python
import grpc
from service.proto import detection_pb2, detection_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = detection_pb2_grpc.DetectionServiceStub(channel)

with open('test.jpg', 'rb') as f:
    image_data = f.read()

import base64
request = detection_pb2.DetectionRequest(
    image_data=base64.b64encode(image_data),
    confidence_threshold=0.25,
    iou_threshold=0.7,
    max_detections=300
)

response = stub.Detect(request)
for det in response.detections:
    print(f"{det.class_name}: {det.confidence:.2f}")
```

#### BatchDetect (批量检测)

```protobuf
rpc BatchDetect(BatchDetectionRequest) returns (BatchDetectionResponse);
```

批量检测多张图像。

**请求 (BatchDetectionRequest):**

| 字段 | 类型 | 说明 |
|------|------|------|
| `requests` | repeated DetectionRequest | 多个检测请求 |

**响应 (BatchDetectionResponse):**

| 字段 | 类型 | 说明 |
|------|------|------|
| `responses` | repeated DetectionResponse | 多个检测响应 |
| `total_latency_ms` | float | 总处理时延 (毫秒) |

#### HealthCheck (健康检查)

```protobuf
rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
```

检查服务健康状态。

**响应 (HealthCheckResponse):**

| 字段 | 类型 | 说明 |
|------|------|------|
| `healthy` | bool | 服务是否健康 |
| `version` | string | 服务版本 |
| `status` | string | 状态描述 |

#### StreamingDetect (流式检测)

```protobuf
rpc StreamingDetect(stream DetectionRequest) returns (stream DetectionResponse);
```

流式图像检测，适用于视频流或连续数据流。

## 类别定义

DOTA v1.0 数据集的 15 个类别：

| ID | 类别名称 | 说明 |
|----|----------|------|
| 0 | plane | 飞机 |
| 1 | baseball-diamond | 棒球场 |
| 2 | bridge | 桥梁 |
| 3 | ground-track-field | 田径场 |
| 4 | small-vehicle | 小型车辆 |
| 5 | large-vehicle | 大型车辆 |
| 6 | ship | 船舶 |
| 7 | tennis-court | 网球场 |
| 8 | basketball-court | 篮球场 |
| 9 | storage-tank | 储罐 |
| 10 | soccer-ball-field | 足球场 |
| 11 | roundabout | 环岛 |
| 12 | harbor | 港口/码头 |
| 13 | swimming-pool | 游泳池 |
| 14 | helicopter | 直升机 |

## 边界框格式

旋转框使用 5 参数表示：`[cx, cy, w, h, angle]`

- `cx, cy`: 边界框中心点坐标
- `w, h`: 边界框宽度和高度
- `angle`: 旋转角度 (弧度制, 范围 [-π/4, 3π/4))

转换为 4 角点坐标可以使用 `utils.obb_utils.rbox2poly()` 函数。

## 错误处理

### gRPC 状态码

| 状态码 | 场景 |
|--------|------|
| `OK` | 成功 |
| `INTERNAL` | 推理内部错误 |
| `INVALID_ARGUMENT` | 请求参数无效 |
| `UNAVAILABLE` | 服务不可用 |

### 应用层错误

通过 `DetectionResponse.error_message` 字段返回：

- `"No image data or path provided"` — 缺少图像
- `"Failed to decode image"` — 图像解码失败
- 其他内部错误描述

## 性能指标

| 指标 | 目标值 |
|------|--------|
| P95 端到端时延 | ≤ 20ms |
| 最大消息大小 | 50MB |
| 最大并发流 | 100 |
| 工作线程数 | 10 |
