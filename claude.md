# Remote Sensing OBB Detection Project

## Project Overview
遥感卫星图像旋转目标检测系统，基于 YOLO-OBB-v8n，完成从 GPU 训练到昇腾 NPU 部署的全流程。

## Current State (as of 2026-04-03)

### Completed Files
| File | Status |
|------|--------|
| `configs/training_config.yaml` | Done |
| `configs/npu_inference_config.yaml` | Done |
| `configs/service_config.yaml` | Done |
| `data/scripts/download_dota.py` | Done |
| `data/scripts/prepare_dota.py` | Done |
| `models/training/train.py` | Done |
| `models/export/export_onnx.py` | Done |
| `inference/acl_runtime/acl_inference.py` | Done (cv2 import fixed) |
| `service/proto/detection.proto` | Done |
| `utils/obb_utils.py` | Done |
| `utils/metrics.py` | Done |
| `utils/callbacks.py` | Done |
| `service/grpc_server.py` | Done |
| `service/grpc_client.py` | Done |
| `inference/benchmark/benchmark_latency.py` | Done |
| `scripts/export_model.sh` | Done |
| `scripts/run_training.sh` | Done |
| `scripts/start_service.sh` | Done |
| `README.md` | Done |
| `requirements/` (training, inference, service) | Done |
| `tests/test_train.py` | Done |

### Remaining Work
1. **docs/** — 4 documents: setup_guide.md, training_guide.md, npu_deployment.md, api_reference.md
2. **tests/** — integration/unit tests for gRPC, inference, and utils

### Known Issues in Existing Code
- `grpc_server.py` L101: uses `np` without importing `numpy` (missing `import numpy as np`)
- `grpc_server.py` L126: `self.CLASS_NAMES.get()` — CLASS_NAMES is a list, not a dict; should use `self.CLASS_NAMES[int(result.class_ids[i])]
- `grpc_client.py` L90: imports `rbox2poly` from `utils.obb_utils` — verify the function signature matches what's expected (protobuf `box` field is 5 floats: cx, cy, w, h, angle)
- `acl_inference.py`: uses Mock engine when OM model not found — this is expected for non-NPU environments
- gRPC protobuf Python stubs need to be generated before running server/client:
  ```bash
  python -m grpc_tools.protoc -I./service/proto --python_out=./service/proto --grpc_python_out=./service/proto ./service/proto/detection.proto
  ```

## Project Structure
```
remote-sensing/
├── configs/              # YAML configs (training, inference, service)
├── data/scripts/         # Dataset download & preparation
├── models/training/      # Training script (train.py)
├── models/export/        # Model export (export_onnx.py)
├── inference/
│   ├── acl_runtime/      # ACL inference engine (acl_inference.py)
│   └── benchmark/        # Latency benchmark (benchmark_latency.py)
├── service/
│   ├── proto/            # Protobuf definitions (detection.proto)
│   ├── grpc_server.py    # gRPC server implementation
│   └── grpc_client.py    # gRPC client implementation
├── utils/                # Shared utilities (obb_utils, metrics, callbacks)
├── requirements/         # Dependency files
├── scripts/              # Shell shortcut scripts
├── docs/                 # (TODO: 4 documents)
└── tests/                # (TODO: test scripts)
```

## Key Technical Details
- **Model**: YOLO-OBB-v8n (Oriented Bounding Box)
- **Classes**: 15 DOTA v1.0 categories (plane, baseball-diamond, bridge, etc.)
- **Target Platform**: 昇腾 Ascend 310B/P NPU
- **Latency Target**: P95 ≤ 20ms
- **mAP Target**: ≥ 0.80 @ IoU=0.5
- **Service**: gRPC on port 50051, max message 50MB
