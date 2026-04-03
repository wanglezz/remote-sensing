#!/bin/bash
# 训练启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认参数
CONFIG="${CONFIG:-./configs/training_config.yaml}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-1024}"
DEVICE="${DEVICE:-0}"

echo "=============================================="
echo "YOLO-OBB 训练脚本"
echo "=============================================="
echo "配置文件：$CONFIG"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH"
echo "图像尺寸：$IMGSZ"
echo "设备：$DEVICE"
echo ""

# 检查配置文件
if [ ! -f "$CONFIG" ]; then
    echo "错误：配置文件不存在：$CONFIG"
    exit 1
fi

# 检查数据集
DATASET_ROOT="${DOTA_DATASET_ROOT:-./data/datasets/DOTA}"
if [ ! -d "$DATASET_ROOT/yolo_obb" ]; then
    echo "警告：数据集未准备，请先运行预处理脚本"
    echo "运行：python data/scripts/prepare_dota.py"
    exit 1
fi

# 启动训练
echo "开始训练..."
python "$PROJECT_ROOT/models/training/train.py" \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --imgsz "$IMGSZ" \
    --device "$DEVICE"

echo ""
echo "=============================================="
echo "训练完成！"
echo "=============================================="
