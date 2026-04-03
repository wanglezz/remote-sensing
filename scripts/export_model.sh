#!/bin/bash
# 模型导出脚本
# PyTorch → ONNX → OM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认参数
MODEL_PATH="${MODEL_PATH:-./models/training/runs/DO TA_OBB/train/weights/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./models/om_models}"
IMGSZ="${IMGSZ:-1024}"

# ATC 工具参数 (昇腾环境)
ATC_PATH="${ATC_PATH:-/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux-gnu/ascend-toolkit/convert/bin/atc}"

echo "=============================================="
echo "模型导出脚本"
echo "=============================================="
echo "模型路径：$MODEL_PATH"
echo "输出目录：$OUTPUT_DIR"
echo "图像尺寸：$IMGSZ"
echo ""

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误：模型文件不存在：$MODEL_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# Step 1: PyTorch → ONNX
echo "Step 1: 导出 ONNX 模型..."
python "$PROJECT_ROOT/models/export/export_onnx.py" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_DIR/yolov8n_obb.onnx" \
    --imgsz "$IMGSZ" \
    --opset 11 \
    --verify

# Step 2: ONNX → OM (需要昇腾环境)
echo ""
echo "Step 2: 转换 OM 模型..."

if command -v $ATC_PATH &> /dev/null; then
    echo "使用 ATC 工具：$ATC_PATH"

    $ATC_PATH \
        --model="$OUTPUT_DIR/yolov8n_obb.onnx" \
        --framework=5 \
        --output="$OUTPUT_DIR/yolov8n_obb" \
        --input_format=NCHW \
        --input='images:1,3,'$IMGSZ','$IMGSZ \
        --output_type=FP32 \
        --soc_version=Ascend310P3 \
        --insert_op_conf="$OUTPUT_DIR/aipp.config"

    echo "✓ OM 模型已生成：$OUTPUT_DIR/yolov8n_obb.om"
else
    echo "⚠ ATC 工具未找到，跳过 OM 转换"
    echo "请在昇腾 NPU 环境中运行此脚本，或手动执行:"
    echo "  atc --model=$OUTPUT_DIR/yolov8n_obb.onnx --framework=5 --output=$OUTPUT_DIR/yolov8n_obb --input_format=NCHW --input='images:1,3,$IMGSZ,$IMGSZ'"
fi

echo ""
echo "=============================================="
echo "导出完成！"
echo "=============================================="
