#!/bin/bash
# gRPC 服务启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认参数
CONFIG="${CONFIG:-./configs/service_config.yaml}"
HOST="${SERVICE_HOST:-0.0.0.0}"
PORT="${SERVICE_PORT:-50051}"

echo "=============================================="
echo "gRPC 检测服务"
echo "=============================================="
echo "配置文件：$CONFIG"
echo "监听地址：$HOST:$PORT"
echo ""

# 检查配置文件
if [ ! -f "$CONFIG" ]; then
    echo "错误：配置文件不存在：$CONFIG"
    exit 1
fi

# 检查模型文件
OM_PATH=$(grep 'om_path' "$CONFIG" | awk -F': ' '{print $2}' | tr -d ' ')
if [ ! -f "$OM_PATH" ]; then
    echo "警告：模型文件不存在：$OM_PATH"
    echo "服务将以 Mock 模式运行"
fi

# 启动服务
echo "启动 gRPC 服务..."
python "$PROJECT_ROOT/service/grpc_server.py" \
    --config "$CONFIG" \
    --host "$HOST" \
    --port "$PORT"
