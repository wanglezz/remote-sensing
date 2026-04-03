"""
gRPC 检测客户端示例

演示如何调用检测服务
"""

import argparse
import base64
import cv2
import numpy as np
from typing import Optional

import grpc

try:
    from service.proto import detection_pb2
    from service.proto import detection_pb2_grpc
except ImportError:
    print("错误：请先生成 protobuf 模块")
    print("运行：python -m grpc_tools.protoc -I./service/proto --python_out=./service/proto --grpc_python_out=./service/proto ./service/proto/detection.proto")
    import sys
    sys.exit(1)


def create_stub(server_address: str):
    """创建 gRPC 存根"""
    channel = grpc.insecure_channel(server_address)
    return detection_pb2_grpc.DetectionServiceStub(channel)


def detect_image(
    stub: detection_pb2_grpc.DetectionServiceStub,
    image_path: Optional[str] = None,
    image_data: Optional[bytes] = None,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7
) -> detection_pb2.DetectionResponse:
    """
    检测图像

    Args:
        stub: gRPC 存根
        image_path: 图像路径
        image_data: 图像二进制数据
        conf_thres: 置信度阈值
        iou_thres: NMS IoU 阈值

    Returns:
        检测响应
    """
    # 准备请求
    request = detection_pb2.DetectionRequest(
        confidence_threshold=conf_thres,
        iou_threshold=iou_thres,
        max_detections=300
    )

    if image_data:
        request.image_data = base64.b64encode(image_data)
    elif image_path:
        # 读取图像并编码
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        request.image_data = base64.b64encode(image_bytes)
    else:
        raise ValueError("需要指定 image_path 或 image_data")

    # 调用 RPC
    response = stub.Detect(request)

    return response


def draw_detections(
    image: np.ndarray,
    response: detection_pb2.DetectionResponse,
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制检测结果

    Args:
        image: 原始图像
        response: 检测响应
        thickness: 线条粗细

    Returns:
        绘制后的图像
    """
    from utils.obb_utils import rbox2poly

    img = image.copy()

    for det in response.detections:
        # 获取多边形
        box = np.array(det.box)
        poly = rbox2poly(box).astype(np.int32)
        poly_pts = poly.reshape(4, 2)

        # 随机颜色
        color = (
            int(hash(det.class_name) % 256),
            int((hash(det.class_name) * 7) % 256),
            int((hash(det.class_name) * 13) % 256)
        )

        # 绘制边框
        cv2.polylines(img, [poly_pts], True, color, thickness)

        # 绘制标签
        label = f"{det.class_name}: {det.confidence:.2f}"
        cx, cy = int(box[0]), int(box[1])
        cv2.putText(img, label, (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img


def main():
    parser = argparse.ArgumentParser(description='gRPC 检测客户端')
    parser.add_argument(
        '--server',
        type=str,
        default='localhost:50051',
        help='gRPC 服务器地址'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='输入图像路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出图像路径 (默认：不保存)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='置信度阈值'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='NMS IoU 阈值'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )

    args = parser.parse_args()

    # 创建连接
    print(f"连接到服务器：{args.server}")
    stub = create_stub(args.server)

    # 健康检查
    try:
        health_response = stub.HealthCheck(detection_pb2.HealthCheckRequest())
        print(f"服务器状态：{health_response.status} (v{health_response.version})")
    except grpc.RpcError as e:
        print(f"健康检查失败：{e}")
        return

    # 检测图像
    print(f"\n检测图像：{args.image}")

    try:
        response = detect_image(
            stub,
            image_path=args.image,
            conf_thres=args.conf,
            iou_thres=args.iou
        )

        # 打印结果
        print(f"\n处理时延：{response.latency_ms:.2f} ms")
        print(f"检测到 {len(response.detections)} 个目标")

        if args.verbose:
            for det in response.detections:
                print(f"  - {det.class_name}: {det.confidence:.2f} "
                      f"box=[{det.box[0]:.1f}, {det.box[1]:.1f}, "
                      f"{det.box[2]:.1f}x{det.box[3]:.1f}, "
                      f"{det.box[4]:.2f}rad]")

        # 显示/保存结果
        if args.output or args.verbose:
            image = cv2.imread(args.image)
            result = draw_detections(image, response)

            if args.output:
                cv2.imwrite(args.output, result)
                print(f"\n结果已保存到：{args.output}")

    except grpc.RpcError as e:
        print(f"检测失败：{e}")


if __name__ == '__main__':
    main()
