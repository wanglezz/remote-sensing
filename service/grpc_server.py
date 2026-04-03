"""
gRPC 检测服务

提供基于 gRPC 的目标检测服务接口
"""

import os
import sys
import cv2
import yaml
import time
import logging
from pathlib import Path
from concurrent import futures
from typing import Optional, Dict, Any

import grpc
import numpy as np
from grpc import StatusCode

# 导入生成的 protobuf 模块 (需要先生成)
try:
    from service.proto import detection_pb2
    from service.proto import detection_pb2_grpc
except ImportError:
    print("错误：请先生成 protobuf 模块")
    print("运行：python -m grpc_tools.protoc -I./service/proto --python_out=./service/proto --grpc_python_out=./service/proto ./service/proto/detection.proto")
    sys.exit(1)

# 导入推理引擎
from inference.acl_runtime.acl_inference import AclInference


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class DetectionServicer(detection_pb2_grpc.DetectionServiceServicer):
    """检测服务实现"""

    # 类别名称 (DOTA v1.0)
    CLASS_NAMES = [
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank', 'soccer-ball-field',
        'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        初始化服务

        Args:
            config: 服务配置
        """
        self.config = config

        # 初始化推理引擎
        inference_config = config.get('inference', {})
        om_path = config.get('model', {}).get('om_path', '')

        if os.path.exists(om_path):
            self.inference_engine = AclInference(
                om_path=om_path,
                device_id=config.get('npu', {}).get('device_id', 0),
                conf_thres=inference_config.get('conf_thres', 0.25),
                iou_thres=inference_config.get('iou_thres', 0.7),
                max_det=inference_config.get('max_det', 300)
            )
            logger.info(f"推理引擎已初始化：{om_path}")
        else:
            logger.warning(f"模型文件不存在：{om_path}，使用 Mock 引擎")
            self.inference_engine = AclInference()

    def Detect(
        self,
        request: detection_pb2.DetectionRequest,
        context: grpc.ServicerContext
    ) -> detection_pb2.DetectionResponse:
        """
        单次检测 RPC

        Args:
            request: 检测请求
            context: gRPC 上下文

        Returns:
            检测响应
        """
        start_time = time.perf_counter()

        try:
            # 获取图像
            if request.image_data:
                # base64 解码
                import base64
                image_bytes = base64.b64decode(request.image_data)
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            elif request.image_path:
                # 从文件读取
                image = cv2.imread(request.image_path)
            else:
                return detection_pb2.DetectionResponse(
                    error_message="No image data or path provided"
                )

            if image is None:
                return detection_pb2.DetectionResponse(
                    error_message="Failed to decode image"
                )

            # 推理
            result = self.inference_engine(image)

            # 构建响应
            detections = []
            for i in range(len(result.boxes)):
                det = detection_pb2.Detection(
                    box=result.boxes[i].tolist(),
                    confidence=float(result.scores[i]),
                    class_id=int(result.class_ids[i]),
                    class_name=self.CLASS_NAMES[int(result.class_ids[i])] if int(result.class_ids[i]) < len(self.CLASS_NAMES) else 'unknown'
                )
                detections.append(det)

            latency_ms = (time.perf_counter() - start_time) * 1000

            return detection_pb2.DetectionResponse(
                detections=detections,
                latency_ms=latency_ms,
                error_message="",
                image_width=image.shape[1],
                image_height=image.shape[0]
            )

        except Exception as e:
            logger.exception(f"检测失败：{e}")
            context.set_code(StatusCode.INTERNAL)
            context.set_details(str(e))
            return detection_pb2.DetectionResponse(
                error_message=str(e)
            )

    def HealthCheck(
        self,
        request: detection_pb2.HealthCheckRequest,
        context: grpc.ServicerContext
    ) -> detection_pb2.HealthCheckResponse:
        """健康检查"""
        return detection_pb2.HealthCheckResponse(
            healthy=True,
            version="1.0.0",
            status="OK"
        )

    def BatchDetect(
        self,
        request: detection_pb2.BatchDetectionRequest,
        context: grpc.ServicerContext
    ) -> detection_pb2.BatchDetectionResponse:
        """批量检测"""
        start_time = time.perf_counter()

        responses = []
        for det_request in request.requests:
            response = self.Detect(det_request, context)
            responses.append(response)

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        return detection_pb2.BatchDetectionResponse(
            responses=responses,
            total_latency_ms=total_latency_ms
        )

    def StreamingDetect(
        self,
        request_iterator,
        context: grpc.ServicerContext
    ):
        """流式检测"""
        for request in request_iterator:
            response = self.Detect(request, context)
            yield response


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def serve(config_path: str = './configs/service_config.yaml'):
    """启动 gRPC 服务"""
    # 加载配置
    config = load_config(config_path)
    server_config = config.get('server', {})
    grpc_config = config.get('grpc', {})

    # 创建服务器
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=server_config.get('max_workers', 10)),
        options=[
            ('grpc.max_send_message_length', grpc_config.get('max_message_length', 52428800)),
            ('grpc.max_receive_message_length', grpc_config.get('max_message_length', 52428800)),
        ]
    )

    # 注册服务
    servicer = DetectionServicer(config)
    detection_pb2_grpc.add_DetectionServiceServicer_to_server(servicer, server)

    # 绑定地址
    host = server_config.get('host', '0.0.0.0')
    port = server_config.get('port', 50051)
    server.add_insecure_port(f'{host}:{port}')

    # 启动服务器
    server.start()
    logger.info(f"gRPC 服务已启动：{host}:{port}")

    # 等待终止
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务...")
        server.stop(grace=server_config.get('graceful_shutdown', 5))
        servicer.inference_engine.release()
        logger.info("服务已关闭")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='gRPC 检测服务')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/service_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='监听地址'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='监听端口'
    )

    args = parser.parse_args()

    # 加载配置并覆盖
    config = load_config(args.config)
    if args.host:
        config['server']['host'] = args.host
    if args.port:
        config['server']['port'] = args.port

    serve(config)
