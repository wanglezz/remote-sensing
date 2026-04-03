"""
gRPC 服务集成测试
"""

import os
import sys
import pytest
import numpy as np
import yaml
from unittest.mock import Mock, patch


class TestConfigLoading:
    """配置加载测试"""

    def test_load_service_config(self):
        """应能加载 service_config.yaml"""
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'service_config.yaml'
        )
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert 'server' in config
        assert 'grpc' in config
        assert 'inference' in config
        assert config['server']['port'] == 50051

    def test_load_training_config(self):
        """应能加载 training_config.yaml"""
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'training_config.yaml'
        )
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert 'model' in config
        assert config['model']['architecture'] == 'yolov8n-obb'
        assert config['data']['nc'] == 15

    def test_load_npu_config(self):
        """应能加载 npu_inference_config.yaml"""
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'npu_inference_config.yaml'
        )
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert 'model' in config
        assert 'npu' in config
        assert 'performance' in config
        assert config['performance']['target_latency_ms'] == 20


class TestDetectionServicer:
    """DetectionServicer 测试"""

    @pytest.fixture
    def mock_config(self):
        """Mock 配置"""
        return {
            'model': {'om_path': '/nonexistent/model.om'},
            'npu': {'device_id': 0},
            'inference': {
                'conf_thres': 0.25,
                'iou_thres': 0.7,
                'max_det': 300,
            }
        }

    @pytest.fixture
    def mock_request(self):
        """Mock DetectionRequest"""
        request = Mock()
        request.image_data = None
        request.image_path = ''
        request.confidence_threshold = 0.25
        request.iou_threshold = 0.7
        request.max_detections = 300
        return request

    def test_health_check(self, mock_config):
        """健康检查应返回 OK"""
        # 使用 mock engine 测试
        from inference.acl_runtime.acl_inference import MockACLInference

        # 构造 servicer 并替换 engine
        with patch('service.grpc_server.AclInference', MockACLInference):
            from service.grpc_server import DetectionServicer
            servicer = DetectionServicer(mock_config)

            # Mock request/response
            mock_health_request = Mock()
            mock_context = Mock()

            response = servicer.HealthCheck(mock_health_request, mock_context)
            assert response.healthy is True
            assert response.status == "OK"
            assert response.version == "1.0.0"

    def test_detect_no_image(self, mock_config, mock_request):
        """无图像时应返回错误"""
        with patch('service.grpc_server.AclInference', return_value=Mock()):
            from service.grpc_server import DetectionServicer
            servicer = DetectionServicer(mock_config)

            mock_context = Mock()
            response = servicer.Detect(mock_request, mock_context)
            assert "No image data" in response.error_message


class TestProtoDefinition:
    """Protobuf 定义验证"""

    def test_proto_file_exists(self):
        """proto 文件应存在"""
        proto_path = os.path.join(
            os.path.dirname(__file__), '..', 'service', 'proto', 'detection.proto'
        )
        assert os.path.exists(proto_path)

    def test_proto_content(self):
        """proto 文件应包含必要的服务定义"""
        proto_path = os.path.join(
            os.path.dirname(__file__), '..', 'service', 'proto', 'detection.proto'
        )
        with open(proto_path, 'r') as f:
            content = f.read()

        assert 'service DetectionService' in content
        assert 'rpc Detect' in content
        assert 'rpc HealthCheck' in content
        assert 'rpc BatchDetect' in content
        assert 'rpc StreamingDetect' in content
        assert 'DetectionRequest' in content
        assert 'DetectionResponse' in content


class TestIntegration:
    """端到端集成测试"""

    def test_config_consistency(self):
        """各配置文件中的参数应一致"""
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')

        with open(os.path.join(config_dir, 'service_config.yaml'), 'r') as f:
            service_config = yaml.safe_load(f)
        with open(os.path.join(config_dir, 'npu_inference_config.yaml'), 'r') as f:
            npu_config = yaml.safe_load(f)

        # 推理阈值应一致
        service_conf = service_config['inference']['conf_thres']
        npu_conf = npu_config['inference']['conf_thres']
        assert service_conf == npu_conf, \
            f"conf_thres 不一致: service={service_conf}, npu={npu_conf}"

        service_iou = service_config['inference']['iou_thres']
        npu_iou = npu_config['inference']['iou_thres']
        assert service_iou == npu_iou, \
            f"iou_thres 不一致: service={service_iou}, npu={npu_iou}"
