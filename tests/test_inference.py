"""
推理引擎单元测试 (使用 Mock 模式)
"""

import numpy as np
import pytest
from inference.acl_runtime.acl_inference import MockACLInference, InferenceResult


class TestMockInference:
    """Mock 推理引擎测试"""

    def test_initialization(self):
        """Mock 引擎应能初始化"""
        engine = MockACLInference(conf_thres=0.25, iou_thres=0.7)
        assert engine.conf_thres == 0.25

    def test_call_returns_result(self):
        """调用应返回 InferenceResult"""
        engine = MockACLInference()
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        result = engine(image)

        assert isinstance(result, InferenceResult)
        assert hasattr(result, 'boxes')
        assert hasattr(result, 'scores')
        assert hasattr(result, 'class_ids')

    def test_result_shapes(self):
        """结果应有正确的形状"""
        engine = MockACLInference()
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = engine(image)

        n = len(result.scores)
        assert result.boxes.shape == (n, 5)
        assert result.scores.shape == (n,)
        assert result.class_ids.shape == (n,)

    def test_scores_range(self):
        """置信度应在 [0, 1] 范围内"""
        engine = MockACLInference()
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = engine(image)

        if len(result.scores) > 0:
            assert np.all(result.scores >= 0)
            assert np.all(result.scores <= 1)

    def test_class_ids_range(self):
        """类别 ID 应在 [0, 14] 范围内"""
        engine = MockACLInference()
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = engine(image)

        if len(result.class_ids) > 0:
            assert np.all(result.class_ids >= 0)
            assert np.all(result.class_ids < 15)

    def test_release(self):
        """release 方法应无异常"""
        engine = MockACLInference()
        engine.release()  # 不应抛出异常

    def test_different_image_sizes(self):
        """不同尺寸的图像都应能处理"""
        engine = MockACLInference()
        for size in [(320, 320), (640, 640), (1024, 1024), (1920, 1080)]:
            image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            result = engine(image)
            assert isinstance(result, InferenceResult)
