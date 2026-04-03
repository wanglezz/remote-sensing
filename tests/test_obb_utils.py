"""
OBB 工具函数单元测试
"""

import numpy as np
import pytest
from utils.obb_utils import (
    rbox2poly,
    poly2rbox,
    obb_iou,
    obb_iou_batch,
    rotate_nms,
    xywhr2xyxyxyxy,
    OBB,
)


class TestRbox2Poly:
    """rbox2poly 测试"""

    def test_identity_box(self):
        """无旋转的框应返回矩形角点"""
        rbox = np.array([100, 100, 20, 10, 0])
        poly = rbox2poly(rbox)
        assert poly.shape == (8,)
        # 角点应为 (90,95), (110,95), (110,105), (90,105)
        assert np.allclose(poly[0], 90)  # x1
        assert np.allclose(poly[1], 95)  # y1

    def test_rotated_box(self):
        """45 度旋转"""
        rbox = np.array([0, 0, np.sqrt(2), np.sqrt(2), np.pi / 4])
        poly = rbox2poly(rbox)
        assert poly.shape == (8,)

    def test_center_preserved(self):
        """旋转后中心点不变"""
        cx, cy = 50.0, 75.0
        rbox = np.array([cx, cy, 20, 10, np.pi / 6])
        poly = rbox2poly(rbox)
        poly_pts = poly.reshape(4, 2)
        center = poly_pts.mean(axis=0)
        assert np.allclose(center, [cx, cy])


class TestPoly2Rbox:
    """poly2rbox 测试"""

    def test_rect_to_rbox(self):
        """矩形多边形转 rbox"""
        poly = np.array([90, 95, 110, 95, 110, 105, 90, 105])
        cx, cy, w, h, angle = poly2rbox(poly)
        assert np.isclose(cx, 100)
        assert np.isclose(cy, 100)


class TestOBBIoU:
    """obb_iou 测试"""

    def test_identical_boxes(self):
        """相同框的 IoU 应为 1"""
        box = np.array([100, 100, 20, 10, 0])
        iou = obb_iou(box, box)
        assert np.isclose(iou, 1.0, atol=1e-5)

    def test_no_overlap(self):
        """不重叠的框 IoU 应为 0"""
        box1 = np.array([0, 0, 10, 10, 0])
        box2 = np.array([100, 100, 10, 10, 0])
        iou = obb_iou(box1, box2)
        assert iou == 0.0

    def test_partial_overlap(self):
        """部分重叠的框 IoU 应在 (0, 1) 之间"""
        box1 = np.array([0, 0, 20, 20, 0])
        box2 = np.array([10, 0, 20, 20, 0])
        iou = obb_iou(box1, box2)
        assert 0.0 < iou < 1.0


class TestOBBIoUBatch:
    """obb_iou_batch 测试"""

    def test_shape(self):
        """输出矩阵形状应正确"""
        boxes1 = np.array([[0, 0, 10, 10, 0], [20, 20, 10, 10, 0]])
        boxes2 = np.array([[5, 5, 10, 10, 0]])
        iou_matrix = obb_iou_batch(boxes1, boxes2)
        assert iou_matrix.shape == (2, 1)


class TestRotateNMS:
    """rotate_nms 测试"""

    def test_no_suppression(self):
        """不重叠的框应全部保留"""
        boxes = np.array([
            [0, 0, 10, 10, 0],
            [100, 100, 10, 10, 0],
        ])
        scores = np.array([0.9, 0.8])
        keep = rotate_nms(boxes, scores, iou_threshold=0.7)
        assert len(keep) == 2

    def test_suppression(self):
        """重叠的框应抑制低分的"""
        boxes = np.array([
            [0, 0, 10, 10, 0],
            [0, 0, 10, 10, 0],
        ])
        scores = np.array([0.9, 0.5])
        keep = rotate_nms(boxes, scores, iou_threshold=0.7)
        assert len(keep) == 1
        assert keep[0] == 0  # 保留高分的

    def test_empty_input(self):
        """空输入应返回空数组"""
        boxes = np.array([]).reshape(0, 5)
        scores = np.array([])
        keep = rotate_nms(boxes, scores)
        assert len(keep) == 0


class TestXywhr2Xyxyxyxy:
    """xywhr2xyxyxyxy 测试"""

    def test_single_box(self):
        """单个框"""
        boxes = np.array([100, 100, 20, 10, 0])
        poly = xywhr2xyxyxyxy(boxes)
        assert poly.shape == (8,)

    def test_batch_boxes(self):
        """批量框"""
        boxes = np.array([
            [0, 0, 10, 10, 0],
            [20, 20, 10, 10, 0],
        ])
        poly = xywhr2xyxyxyxy(boxes)
        assert poly.shape == (2, 8)


class TestOBB:
    """OBB dataclass 测试"""

    def test_creation(self):
        """创建 OBB 实例"""
        obb = OBB(cx=0, cy=0, w=10, h=10, angle=0, class_id=0, confidence=0.9)
        assert obb.cx == 0
        assert obb.confidence == 0.9

    def test_default_confidence(self):
        """默认置信度"""
        obb = OBB(cx=0, cy=0, w=10, h=10, angle=0, class_id=0)
        assert obb.confidence == 1.0
