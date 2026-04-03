"""
OBB (Oriented Bounding Box) 工具函数

提供旋转框的坐标转换、IoU 计算、NMS 等功能
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OBB:
    """旋转框数据结构"""
    cx: float  # 中心点 x
    cy: float  # 中心点 y
    w: float   # 宽度
    h: float   # 高度
    angle: float  # 旋转角度 (弧度制，逆时针为正)
    class_id: int
    confidence: float = 1.0


def rbox2poly(rbox: np.ndarray) -> np.ndarray:
    """
    旋转框转为多边形表示

    Args:
        rbox: [cx, cy, w, h, angle] 或 [cx, cy, w, h, angle, cls, conf]

    Returns:
        polygons: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    cx, cy, w, h, angle = rbox[:5]

    # 计算旋转矩阵
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)

    # 四个角点 (相对于中心)
    dx = w / 2
    dy = h / 2

    corners = np.array([
        [-dx, -dy],
        [dx, -dy],
        [dx, dy],
        [-dx, dy]
    ])

    # 旋转
    rot_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
    rotated = corners @ rot_matrix.T

    # 平移到中心点
    rotated[:, 0] += cx
    rotated[:, 1] += cy

    return rotated.reshape(-1)


def poly2rbox(poly: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    多边形转为旋转框表示

    Args:
        poly: [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns:
        [cx, cy, w, h, angle]
    """
    poly = poly.reshape(4, 2)

    # 最小外接矩形
    rect = cv2.minAreaRect(poly.astype(np.float32))
    (cx, cy), (w, h), angle = rect

    return cx, cy, w, h, np.radians(angle)


def obb_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个旋转框的 IoU

    Args:
        box1, box2: [cx, cy, w, h, angle]

    Returns:
        IoU value
    """
    poly1 = rbox2poly(box1).astype(np.float32)
    poly2 = rbox2poly(box2).astype(np.float32)

    # 转为 OpenCV 格式
    poly1_cv = poly1.reshape(4, 2)
    poly2_cv = poly2.reshape(4, 2)

    # 计算交集面积
    intersection = cv2.intersectConvexConvex(poly1_cv, poly2_cv)
    if intersection is None:
        return 0.0
    inter_area = intersection[0]

    # 计算各自面积
    area1 = cv2.contourArea(poly1_cv)
    area2 = cv2.contourArea(poly2_cv)

    # IoU
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def obb_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    批量计算 IoU 矩阵

    Args:
        boxes1: [N, 5] - [cx, cy, w, h, angle]
        boxes2: [M, 5]

    Returns:
        iou_matrix: [N, M]
    """
    n, m = len(boxes1), len(boxes2)
    iou_matrix = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = obb_iou(boxes1[i], boxes2[j])

    return iou_matrix


def rotate_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.7,
    class_agnostic: bool = False
) -> np.ndarray:
    """
    旋转框 NMS

    Args:
        boxes: [N, 5] - [cx, cy, w, h, angle]
        scores: [N] - 置信度
        iou_threshold: IoU 阈值
        class_agnostic: 是否类别无关 NMS

    Returns:
        keep_indices: 保留的索引
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    # 按置信度排序
    order = np.argsort(-scores)
    keep = []

    while len(order) > 0:
        # 选择最高分的框
        idx = order[0]
        keep.append(idx)

        if len(order) == 1:
            break

        # 计算与其余框的 IoU
        rest_boxes = boxes[order[1:]]
        ious = np.array([
            obb_iou(boxes[idx], rest_box)
            for rest_box in rest_boxes
        ])

        # 保留 IoU 低于阈值的框
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def xywhr2xyxyxyxy(boxes: np.ndarray) -> np.ndarray:
    """
    [cx, cy, w, h, angle] → [x1,y1, x2,y2, x3,y3, x4,y4]

    Args:
        boxes: [N, 5]

    Returns:
        polygons: [N, 8]
    """
    if boxes.ndim == 1:
        return rbox2poly(boxes)

    polygons = np.zeros((len(boxes), 8), dtype=np.float32)
    for i, box in enumerate(boxes):
        polygons[i] = rbox2poly(box)
    return polygons


def draw_obb(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_names: List[str],
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制旋转框

    Args:
        image: BGR 图像
        boxes: [N, 5] - [cx, cy, w, h, angle]
        scores: [N] - 置信度
        class_names: 类别名称列表
        thickness: 线条粗细

    Returns:
        绘制后的图像
    """
    img = image.copy()

    for i, (box, score) in enumerate(zip(boxes, scores)):
        poly = rbox2poly(box).astype(np.int32)
        poly_pts = poly.reshape(4, 2)

        # 绘制边框
        color = (np.random.randint(0, 255),
                 np.random.randint(0, 255),
                 np.random.randint(0, 255))
        cv2.polylines(img, [poly_pts], True, color, thickness)

        # 绘制标签
        class_id = int(box[5]) if len(box) > 5 else 0
        label = f"{class_names[class_id]}: {score:.2f}"
        cv2.putText(img, label, (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img
