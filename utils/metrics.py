"""
评估指标计算工具

支持 mAP、Precision、Recall 等指标计算
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from .obb_utils import obb_iou


@dataclass
class Detection:
    """检测结果"""
    box: np.ndarray  # [cx, cy, w, h, angle]
    score: float
    class_id: int
    matched: bool = False


@dataclass
class GroundTruth:
    """真实标注"""
    box: np.ndarray
    class_id: int
    matched: bool = False


def compute_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """
    计算 Average Precision (11 点插值法)

    Args:
        precisions: 精度序列
        recalls: 召回率序列

    Returns:
        AP value
    """
    if len(recalls) == 0:
        return 0.0

    # 添加边界
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # 单调化
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 11 点插值
    recall_thresholds = np.linspace(0.0, 1.0, 11)
    ap = 0.0

    for t in recall_thresholds:
        idx = np.where(recalls >= t)[0]
        if len(idx) > 0:
            ap += precisions[idx[0]] / 11.0

    return ap


def compute_obb_map(
    predictions: List[Detection],
    ground_truths: List[GroundTruth],
    iou_threshold: float = 0.5,
    num_classes: int = 15
) -> Dict[str, float]:
    """
    计算 OBB 检测的 mAP

    Args:
        predictions: 所有预测结果
        ground_truths: 所有真实标注
        iou_threshold: IoU 阈值
        num_classes: 类别数

    Returns:
        {
            'mAP50': float,
            'AP_per_class': Dict[int, float],
            'precision': float,
            'recall': float
        }
    """
    # 按类别分组
    pred_by_cls = defaultdict(list)
    gt_by_cls = defaultdict(list)

    for pred in predictions:
        pred_by_cls[pred.class_id].append(pred)
    for gt in ground_truths:
        gt_by_cls[gt.class_id].append(gt)

    ap_per_class = {}

    for cls_id in range(num_classes):
        cls_preds = sorted(
            pred_by_cls[cls_id],
            key=lambda x: x.score,
            reverse=True
        )
        cls_gts = gt_by_cls[cls_id]

        if len(cls_gts) == 0:
            ap_per_class[cls_id] = 0.0
            continue

        # 匹配预测和真实框
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        for i, pred in enumerate(cls_preds):
            best_iou = 0.0
            best_gt_idx = -1

            for j, gt in enumerate(cls_gts):
                if gt.matched:
                    continue

                iou = obb_iou(pred.box, gt.box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[i] = 1
                cls_gts[best_gt_idx].matched = True
                pred.matched = True
            else:
                fp[i] = 1

        # 计算 precision 和 recall 曲线
        n_pos = len(cls_gts)
        cum_fp = np.cumsum(fp)
        cum_tp = np.cumsum(tp)

        precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
        recalls = cum_tp / n_pos

        # 计算 AP
        ap_per_class[cls_id] = compute_ap(precisions, recalls)

    # mAP
    mAP50 = np.mean(list(ap_per_class.values()))

    # 总体 precision 和 recall
    total_tp = sum(1 for p in predictions if p.matched)
    total_pred = len(predictions)
    total_gt = len(ground_truths)

    precision = total_tp / (total_pred + 1e-8)
    recall = total_tp / (total_gt + 1e-8)

    return {
        'mAP50': mAP50,
        'AP_per_class': ap_per_class,
        'precision': precision,
        'recall': recall
    }
