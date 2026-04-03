"""
训练回调函数

用于记录训练进度、保存最佳模型等
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json


class Callbacks:
    """训练回调"""

    def __init__(self, save_dir: str, target_metrics: Optional[Dict[str, float]] = None):
        """
        初始化回调

        Args:
            save_dir: 保存目录
            target_metrics: 目标指标，如 {'mAP50': 0.80, 'precision': 0.90}
        """
        self.save_dir = Path(save_dir)
        self.target_metrics = target_metrics or {}
        self.best_metrics = {}
        self.history = []

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        每个 epoch 结束时调用

        Args:
            epoch: 当前 epoch
            metrics: 当前指标
        """
        self.history.append({
            'epoch': epoch,
            **metrics
        })

        # 检查是否达到目标
        if self.target_metrics:
            for metric_name, target_value in self.target_metrics.items():
                current_value = metrics.get(metric_name, 0)
                if current_value >= target_value:
                    print(f"\n✓ 达到目标 {metric_name}: {current_value:.4f} >= {target_value:.4f}")

        # 保存最佳模型
        self._save_best_model(epoch, metrics)

        # 保存训练历史
        self._save_history()

    def _save_best_model(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """保存最佳模型权重"""
        # 主要指标：mAP50
        current_map = metrics.get('mAP50', 0)
        best_map = self.best_metrics.get('mAP50', 0)

        if current_map > best_map:
            self.best_metrics = {
                'mAP50': current_map,
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'epoch': epoch
            }
            print(f"\n新的最佳模型 (epoch {epoch}): mAP50={current_map:.4f}")

    def _save_history(self) -> None:
        """保存训练历史"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_summary(self) -> str:
        """获取训练摘要"""
        if not self.best_metrics:
            return "暂无训练结果"

        return (
            f"最佳模型 (epoch {self.best_metrics.get('epoch', 'N/A')}):\n"
            f"  mAP50: {self.best_metrics.get('mAP50', 0):.4f}\n"
            f"  Precision: {self.best_metrics.get('precision', 0):.4f}\n"
            f"  Recall: {self.best_metrics.get('recall', 0):.4f}"
        )
