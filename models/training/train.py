"""
YOLO-OBB 训练脚本

基于 Ultralytics YOLO-OBB 框架训练旋转目标检测模型
"""

import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO
from utils.callbacks import Callbacks


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train(
    config_path: str = './configs/training_config.yaml',
    **kwargs
) -> Optional[YOLO]:
    """
    训练 YOLO-OBB 模型

    Args:
        config_path: 配置文件路径
        **kwargs: 覆盖配置参数

    Returns:
        训练好的模型
    """
    # 加载配置
    config = load_config(config_path)

    data_config = config['data']
    model_config = config['model']
    train_config = config['train']
    checkpoint_config = config['checkpoint']

    # 合并命令行参数
    for key, value in kwargs.items():
        if value is not None:
            if '.' in key:
                # 嵌套参数，如 data.dataset_root
                keys = key.split('.')
                target = config
                for k in keys[:-1]:
                    target = target.get(k, {})
                target[keys[-1]] = value
            else:
                # 顶层参数
                if key in train_config:
                    train_config[key] = value

    # 创建保存目录
    save_dir = Path(checkpoint_config['save_dir']) / checkpoint_config['project']
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    if model_config.get('pretrained', True):
        # 使用预训练权重
        model = YOLO('yolov8n-obb.pt')
        print("使用预训练权重：yolov8n-obb.pt")
    else:
        # 从头训练
        model = YOLO(model_config['architecture'])
        print("从头训练模型")

    # 训练参数
    train_args = {
        # 数据
        'data': data_config.get('dataset_yaml', './data/datasets/DOTA/yolo_obb/dataset.yaml'),
        'imgsz': train_config['imgsz'],

        # 训练超参数
        'epochs': train_config['epochs'],
        'batch': train_config['batch_size'],
        'optimizer': train_config['optimizer'],
        'lr0': train_config['lr0'],
        'lrf': train_config['lrf'],
        'momentum': train_config['momentum'],
        'weight_decay': train_config['weight_decay'],
        'warmup_epochs': train_config['warmup_epochs'],
        'warmup_momentum': train_config['warmup_momentum'],

        # 数据增强
        'hsv_h': train_config['augment']['hsv_h'],
        'hsv_s': train_config['augment']['hsv_s'],
        'hsv_v': train_config['augment']['hsv_v'],
        'scale': train_config['augment']['scale'],
        'translate': train_config['augment']['translate'],
        'mosaic': train_config['augment']['mosaic'],
        'mixup': train_config['augment']['mixup'],

        # Loss 系数
        'box': train_config['box'],
        'cls': train_config['cls'],
        'dfl': train_config['dfl'],

        # 其他
        'amp': train_config['amp'],
        'patience': train_config['patience'],
        'save_period': train_config['save_period'],
        'project': str(save_dir),
        'name': checkpoint_config['name'],
        'exist_ok': checkpoint_config['exist_ok'],
        'verbose': True,
        'save': True,
        'save_txt': True,
        'save_csv': True,
    }

    # 设备配置
    device = train_config.get('device', 0)
    if isinstance(device, int) and device >= 0:
        train_args['device'] = str(device)

    # 恢复训练
    if checkpoint_config.get('resume', False):
        # 查找最新的 checkpoint
        last_ckpt = save_dir / checkpoint_config['name'] / 'weights' / 'last.pt'
        if last_ckpt.exists():
            train_args['resume'] = str(last_ckpt)
            print(f"从 checkpoint 恢复训练：{last_ckpt}")

    print("\n" + "=" * 50)
    print("开始训练 YOLO-OBB 模型")
    print("=" * 50)
    print(f"数据集：{data_config['dataset_name']}")
    print(f"图像尺寸：{train_config['imgsz']}")
    print(f"Batch size: {train_config['batch_size']}")
    print(f"Epochs: {train_config['epochs']}")
    print(f"保存目录：{save_dir}")
    print("=" * 50 + "\n")

    # 开始训练
    results = model.train(**train_args)

    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)

    return model


def validate(
    model_path: str,
    config_path: str = './configs/training_config.yaml',
    data_path: Optional[str] = None
) -> Dict[str, float]:
    """
    验证模型

    Args:
        model_path: 模型权重路径
        config_path: 配置文件路径
        data_path: 数据配置路径 (可选)

    Returns:
        验证结果指标
    """
    config = load_config(config_path)
    val_config = config['val']

    # 加载模型
    model = YOLO(model_path)

    # 验证参数
    val_args = {
        'data': data_path or config['data'].get('dataset_yaml'),
        'imgsz': val_config['imgsz'],
        'batch': val_config['batch_size'],
        'iou': val_config['iou_thres_nms'],
        'conf': val_config['conf_thres'],
        'save_json': val_config.get('save_json', False),
        'save_hybrid': val_config.get('save_hybrid', False),
        'verbose': True,
    }

    print("\n" + "=" * 50)
    print("开始验证模型")
    print("=" * 50)
    print(f"模型：{model_path}")
    print(f"IoU 阈值：{val_config['iou_thres']}")
    print("=" * 50 + "\n")

    # 执行验证
    metrics = model.val(**val_args)

    print("\n" + "=" * 50)
    print("验证结果")
    print("=" * 50)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print("=" * 50)

    return {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr)
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLO-OBB 训练脚本')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/training_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='数据集配置路径 (覆盖配置)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='训练轮数 (覆盖配置)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=None,
        help='批次大小 (覆盖配置)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=None,
        help='输入图像尺寸 (覆盖配置)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='训练设备 (覆盖配置)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='从 checkpoint 恢复训练'
    )
    parser.add_argument(
        '--validate',
        type=str,
        default=None,
        help='验证指定模型 (模型路径)'
    )

    args = parser.parse_args()

    if args.validate:
        # 验证模式
        validate(args.validate, args.config)
    else:
        # 训练模式
        train(
            config_path=args.config,
            data=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            resume=args.resume
        )
