"""
DOTA 数据集预处理脚本

将 DOTA 原始标注格式转换为 YOLO-OBB 格式
原始格式：x1 y1 x2 y2 x3 y4 x4 y5 class_name difficulty
YOLO-OBB 格式：class_id cx cy w h angle (归一化)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
import shutil


# DOTA 类别映射
DOTA_CLASSES = [
    'plane',
    'baseball-diamond',
    'bridge',
    'ground-track-field',
    'small-vehicle',
    'large-vehicle',
    'ship',
    'tennis-court',
    'basketball-court',
    'storage-tank',
    'soccer-ball-field',
    'roundabout',
    'harbor',
    'swimming-pool',
    'helicopter'
]

CLASS_TO_ID = {cls: idx for idx, cls in enumerate(DOTA_CLASSES)}


def parse_dota_label(label_path: str) -> List[Dict]:
    """
    解析 DOTA 原始标注文件

    Args:
        label_path: 标注文件路径

    Returns:
        标注列表，每项包含：
        {
            'poly': [x1,y1,x2,y2,x3,y3,x4,y4],
            'class': class_name,
            'difficulty': int
        }
    """
    annotations = []

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 9:
                continue

            # 解析多边形坐标
            poly = [float(x) for x in parts[:8]]
            class_name = parts[8]
            difficulty = int(parts[9]) if len(parts) > 9 else 0

            annotations.append({
                'poly': poly,
                'class': class_name,
                'difficulty': difficulty
            })

    return annotations


def poly2rbox(poly: np.ndarray, img_w: int, img_h: int) -> Tuple[float, float, float, float, float]:
    """
    将多边形转换为旋转框 (归一化)

    Args:
        poly: [x1, y1, x2, y2, x3, y3, x4, y4]
        img_w, img_h: 图像宽高

    Returns:
        (cx_norm, cy_norm, w_norm, h_norm, angle)
    """
    poly_pts = np.array(poly).reshape(4, 2)

    # 计算最小外接矩形
    rect = cv2.minAreaRect(poly_pts.astype(np.float32))
    (cx, cy), (w, h), angle = rect

    # 归一化
    cx_norm = cx / img_w
    cy_norm = cy / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # 角度转换 (DOTA 的角度定义)
    # OpenCV 返回的角度范围是 [-90, 0)，需要转换为 [0, π/2)
    angle_rad = np.radians(angle)

    # YOLO-OBB 的角度定义：从 x 轴正方向逆时针旋转
    # 需要调整角度到 [-π/4, 3π/4) 范围
    if angle < -45:
        angle = angle + 90
    angle_rad = np.radians(angle)

    return cx_norm, cy_norm, w_norm, h_norm, angle_rad


def convert_label(
    label_path: str,
    output_path: str,
    img_w: int,
    img_h: int,
    skip_difficult: bool = True
) -> int:
    """
    转换单个标注文件

    Args:
        label_path: DOTA 标注文件路径
        output_path: 输出 YOLO 格式标注路径
        img_w, img_h: 对应图像的宽高
        skip_difficult: 是否跳过困难样本

    Returns:
        转换的标注数量
    """
    annotations = parse_dota_label(label_path)

    yolo_labels = []

    for ann in annotations:
        if skip_difficult and ann['difficulty'] != 0:
            continue

        class_name = ann['class']
        if class_name not in CLASS_TO_ID:
            continue

        class_id = CLASS_TO_ID[class_name]
        poly = np.array(ann['poly'])

        try:
            cx, cy, w, h, angle = poly2rbox(poly, img_w, img_h)

            # 过滤无效标注
            if w <= 0 or h <= 0:
                continue
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                continue

            yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {angle:.6f}")

        except Exception as e:
            print(f"转换失败 {label_path}: {e}")
            continue

    # 写入文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for label in yolo_labels:
            f.write(label + '\n')

    return len(yolo_labels)


def prepare_dota(
    dataset_root: str = './data/datasets/DOTA',
    output_format: str = 'yolo-obb',
    image_size: Tuple[int, int] = (1024, 1024),
    skip_difficult: bool = True,
    clean_existing: bool = False
) -> None:
    """
    准备 DOTA 数据集

    Args:
        dataset_root: 数据集根目录
        output_format: 输出格式 ('yolo-obb')
        image_size: 目标图像尺寸 (宽，高)
        skip_difficult: 是否跳过困难样本
        clean_existing: 是否清理已存在的输出目录
    """
    dataset_root = Path(dataset_root)

    # 输入目录
    if (dataset_root / 'origin').exists():
        origin_root = dataset_root / 'origin'
    else:
        # 尝试其他可能的目录结构
        origin_root = dataset_root

    # 输出目录
    yolo_root = dataset_root / 'yolo_obb'

    if clean_existing and yolo_root.exists():
        shutil.rmtree(yolo_root)

    yolo_root.mkdir(parents=True, exist_ok=True)

    splits = ['train', 'val']

    for split in splits:
        print(f"\n处理 {split} 集...")

        # 目录结构
        split_images = yolo_root / split / 'images'
        split_labels = yolo_root / split / 'labels'
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)

        # 查找图像和标注目录
        if (origin_root / 'train' / 'images').exists():
            images_dir = origin_root / split / 'images'
            labels_dir = origin_root / split / 'labels'
        elif (origin_root / 'images').exists():
            images_dir = origin_root / 'images'
            labels_dir = origin_root / 'labels'
        else:
            # 尝试 DOTA 原始解压目录
            images_dir = origin_root / split
            labels_dir = origin_root / f'{split}_labelTxt'

        if not images_dir.exists():
            print(f"  警告：找不到 {split} 图像目录")
            continue

        if not labels_dir.exists():
            print(f"  警告：找不到 {split} 标注目录")
            continue

        # 处理每个样本
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.bmp']
        total_images = 0
        total_annotations = 0

        for img_path in images_dir.glob('*'):
            if img_path.suffix.lower() not in image_extensions:
                continue

            total_images += 1

            # 读取图像尺寸
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                orig_h, orig_w = img.shape[:2]
            except Exception as e:
                print(f"  读取图像失败 {img_path}: {e}")
                continue

            # 复制图像 (可选：这里可以添加图像裁剪/缩放逻辑)
            dst_img_path = split_images / f"{img_path.stem}.png"
            if image_size is not None:
                # 缩放到目标尺寸
                img_resized = cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(dst_img_path), img_resized)
            else:
                shutil.copy(img_path, dst_img_path)

            # 查找对应标注文件
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                # 尝试其他可能的标注文件扩展名
                label_path = labels_dir / f"{img_path.stem}.Txt"

            if not label_path.exists():
                continue

            # 转换标注
            dst_label_path = split_labels / f"{img_path.stem}.txt"

            if image_size is not None:
                new_w, new_h = image_size
            else:
                new_w, new_h = orig_w, orig_h

            num_anns = convert_label(
                str(label_path),
                str(dst_label_path),
                new_w,
                new_h,
                skip_difficult=skip_difficult
            )

            total_annotations += num_anns

        print(f"  完成：{total_images} 张图像，{total_annotations} 个标注")

    # 创建数据集 YAML 配置
    create_dataset_yaml(yolo_root, dataset_root)

    print(f"\n数据集准备完成！")
    print(f"输出目录：{yolo_root}")


def create_dataset_yaml(yolo_root: Path, dataset_root: Path) -> None:
    """创建 YOLO 数据集配置文件"""

    yaml_content = f"""# DOTA dataset for YOLO-OBB
path: {dataset_root.absolute()}
train: yolo_obb/train/images
val: yolo_obb/val/images

# 类别定义
names:
  0: plane
  1: baseball-diamond
  2: bridge
  3: ground-track-field
  4: small-vehicle
  5: large-vehicle
  6: ship
  7: tennis-court
  8: basketball-court
  9: storage-tank
  10: soccer-ball-field
  11: roundabout
  12: harbor
  13: swimming-pool
  14: helicopter

nc: 15
"""

    yaml_path = yolo_root / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"数据集配置已保存到：{yaml_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='预处理 DOTA 数据集')
    parser.add_argument(
        '--root',
        type=str,
        default='./data/datasets/DOTA',
        help='数据集根目录'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        default=[1024, 1024],
        help='目标图像尺寸 [宽，高]'
    )
    parser.add_argument(
        '--no-skip-difficult',
        action='store_true',
        help='不跳过困难样本'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='清理已存在的输出目录'
    )

    args = parser.parse_args()

    prepare_dota(
        dataset_root=args.root,
        image_size=tuple(args.image_size),
        skip_difficult=not args.no_skip_difficult,
        clean_existing=args.clean
    )
