"""
DOTA 数据集预处理脚本

将 DOTA 原始标注格式转换为 YOLOv8-OBB 格式
原始格式：x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
YOLOv8-OBB 格式：class_id x1 y1 x2 y2 x3 y3 x4 y4（归一化，四顶点）

对齐 data_prepare.py：
  - 输入：RAW_IMAGE_DIR（大图）+ RAW_LABEL_DIR（labelTxt 标注）
  - 输出：OUTPUT_DIR/images/{train,val,test} + labels/{train,val,test} + data.yaml
  - 标注格式：四顶点归一化（而非 cx cy w h angle）
  - 类别映射顺序与 data_prepare.py 一致
  - 滑窗切图（crop_size=1024, stride=512，50% 重叠）
  - 图像保存为 .jpg
  - 随机划分 train/val/test（7:2:1）
"""

import os
import cv2
import numpy as np
import shutil
import yaml
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


# ══════════════════════════════════════════════════
#  全局路径配置（与 data_prepare.py 保持一致）
# ══════════════════════════════════════════════════
RAW_IMAGE_DIR = "/data/fcj/raw_data/images"    # 原始大图目录
RAW_LABEL_DIR = "/data/fcj/raw_data/labelTxt"  # DOTA 标注目录
OUTPUT_DIR    = "/data/fcj/output"             # 输出根目录

# ══════════════════════════════════════════════════
#  类别映射（与 data_prepare.py 保持一致）
# ══════════════════════════════════════════════════
CATEGORY_MAP = {
    "plane":              0,
    "ship":               1,
    "storage-tank":       2,
    "baseball-diamond":   3,
    "tennis-court":       4,
    "basketball-court":   5,
    "ground-track-field": 6,
    "harbor":             7,
    "bridge":             8,
    "large-vehicle":      9,
    "small-vehicle":      10,
    "helicopter":         11,
    "roundabout":         12,
    "soccer-ball-field":  13,
    "swimming-pool":      14,
}

# 若只关注部分类别，在此过滤（None = 保留全部）
KEEP_CLASSES: Optional[set] = None  # 例如: {"plane", "ship"}

# 训练/验证/测试划分比例
SPLIT_RATIO = {"train": 0.7, "val": 0.2, "test": 0.1}

# 切图参数
CROP_SIZE      = 1024   # 切图大小（像素），V100 16G 建议 1024
CROP_STRIDE    = 512    # 滑窗步长（50% 重叠）
MIN_AREA_RATIO = 0.2    # 目标框被裁剪后保留的最小面积比


# ══════════════════════════════════════════════════
#  1. 解析 DOTA 标注文件
# ══════════════════════════════════════════════════
def parse_dota_label(label_path: str) -> List[Dict]:
    """
    解析 DOTA 原始标注文件

    Returns:
        标注列表，每项包含：
        {
            'poly': np.array (4,2),  # 四边形顶点，绝对像素坐标
            'category': str,
            'difficult': int
        }
    """
    annotations = []

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过 DOTA 文件头（imagesource / gsd）
            if line.startswith('imagesource') or line.startswith('gsd'):
                continue
            if not line:
                continue

            parts = line.split()
            if len(parts) < 9:
                continue

            try:
                coords = list(map(float, parts[:8]))
                class_name = parts[8].lower()
                difficulty = int(parts[9]) if len(parts) > 9 else 0
            except ValueError:
                continue

            if KEEP_CLASSES and class_name not in KEEP_CLASSES:
                continue
            if class_name not in CATEGORY_MAP:
                continue

            poly = np.array(coords, dtype=np.float32).reshape(4, 2)
            annotations.append({
                'poly':      poly,
                'category':  class_name,
                'difficult': difficulty
            })

    return annotations


# ══════════════════════════════════════════════════
#  2. 切图 + 标注裁剪
# ══════════════════════════════════════════════════
def polygon_area(poly: np.ndarray) -> float:
    """Shoelace 公式计算多边形面积"""
    n = len(poly)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1]
        area -= poly[j][0] * poly[i][1]
    return abs(area) / 2.0


def clip_polygon_to_box(poly: np.ndarray, x0, y0, x1, y1) -> Optional[np.ndarray]:
    """
    用 Sutherland-Hodgman 算法将多边形裁剪到矩形框 [x0,x1]×[y0,y1]。
    返回裁剪后顶点数组，若结果为空则返回 None。
    """
    def inside(p, edge):
        return (edge[2] - edge[0]) * (p[1] - edge[1]) - \
               (edge[3] - edge[1]) * (p[0] - edge[0]) >= 0

    def intersection(p1, p2, edge):
        x1_, y1_ = p1
        x2_, y2_ = p2
        ex1, ey1, ex2, ey2 = edge
        dx, dy = x2_ - x1_, y2_ - y1_
        edx, edy = ex2 - ex1, ey2 - ey1
        denom = dx * edy - dy * edx
        if abs(denom) < 1e-10:
            return p1
        t = ((ex1 - x1_) * edy - (ey1 - y1_) * edx) / denom
        return [x1_ + t * dx, y1_ + t * dy]

    edges = [
        (x0, y0, x1, y0),  # 上边
        (x1, y0, x1, y1),  # 右边
        (x1, y1, x0, y1),  # 下边
        (x0, y1, x0, y0),  # 左边
    ]

    output = poly.tolist()
    for edge in edges:
        if not output:
            return None
        inp = output
        output = []
        for idx in range(len(inp)):
            cur = inp[idx]
            prev = inp[idx - 1]
            if inside(cur, edge):
                if not inside(prev, edge):
                    output.append(intersection(prev, cur, edge))
                output.append(cur)
            elif inside(prev, edge):
                output.append(intersection(prev, cur, edge))

    if len(output) < 3:
        return None
    return np.array(output, dtype=np.float32)


def crop_image_and_labels(
    img: np.ndarray,
    annotations: List[Dict],
    crop_size: int,
    stride: int
) -> List[Tuple]:
    """
    对大图进行滑窗切割。
    返回列表，每个元素为 (crop_img, crop_annotations, x_offset, y_offset)。
    """
    h, w = img.shape[:2]
    crops = []

    x_starts = list(range(0, max(w - crop_size, 0) + 1, stride))
    y_starts = list(range(0, max(h - crop_size, 0) + 1, stride))

    # 补充覆盖右/下边缘的最后一个窗口
    if not x_starts or x_starts[-1] + crop_size < w:
        x_starts.append(max(w - crop_size, 0))
    if not y_starts or y_starts[-1] + crop_size < h:
        y_starts.append(max(h - crop_size, 0))

    for y0 in y_starts:
        for x0 in x_starts:
            x1_win = x0 + crop_size
            y1_win = y0 + crop_size

            crop = img[y0:y1_win, x0:x1_win].copy()
            actual_h, actual_w = crop.shape[:2]
            if actual_h < crop_size or actual_w < crop_size:
                padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                padded[:actual_h, :actual_w] = crop
                crop = padded

            crop_annots = []
            for ann in annotations:
                poly_abs = ann['poly'].copy()

                # 平移到切图坐标系
                poly_crop = poly_abs - np.array([x0, y0], dtype=np.float32)

                orig_area = polygon_area(poly_abs)
                if orig_area < 1e-6:
                    continue

                clipped_poly = clip_polygon_to_box(poly_crop, 0, 0, crop_size, crop_size)
                if clipped_poly is None:
                    continue

                clipped_area = polygon_area(clipped_poly)
                if clipped_area / orig_area < MIN_AREA_RATIO:
                    continue

                # 用最小外接旋转矩形的四顶点作为最终标注
                clipped_pts = clipped_poly.astype(np.float32)
                rect = cv2.minAreaRect(clipped_pts)
                box_pts = cv2.boxPoints(rect)          # shape (4,2)，顺时针
                box_pts = np.clip(box_pts, 0, crop_size - 1)

                crop_annots.append({
                    'category':  ann['category'],
                    'difficult': ann['difficult'],
                    'poly':      box_pts
                })

            crops.append((crop, crop_annots, x0, y0))

    return crops


# ══════════════════════════════════════════════════
#  3. 保存切图及 YOLOv8-OBB 格式标注
# ══════════════════════════════════════════════════
def save_crop(
    crop_img: np.ndarray,
    crop_annots: List[Dict],
    img_save_path: str,
    lbl_save_path: str,
    skip_difficult: bool = True
) -> None:
    """
    保存切图和对应的 YOLOv8-OBB 格式标注 txt。

    YOLOv8-OBB 格式（每行）：
      class_id  x1 y1  x2 y2  x3 y3  x4 y4
    所有坐标均归一化到 [0, 1]。
    """
    cv2.imwrite(img_save_path, crop_img)

    h, w = crop_img.shape[:2]
    lines = []
    for ann in crop_annots:
        if skip_difficult and ann['difficult'] != 0:
            continue

        cls_id = CATEGORY_MAP[ann['category']]
        poly_norm = ann['poly'].copy().astype(np.float64)
        poly_norm[:, 0] /= w
        poly_norm[:, 1] /= h
        poly_norm = np.clip(poly_norm, 0.0, 1.0)

        coords_str = ' '.join(f'{v:.6f}' for v in poly_norm.flatten())
        lines.append(f'{cls_id} {coords_str}')

    with open(lbl_save_path, 'w') as f:
        f.write('\n'.join(lines))


# ══════════════════════════════════════════════════
#  4. 主函数
# ══════════════════════════════════════════════════
def prepare_dota(
    raw_image_dir: str = RAW_IMAGE_DIR,
    raw_label_dir: str = RAW_LABEL_DIR,
    output_dir: str = OUTPUT_DIR,
    crop_size: int = CROP_SIZE,
    crop_stride: int = CROP_STRIDE,
    skip_difficult: bool = True,
    clean_existing: bool = False,
    random_seed: int = 42
) -> None:
    """
    准备 DOTA 数据集（滑窗切图 + YOLOv8-OBB 四顶点格式）

    目录约定（与 data_prepare.py 完全一致）：
      raw_image_dir/   ← 原始大图，如 P0001.png
      raw_label_dir/   ← DOTA 标注，如 P0001.txt
      output_dir/
        images/{train,val,test}/
        labels/{train,val,test}/
        data.yaml

    Args:
        raw_image_dir:  原始大图目录
        raw_label_dir:  DOTA 标注目录（labelTxt 格式）
        output_dir:     输出根目录
        crop_size:      切图大小（像素）
        crop_stride:    滑窗步长
        skip_difficult: 是否跳过困难样本
        clean_existing: 是否清理已存在的输出目录
        random_seed:    随机种子，用于 train/val/test 划分
    """
    random.seed(random_seed)

    raw_img_dir = Path(raw_image_dir)
    raw_lbl_dir = Path(raw_label_dir)
    output_dir  = Path(output_dir)

    if clean_existing and output_dir.exists():
        shutil.rmtree(output_dir)

    # 临时目录（切图后统一划分）
    tmp_img_dir = output_dir / '_tmp_images'
    tmp_lbl_dir = output_dir / '_tmp_labels'
    tmp_img_dir.mkdir(parents=True, exist_ok=True)
    tmp_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    all_img_paths = sorted([
        p for p in raw_img_dir.iterdir()
        if p.suffix.lower() in img_extensions
    ])
    print(f'[INFO] 共找到 {len(all_img_paths)} 张原始图像')

    all_stems = []

    for img_path in tqdm(all_img_paths, desc='切图进度'):
        # 查找标注文件（兼容 .txt / .Txt）
        lbl_path = raw_lbl_dir / f'{img_path.stem}.txt'
        if not lbl_path.exists():
            lbl_path = raw_lbl_dir / f'{img_path.stem}.Txt'
        if not lbl_path.exists():
            print(f'[WARN] 标注文件不存在，跳过：{img_path.stem}')
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f'[WARN] 无法读取图像，跳过：{img_path}')
            continue

        annotations = parse_dota_label(str(lbl_path))
        crops = crop_image_and_labels(img, annotations, crop_size, crop_stride)

        for crop_img, crop_annots, x0, y0 in crops:
            # 跳过无有效标注的切图（如需保留背景片可注释此行）
            valid_annots = [
                a for a in crop_annots
                if not (skip_difficult and a['difficult'] != 0)
            ]
            if not valid_annots:
                continue

            stem = f'{img_path.stem}__cx{x0}_cy{y0}'
            img_save = tmp_img_dir / f'{stem}.jpg'
            lbl_save = tmp_lbl_dir / f'{stem}.txt'
            save_crop(
                crop_img, crop_annots,
                str(img_save), str(lbl_save),
                skip_difficult=skip_difficult
            )
            all_stems.append(stem)

    print(f'[INFO] 切图完成，共生成 {len(all_stems)} 张有效切图')

    # ── 随机划分 train / val / test ────────────────
    random.shuffle(all_stems)
    n = len(all_stems)
    n_train = int(n * SPLIT_RATIO['train'])
    n_val   = int(n * SPLIT_RATIO['val'])

    splits = {
        'train': all_stems[:n_train],
        'val':   all_stems[n_train:n_train + n_val],
        'test':  all_stems[n_train + n_val:]
    }

    for split, stems in splits.items():
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for stem in tqdm(stems, desc=f'复制 {split} 集'):
            shutil.copy(
                tmp_img_dir / f'{stem}.jpg',
                output_dir / 'images' / split / f'{stem}.jpg'
            )
            shutil.copy(
                tmp_lbl_dir / f'{stem}.txt',
                output_dir / 'labels' / split / f'{stem}.txt'
            )
        print(f'[INFO] {split}: {len(stems)} 张')

    # 清理临时目录
    shutil.rmtree(tmp_img_dir)
    shutil.rmtree(tmp_lbl_dir)

    # ── 生成 data.yaml ─────────────────────────────
    create_dataset_yaml(output_dir)

    print(f'\n[INFO] 数据集准备完成！输出目录：{output_dir}')
    print(f'       train: {len(splits["train"])}  '
          f'val: {len(splits["val"])}  '
          f'test: {len(splits["test"])}')


def create_dataset_yaml(output_dir: Path) -> None:
    """生成 YOLOv8 数据集配置文件"""
    nc = len(CATEGORY_MAP)
    names_list = [k for k, _ in sorted(CATEGORY_MAP.items(), key=lambda x: x[1])]

    data_yaml = {
        'path':  str(output_dir.resolve()),
        'train': 'images/train',
        'val':   'images/val',
        'test':  'images/test',
        'nc':    nc,
        'names': names_list
    }

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)

    print(f'[INFO] data.yaml 已生成：{yaml_path}')


# ══════════════════════════════════════════════════
#  命令行入口
# ══════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='预处理 DOTA 数据集（YOLOv8-OBB 四顶点格式）')
    parser.add_argument(
        '--image-dir',
        type=str,
        default=RAW_IMAGE_DIR,
        help=f'原始大图目录（默认：{RAW_IMAGE_DIR}）'
    )
    parser.add_argument(
        '--label-dir',
        type=str,
        default=RAW_LABEL_DIR,
        help=f'DOTA 标注目录（默认：{RAW_LABEL_DIR}）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_DIR,
        help=f'输出根目录（默认：{OUTPUT_DIR}）'
    )
    parser.add_argument(
        '--crop-size',
        type=int,
        default=CROP_SIZE,
        help=f'切图大小（像素），默认 {CROP_SIZE}'
    )
    parser.add_argument(
        '--crop-stride',
        type=int,
        default=CROP_STRIDE,
        help=f'滑窗步长，默认 {CROP_STRIDE}（50%% 重叠）'
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
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )

    args = parser.parse_args()

    prepare_dota(
        raw_image_dir=args.image_dir,
        raw_label_dir=args.label_dir,
        output_dir=args.output_dir,
        crop_size=args.crop_size,
        crop_stride=args.crop_stride,
        skip_difficult=not args.no_skip_difficult,
        clean_existing=args.clean,
        random_seed=args.seed
    )