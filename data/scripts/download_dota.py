"""
DOTA 数据集下载脚本

DOTA v1.0 数据集包含 2806 张遥感图像，15 个类别
官方地址：https://captain-whu.github.io/DOTA/
"""

import os
import requests
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
import hashlib


# DOTA v1.0 数据集 URL (训练集和验证集)
DOTA_V1_URLS = {
    # 训练集图像 (part1, part2)
    'train_images_part1': 'https://capstorage.blob.core.chinacloudapi.cn/dataset/dota/v1/origin/train.zip',
    'train_images_part2': 'https://capstorage.blob.core.chinacloudapi.cn/dataset/dota/v1/origin/val.zip',
    # 验证集图像
    'val_images': 'https://capstorage.blob.core.chinacloudapi.cn/dataset/dota/v1/origin/test.zip',
    # 标注文件 (需要向官方申请)
    # 'train_labels': 'https://capstorage.blob.core.chinacloudapi.cn/dataset/dota/v1/origin/train_labels.zip',
}

# 镜像源 (如果官方源下载失败)
MIRROR_URLS = {
    'train_images': 'https://remap-dataset.oss-cn-beijing.aliyuncs.com/DOTA/train.zip',
    'val_images': 'https://remap-dataset.oss-cn-beijing.aliyuncs.com/DOTA/val.zip',
    'train_labels': 'https://remap-dataset.oss-cn-beijing.aliyuncs.com/DOTA/train_labels.zip',
    'val_labels': 'https://remap-dataset.oss-cn-beijing.aliyuncs.com/DOTA/val_labels.zip',
}


def download_file(
    url: str,
    dest_path: str,
    chunk_size: int = 8192,
    use_mirror: bool = False
) -> bool:
    """
    下载文件

    Args:
        url: 下载 URL
        dest_path: 保存路径
        chunk_size: 分块大小
        use_mirror: 是否使用镜像源

    Returns:
        是否成功
    """
    # 尝试镜像源
    if use_mirror:
        for key, mirror_url in MIRROR_URLS.items():
            if key in url.lower():
                url = mirror_url
                break

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except requests.RequestException as e:
        print(f"下载失败 {url}: {e}")
        return False


def verify_md5(file_path: str, expected_md5: str) -> bool:
    """验证文件 MD5"""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest() == expected_md5


def download_dota(
    dataset_root: str = './data/datasets/DOTA',
    version: str = 'v1.0',
    splits: List[str] = ['train', 'val'],
    use_mirror: bool = True,
    skip_existing: bool = True
) -> None:
    """
    下载 DOTA 数据集

    Args:
        dataset_root: 数据集根目录
        version: 数据集版本 ('v1.0' 或 'v2.0')
        splits: 需要下载的分割 ('train', 'val', 'test')
        use_mirror: 是否使用镜像源
        skip_existing: 跳过已存在的文件
    """
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    download_dir = dataset_root / 'downloads'
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始下载 DOTA {version} 数据集")
    print(f"保存路径：{dataset_root}")
    print(f"使用镜像源：{use_mirror}")
    print("-" * 50)

    # 定义需要下载的文件
    files_to_download = []

    if 'train' in splits:
        files_to_download.extend([
            ('train_images.zip', MIRROR_URLS['train_images'] if use_mirror else DOTA_V1_URLS['train_images_part1']),
            ('train_labels.zip', MIRROR_URLS['train_labels']),
        ])

    if 'val' in splits:
        files_to_download.extend([
            ('val_images.zip', MIRROR_URLS['val_images'] if use_mirror else DOTA_V1_URLS['val_images']),
            ('val_labels.zip', MIRROR_URLS['val_labels']),
        ])

    # 下载文件
    for filename, url in files_to_download:
        dest_path = download_dir / filename

        if skip_existing and dest_path.exists():
            print(f"跳过已存在：{filename}")
            continue

        print(f"\n下载：{filename}")
        success = download_file(url, str(dest_path), use_mirror=use_mirror)

        if success:
            print(f"✓ 下载完成：{filename}")
        else:
            print(f"✗ 下载失败：{filename}")

    print("\n" + "=" * 50)
    print("下载完成！请运行 prepare_dota.py 进行数据预处理")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='下载 DOTA 数据集')
    parser.add_argument(
        '--root',
        type=str,
        default='./data/datasets/DOTA',
        help='数据集根目录'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0',
        help='数据集版本'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        help='需要下载的分割'
    )
    parser.add_argument(
        '--no-mirror',
        action='store_true',
        help='不使用镜像源'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='不跳过已存在的文件'
    )

    args = parser.parse_args()

    download_dota(
        dataset_root=args.root,
        version=args.version,
        splits=args.splits,
        use_mirror=not args.no_mirror,
        skip_existing=not args.no_skip
    )
