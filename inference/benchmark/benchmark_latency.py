"""
性能基准测试脚本

测试端到端推理时延
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import yaml

from inference.acl_runtime.acl_inference import AclInference


def create_test_image(size: int = 1024) -> np.ndarray:
    """创建测试图像"""
    # 生成随机噪声图像，模拟遥感图像
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return img


def benchmark_latency(
    config_path: str = './configs/npu_inference_config.yaml',
    num_warmup: int = 10,
    num_iters: int = 100,
    image_size: int = 1024
) -> Dict[str, float]:
    """
    基准测试推理时延

    Args:
        config_path: 配置文件路径
        num_warmup: 预热次数
        num_iters: 测试迭代次数
        image_size: 图像尺寸

    Returns:
        时延统计结果
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 初始化推理引擎
    om_path = config.get('model', {}).get('om_path', '')

    if Path(om_path).exists():
        print(f"加载模型：{om_path}")
        engine = AclInference(
            om_path=om_path,
            device_id=config.get('npu', {}).get('device_id', 0)
        )
    else:
        print(f"模型文件不存在，使用 Mock 引擎：{om_path}")
        engine = AclInference()

    # 创建测试图像
    test_image = create_test_image(image_size)
    print(f"\n测试图像尺寸：{image_size}x{image_size}")
    print(f"预热迭代：{num_warmup}")
    print(f"测试迭代：{num_iters}")
    print("-" * 50)

    latencies: List[float] = []

    # 预热
    print("预热中...")
    for i in range(num_warmup):
        _ = engine(test_image)

    # 正式测试
    print("开始测试...")
    for i in range(num_iters):
        start = time.perf_counter()
        _ = engine(test_image)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        if (i + 1) % 10 == 0:
            print(f"  进度：{i + 1}/{num_iters}")

    # 统计结果
    latencies = np.array(latencies)

    results = {
        'mean': float(np.mean(latencies)),
        'std': float(np.std(latencies)),
        'min': float(np.min(latencies)),
        'max': float(np.max(latencies)),
        'p50': float(np.percentile(latencies, 50)),
        'p90': float(np.percentile(latencies, 90)),
        'p95': float(np.percentile(latencies, 95)),
        'p99': float(np.percentile(latencies, 99)),
    }

    # 打印结果
    print("\n" + "=" * 50)
    print("时延统计结果 (毫秒)")
    print("=" * 50)
    print(f"平均值： {results['mean']:.2f} ± {results['std']:.2f} ms")
    print(f"最小值： {results['min']:.2f} ms")
    print(f"最大值： {results['max']:.2f} ms")
    print(f"P50:    {results['p50']:.2f} ms")
    print(f"P90:    {results['p90']:.2f} ms")
    print(f"P95:    {results['p95']:.2f} ms")
    print(f"P99:    {results['p99']:.2f} ms")
    print("=" * 50)

    # 检查是否满足目标
    target_latency = config.get('performance', {}).get('target_latency_ms', 20)
    if results['p95'] <= target_latency:
        print(f"✓ 满足目标：P95 时延 {results['p95']:.2f}ms ≤ {target_latency}ms")
    else:
        print(f"✗ 未满足目标：P95 时延 {results['p95']:.2f}ms > {target_latency}ms")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='推理性能基准测试')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/npu_inference_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='预热迭代次数'
    )
    parser.add_argument(
        '--iters',
        type=int,
        default=100,
        help='测试迭代次数'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=1024,
        help='测试图像尺寸'
    )

    args = parser.parse_args()

    benchmark_latency(
        config_path=args.config,
        num_warmup=args.warmup,
        num_iters=args.iters,
        image_size=args.imgsz
    )
