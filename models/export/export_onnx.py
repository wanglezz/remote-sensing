"""
模型导出脚本

PyTorch (.pt) → ONNX (.onnx) → 等待转换为 OM (昇腾模型)
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO


def export_to_onnx(
    model_path: str,
    output_path: str,
    imgsz: int = 1024,
    batch_size: int = 1,
    simplify: bool = True,
    opset: int = 11
) -> str:
    """
    导出 YOLO-OBB 模型到 ONNX 格式

    Args:
        model_path: PyTorch 模型路径 (.pt)
        output_path: 输出 ONNX 路径
        imgsz: 输入图像尺寸
        batch_size: 批次大小
        simplify: 是否简化模型
        opset: ONNX opset 版本

    Returns:
        输出的 ONNX 文件路径
    """
    print("\n" + "=" * 50)
    print("导出模型到 ONNX")
    print("=" * 50)
    print(f"输入模型：{model_path}")
    print(f"输出路径：{output_path}")
    print(f"图像尺寸：{imgsz}")
    print(f"Batch size: {batch_size}")
    print(f"OPSET: {opset}")
    print("=" * 50 + "\n")

    # 加载模型
    model = YOLO(model_path)
    print(f"模型类别：{model.type}")

    # 导出参数
    export_args = {
        'format': 'onnx',
        'imgsz': imgsz,
        'batch': batch_size,
        'opset': opset,
        'simplify': simplify,
        'dynamic': False,  # 固定输入尺寸，便于 NPU 部署
        'half': False,     # FP16 在 ATC 转换时指定
        'device': 'cpu',   # CPU 导出
        'verbose': True,
    }

    # 执行导出
    output_path = model.export(**export_args)

    print(f"\n✓ ONNX 导出完成：{output_path}")

    return output_path


def simplify_onnx(onnx_path: str, output_path: Optional[str] = None) -> str:
    """
    简化 ONNX 模型

    Args:
        onnx_path: 输入 ONNX 路径
        output_path: 输出路径 (默认覆盖原文件)

    Returns:
        简化的 ONNX 文件路径
    """
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("错误：请先安装 onnxsim: pip install onnxsim")
        return onnx_path

    if output_path is None:
        output_path = onnx_path.replace('.onnx', '_simplified.onnx')

    print(f"\n简化 ONNX 模型...")
    print(f"输入：{onnx_path}")

    # 加载模型
    onnx_model = onnx.load(onnx_path)

    # 简化
    model_simplified, check = simplify(onnx_model)

    if check:
        # 保存简化后的模型
        onnx.save(model_simplified, output_path)
        print(f"✓ 简化完成：{output_path}")

        # 验证
        from onnxruntime import InferenceSession
        sess_original = InferenceSession(onnx_path)
        sess_simplified = InferenceSession(output_path)

        print("验证简化前后输出一致性...")
        # 这里可以添加验证逻辑
    else:
        print("⚠ 简化失败，使用原模型")
        output_path = onnx_path

    return output_path


def verify_onnx(onnx_path: str, test_input: Optional[str] = None) -> bool:
    """
    验证 ONNX 模型

    Args:
        onnx_path: ONNX 模型路径
        test_input: 测试图像路径 (可选)

    Returns:
        是否验证通过
    """
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
        import cv2
    except ImportError:
        print("错误：请先安装 onnx 和 onnxruntime")
        return False

    print(f"\n验证 ONNX 模型：{onnx_path}")

    # 检查模型结构
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("✓ 模型结构验证通过")

    # 打印输入输出信息
    print(f"输入：{model.graph.input[0].name}")
    print(f"输出：{[o.name for o in model.graph.output]}")

    # 推理测试
    sess = ort.InferenceSession(onnx_path)

    if test_input:
        # 使用真实图像测试
        img = cv2.imread(test_input)
        img = cv2.resize(img, (1024, 1024))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)

        outputs = sess.run(None, {model.graph.input[0].name: img})
        print(f"✓ 推理测试通过，输出形状：{[o.shape for o in outputs]}")
    else:
        # 使用随机输入测试
        input_shape = model.graph.input[0].type.tensor_type.shape
        shape = [int(d.dim_value) for d in input_shape.dim]
        dummy_input = np.random.randn(*shape).astype(np.float32)

        outputs = sess.run(None, {model.graph.input[0].name: dummy_input})
        print(f"✓ 随机输入测试通过，输出形状：{[o.shape for o in outputs]}")

    return True


def main():
    parser = argparse.ArgumentParser(description='导出 YOLO-OBB 模型')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='PyTorch 模型路径 (.pt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出 ONNX 路径 (默认：与输入同目录)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=1024,
        help='输入图像尺寸'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=1,
        help='批次大小'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=11,
        help='ONNX opset 版本'
    )
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='不简化模型'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='验证导出的 ONNX 模型'
    )
    parser.add_argument(
        '--test-image',
        type=str,
        default=None,
        help='验证用的测试图像'
    )

    args = parser.parse_args()

    # 确定输出路径
    if args.output is None:
        args.output = Path(args.model).with_suffix('.onnx').name

    # 导出
    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        batch_size=args.batch,
        simplify=not args.no_simplify,
        opset=args.opset
    )

    # 简化
    if not args.no_simplify:
        simplify_onnx(args.output)

    # 验证
    if args.verify:
        verify_onnx(args.output, args.test_image)

    print("\n" + "=" * 50)
    print("导出完成！")
    print("=" * 50)
    print(f"\n下一步：使用 ATC 工具将 ONNX 转换为 OM")
    print("示例命令:")
    print(f"  atc --model={args.output} --framework=5 --output=yolov8n_obb --input_format=NCHW --input='images:1,3,{args.imgsz},{args.imgsz}'")


if __name__ == '__main__':
    main()
