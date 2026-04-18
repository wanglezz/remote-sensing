"""
模型导出脚本

PyTorch (.pt) → ONNX (.onnx) → 等待转换为 OM (昇腾模型)
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Optional

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

    Ultralytics 的 export() 会固定将 .onnx 保存到与 .pt 相同目录，
    本函数在导出后将文件移动到用户指定的 output_path。

    Args:
        model_path: PyTorch 模型路径 (.pt)
        output_path: 期望的输出 ONNX 路径
        imgsz: 输入图像尺寸
        batch_size: 批次大小
        simplify: 是否简化模型（由 Ultralytics 内置 onnxslim 完成）
        opset: ONNX opset 版本

    Returns:
        实际保存的 ONNX 文件路径
    """
    print("\n" + "=" * 50)
    print("导出模型到 ONNX")
    print("=" * 50)
    print(f"输入模型：{model_path}")
    print(f"目标路径：{output_path}")
    print(f"图像尺寸：{imgsz}")
    print(f"Batch size: {batch_size}")
    print(f"OPSET: {opset}")
    print("=" * 50 + "\n")

    model = YOLO(model_path)

    export_args = {
        'format':   'onnx',
        'imgsz':    imgsz,
        'batch':    batch_size,
        'opset':    opset,
        'simplify': simplify,   # Ultralytics 内置 onnxslim，无需额外调用
        'dynamic':  False,      # 固定输入尺寸，便于 NPU 部署
        'half':     False,      # FP16 在 ATC 转换时指定
        'device':   'cpu',
        'verbose':  True,
    }

    # Ultralytics 固定保存到 .pt 同目录，返回实际路径
    actual_path = model.export(**export_args)
    actual_path = Path(str(actual_path))

    # 将文件移动到用户指定路径
    output_path = Path(output_path)
    if actual_path.resolve() != output_path.resolve():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(actual_path, output_path)
        print(f"\n✓ 已复制到目标路径：{output_path}")
    else:
        print(f"\n✓ ONNX 已保存：{output_path}")

    return str(output_path)


def verify_onnx(onnx_path: str, test_input: Optional[str] = None) -> bool:
    """
    验证 ONNX 模型结构和推理

    Args:
        onnx_path: ONNX 模型路径
        test_input: 测试图像路径（可选，不填则用随机输入）

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
    inp  = model.graph.input[0]
    outs = [o.name for o in model.graph.output]
    inp_shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  输入  : {inp.name}  shape={inp_shape}")
    print(f"  输出  : {outs}")

    # 推理测试
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    if test_input:
        img = cv2.imread(test_input)
        img = cv2.resize(img, (inp_shape[3], inp_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        dummy = np.expand_dims(img, 0)
        print(f"  使用真实图像：{test_input}")
    else:
        dummy = np.random.randn(*inp_shape).astype(np.float32)
        print(f"  使用随机输入：shape={inp_shape}")

    outputs = sess.run(None, {inp.name: dummy})
    print(f"✓ 推理测试通过，输出形状：{[o.shape for o in outputs]}")

    return True


def main():
    parser = argparse.ArgumentParser(description='导出 YOLO-OBB 模型到 ONNX')
    parser.add_argument('--model',  type=str, required=True,
                        help='PyTorch 模型路径 (.pt)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 ONNX 路径（默认：与 .pt 同目录）')
    parser.add_argument('--imgsz',  type=int, default=1024,
                        help='输入图像尺寸')
    parser.add_argument('--batch',  type=int, default=1,
                        help='批次大小')
    parser.add_argument('--opset',  type=int, default=11,
                        help='ONNX opset 版本')
    parser.add_argument('--no-simplify', action='store_true',
                        help='不使用 onnxslim 简化模型')
    parser.add_argument('--verify', action='store_true',
                        help='导出后验证 ONNX 模型')
    parser.add_argument('--test-image', type=str, default=None,
                        help='验证用的测试图像路径')

    args = parser.parse_args()

    # 默认输出路径：与 .pt 同目录，扩展名改为 .onnx
    if args.output is None:
        args.output = str(Path(args.model).with_suffix('.onnx'))

    # 导出
    final_path = export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        batch_size=args.batch,
        simplify=not args.no_simplify,
        opset=args.opset,
    )

    # 验证
    if args.verify:
        verify_onnx(final_path, args.test_image)

    print("\n" + "=" * 50)
    print("导出完成！")
    print("=" * 50)
    print(f"\n  ONNX 文件：{final_path}")
    print(f"\n下一步：在昇腾服务器上用 ATC 转换为 OM")
    print(f"  atc --model={final_path} \\")
    print(f"      --framework=5 \\")
    print(f"      --output=models/om_models/yolov8n_obb \\")
    print(f"      --input_format=NCHW \\")
    print(f"      --input_shape='images:1,3,{args.imgsz},{args.imgsz}' \\")
    print(f"      --soc_version=Ascend310B \\")
    print(f"      --op_select_implmode=high_performance")


if __name__ == '__main__':
    main()