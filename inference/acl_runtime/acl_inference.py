"""
昇腾 ACL 推理引擎封装

提供基于 Ascend Computing Language 的模型推理接口
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


# 尝试导入 ACL，如果不在昇腾环境则使用 Mock
try:
    import ascendacl as acl
    ACL_AVAILABLE = True
except ImportError:
    ACL_AVAILABLE = False
    print("警告：ascendacl 未安装，将在模拟模式下运行")
    print("请在昇腾 NPU 环境中安装 CANN Toolkit")


@dataclass
class InferenceResult:
    """推理结果"""
    boxes: np.ndarray      # [N, 5] - [cx, cy, w, h, angle]
    scores: np.ndarray     # [N] - 置信度
    class_ids: np.ndarray  # [N] - 类别 ID


class ACLInference:
    """
    ACL 推理引擎封装

    支持 YOLO-OBB 模型的推理和后处理
    """

    def __init__(
        self,
        om_path: str,
        device_id: int = 0,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        max_det: int = 300
    ):
        """
        初始化推理引擎

        Args:
            om_path: OM 模型路径
            device_id: NPU 设备 ID
            conf_thres: 置信度阈值
            iou_thres: NMS IoU 阈值
            max_det: 最大检测框数
        """
        self.om_path = str(om_path)
        self.device_id = device_id
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        # ACL 资源
        self.context = None
        self.stream = None
        self.model = None
        self.input_desc = None
        self.output_desc = None

        # 输入输出缓冲区
        self.input_buffer = None
        self.output_buffers = []

        if ACL_AVAILABLE:
            self._init_acl()

    def _init_acl(self) -> None:
        """初始化 ACL 资源"""
        print(f"初始化 ACL 设备 {self.device_id}...")

        # 初始化 ACL
        ret = acl.init()
        if ret != 0:
            raise RuntimeError(f"ACL 初始化失败：{ret}")

        # 打开设备
        ret = acl.rt.set_device(self.device_id)
        if ret != 0:
            raise RuntimeError(f"ACL 打开设备失败：{ret}")

        # 创建上下文
        ret, self.context = acl.rt.create_context(self.device_id)
        if ret != 0:
            raise RuntimeError(f"ACL 创建上下文失败：{ret}")

        # 创建流
        ret, self.stream = acl.rt.create_stream()
        if ret != 0:
            raise RuntimeError(f"ACL 创建流失败：{ret}")

        # 加载模型
        ret, self.model = acl.mdl.load_from_file(self.om_path)
        if ret != 0:
            raise RuntimeError(f"ACL 加载模型失败：{ret}")

        print(f"✓ 模型加载成功：{self.om_path}")

        # 获取模型输入输出描述
        self._get_model_desc()

        # 分配输入输出缓冲区
        self._create_buffers()

    def _get_model_desc(self) -> None:
        """获取模型输入输出描述"""
        # 输入描述
        self.input_desc = acl.mdl.create_dataset(self.model)
        # 输出描述
        self.output_desc = acl.mdl.create_dataset(self.model)

    def _create_buffers(self) -> None:
        """分配输入输出缓冲区"""
        # 这里需要根据实际模型的输入输出尺寸分配
        # YOLO-OBB 输入：[1, 3, 1024, 1024]
        # 输出：取决于模型结构，通常是多尺度特征图
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理

        Args:
            image: BGR 图像 [H, W, 3]

        Returns:
            预处理后的图像 [1, 3, H, W]
        """
        # Resize
        img = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        # Normalize
        img = img.astype(np.float32) / 255.0

        # Add batch dimension
        img = np.expand_dims(img, 0)

        return img

    def inference(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        执行推理

        Args:
            input_data: 预处理后的输入 [1, 3, H, W]

        Returns:
            模型输出列表
        """
        if not ACL_AVAILABLE:
            # Mock 输出
            return [np.random.randn(1, 84, 262144).astype(np.float32)]

        # 复制输入数据到设备
        # 执行推理
        # 获取输出数据

        # 这里是简化版本，实际需要完整的 ACL 调用流程
        ret = acl.mdl.execute(
            self.model,
            self.input_desc,
            self.output_desc
        )

        if ret != 0:
            raise RuntimeError(f"ACL 推理失败：{ret}")

        # 同步流
        acl.rt.synchronize_stream(self.stream)

        # 获取输出
        outputs = []
        for i in range(acl.mdl.get_num_outputs(self.model)):
            output_data = acl.mdl.get_output_ptr(self.output_desc, i)
            outputs.append(output_data)

        return outputs

    def __call__(self, image: np.ndarray) -> InferenceResult:
        """
        端到端推理

        Args:
            image: BGR 图像

        Returns:
            推理结果
        """
        # 预处理
        input_data = self.preprocess(image)

        # 推理
        outputs = self.inference(input_data)

        # 后处理 (解码 + NMS)
        boxes, scores, class_ids = self.postprocess(outputs)

        return InferenceResult(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids
        )

    def postprocess(
        self,
        outputs: List[np.ndarray],
        img_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        后处理：解码 + NMS

        Args:
            outputs: 模型输出
            img_shape: 原始图像尺寸 (H, W)

        Returns:
            (boxes, scores, class_ids)
        """
        # 导入后处理工具
        from utils.obb_utils import rotate_nms, xywhr2xyxyxyxy

        # YOLO-OBB 输出格式：[batch, 84, num_anchors]
        # 84 = 4 (box) + 1 (angle) + 15 (cls) + 1 (obj)

        output = outputs[0][0]  # [84, num_anchors]

        # 转置
        output = output.T  # [num_anchors, 84]

        # 解码 boxes 和 scores
        boxes = output[:, :5]  # [cx, cy, w, h, angle]
        cls_scores = output[:, 5:]  # [num_classes]

        # 计算置信度
        scores = np.max(cls_scores, axis=1)
        class_ids = np.argmax(cls_scores, axis=1)

        # 过滤低置信度
        mask = scores >= self.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # NMS
        if len(boxes) > 0:
            keep = rotate_nms(boxes, scores, self.iou_thres)
            boxes = boxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

            # 限制最大检测数
            if len(boxes) > self.max_det:
                idx = np.argsort(-scores)[:self.max_det]
                boxes = boxes[idx]
                scores = scores[idx]
                class_ids = class_ids[idx]

        return boxes, scores, class_ids

    def release(self) -> None:
        """释放 ACL 资源"""
        if not ACL_AVAILABLE:
            return

        if self.model is not None:
            acl.mdl.unload(self.model)
        if self.stream is not None:
            acl.rt.destroy_stream(self.stream)
        if self.context is not None:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()

        print("✓ ACL 资源已释放")

    def __del__(self):
        self.release()


# 如果没有 ACL，提供 Mock 实现
class MockACLInference:
    """Mock ACL 推理 (用于开发测试)"""

    def __init__(self, **kwargs):
        self.conf_thres = kwargs.get('conf_thres', 0.25)
        self.iou_thres = kwargs.get('iou_thres', 0.7)

    def __call__(self, image: np.ndarray):
        # 返回随机检测结果
        num_det = np.random.randint(0, 10)
        return InferenceResult(
            boxes=np.random.randn(num_det, 5).astype(np.float32),
            scores=np.random.rand(num_det).astype(np.float32),
            class_ids=np.random.randint(0, 15, num_det)
        )

    def release(self):
        pass


# 导出
if ACL_AVAILABLE:
    AclInference = ACLInference
else:
    AclInference = MockACLInference
