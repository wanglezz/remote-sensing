## 性能优化

### 1. 算子融合

在 ATC 转换时使用 `high_performance` 模式：
```bash
--op_select_implmode=high_performance
```

### 2. Batch 推理

对于高吞吐场景，可以调整 batch size：
```bash
--input_shape="images:4,3,1024,1024"
```

### 3. 多设备并行

多个 NPU 设备可以并行推理：
```python
engines = []
for device_id in range(4):
    engine = AclInference(om_path=om_path, device_id=device_id)
    engines.append(engine)
```

### 4. 量化技术

量化技术是提高模型推理速度和降低内存占用的有效方法。通过将浮点数权重和激活值转换为整数（通常是8位整数），可以在不显著影响精度的情况下，大幅提升模型的推理效率。

#### 量化流程
1. **训练后量化 (Post-Training Quantization)**:
   - 在不需要重新训练的情况下，直接对预训练模型进行量化。
   - 使用昇腾 ATC 工具进行量化：
     ```bash
     atc \
         --model=models/exported/best.onnx \
         --output=models/om_models/yolov8n_obb_quant \
         --soc_version=Ascend310B \
         --input_shape="images:1,3,1024,1024" \
         --insert_op_conf=./configs/quant.cfg \
         --op_select_implmode=high_performance
     ```
   - 配置文件 `quant.cfg` 用于指定量化参数，例如：
     ```ini
     [quant]
     input_format=NCHW
     output_format=INT8
     calibration_data=./data/calibration_data
     calibration_iterations=100
     ```

2. **量化感知训练 (Quantization-Aware Training)**:
   - 在训练过程中就引入量化操作，使模型在训练时就适应量化带来的影响。
   - 修改训练脚本，添加量化感知层，并在训练过程中使用量化模拟器。
   - 示例代码（假设使用 PyTorch）：
     ```python
     import torch
     from torch.quantization import QuantStub, DeQuantStub

     class QuantizedModel(torch.nn.Module):
         def __init__(self, model):
             super(QuantizedModel, self).__init__()
             self.model = model
             self.quant = QuantStub()
             self.dequant = DeQuantStub()

         def forward(self, x):
             x = self.quant(x)
             x = self.model(x)
             x = self.dequant(x)
             return x

     # 原始模型
     model = YOLOv8nOBB()
     # 量化感知模型
     quant_model = QuantizedModel(model)

     # 训练过程
     quant_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
     quant_model = torch.quantization.prepare_qat(quant_model)
     quant_model.train()
     for epoch in range(num_epochs):
         for data, target in train_loader:
             output = quant_model(data)
             loss = criterion(output, target)
             loss.backward()
             optimizer.step()

     # 转换为量化模型
     quant_model = torch.quantization.convert(quant_model)
     ```

3. **校准 (Calibration)**:
   - 校准是为了确定量化的范围，确保量化后的数据分布与原始数据分布一致。
   - 使用校准数据集进行校准，生成量化配置文件：
     ```bash
     python -m torch.quantization.calibrate \
         --data-path ./data/calibration_data \
         --output-path ./configs/quant.cfg \
         --num-calibration-iterations 100
     ```

#### 量化注意事项
- **精度损失**：量化可能会导致一定的精度损失，需要权衡精度和性能。
- **校准数据**：选择合适的校准数据集非常重要，应尽量覆盖各种输入情况。
- **测试验证**：在量化完成后，务必进行充分的测试验证，确保模型性能和精度符合预期。
