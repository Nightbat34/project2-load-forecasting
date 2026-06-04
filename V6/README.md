# Project 2 V6 交付说明

V6 是第二部分模型训练的最终优化版。它保留 V5 的 Stacking 作为正式主模型，并新增 PyTorch + GPU 的 GRU、LSTM、Attention-GRU 时序神经网络探索，用于答辩展示和深度学习方法对比。

## 核心文件

- `project2_forecast_v5_base.py`：传统机器学习与 Stacking 基线脚本。
- `v6_deep_sequence_models.py`：PyTorch GRU/LSTM/Attention-GRU 训练脚本，使用 GPU。
- `v6_generate_final_report.py`：V6 最终汇报生成脚本。
- `report2_generate_model_report.py`：兼容入口，调用 V6 最终汇报生成器。
- `output/project2/report2_model_tuning_optimization.html`：第二部分最终 HTML 汇报。
- `output/project2/part3_integration_notes.md`：第三部分成品展示、网页设计、模型调用对接说明。
- `output/project2/project2_final_prediction_2015_01_11_17.csv`：正式 Stacking 预测结果。
- `output/project2/project2_deep_learning_performance.csv`：GRU/LSTM/Attention-GRU 结果。

## 运行方式

只重新生成最终汇报：

```powershell
cd "F:\Practicum\Data Mining Practicum\V6"
python report2_generate_model_report.py
```

如需重新运行深度学习探索：

```powershell
cd "F:\Practicum\Data Mining Practicum\V6"
$env:KMP_DUPLICATE_LIB_OK="TRUE"
D:\Anaconda\envs\pytorch_gpu\python.exe v6_deep_sequence_models.py
```

如确实需要重新训练完整传统模型：

```powershell
cd "F:\Practicum\Data Mining Practicum\V6"
D:\Anaconda\envs\pytorch_gpu\python.exe project2_forecast_v5_base.py
```

完整传统模型训练会执行 Optuna 搜索和 Stacking 训练，耗时与资源占用明显高于报告生成；V6 当前交付不需要重复执行。
