# Project 2 V5 交付说明

V5 是在 `model2` 和 `v4` 结果基础上的第二次汇报增强版。它没有重复重训大模型，而是复用已经完成的 Optuna、Stacking、统计检验和残差诊断结果，按第一次数据处理汇报的风格生成“训练过程 + 结果展示”报告。

## 核心文件

- `project2_forecast_v5_base.py`：从 v4 复制的完整训练主脚本，可作为 V5 基线代码。
- `v5_pre_analysis_base.py`：从 v4 复制的七步前置分析脚本。
- `report2_generate_model_report.py`：第二次汇报生成脚本，只读取 CSV/PNG 并生成报告。
- `output/project2/report2_model_tuning_optimization.html`：第二次汇报 HTML 展示版。
- `output/project2/report2_model_tuning_optimization.md`：第二次汇报 Markdown 版。
- `output/project2/*.csv`、`*.png`、`models/*.joblib`：从 v4 复制的结果表、图和最终模型。

## 运行方式

只重新生成报告：

```powershell
cd "F:\Practicum\Data Mining Practicum\V5"
python report2_generate_model_report.py
```

如确实需要重新训练完整模型，再运行：

```powershell
cd "F:\Practicum\Data Mining Practicum\V5"
D:\Anaconda\envs\pytorch_gpu\python.exe project2_forecast_v5_base.py
```

完整训练会执行 Optuna 搜索和 Stacking 训练，耗时与资源占用明显高于报告生成；V5 当前交付不需要重复执行。
