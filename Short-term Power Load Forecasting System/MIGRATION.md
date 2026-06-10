# 迁移与对接说明

## 最小交付范围

对接人员如果只需要三类文件，请保留：

```text
frontend/
backend/
models/
```

其中：

- `frontend/`：展示网页和静态数据。
- `backend/`：Python 后端服务和特征工程参考代码。
- `models/`：三个 Stacking 模型和模型特征清单。

## 迁移步骤

1. 复制整个 `Short-term Power Load Forecasting System` 文件夹。
2. 在新电脑安装 Python 依赖：

```powershell
cd "Short-term Power Load Forecasting System\backend"
pip install -r requirements.txt
```

3. 回到根目录，双击 `start_windows.bat`。

## 注意事项

- 前端可直接打开，不要求后端运行。
- 后端模型调用需要 Python 环境。
- 模型输入不是原始天气表，而是完成特征工程后的 22 个特征。
- 线上融合时，建议由统一后端负责补齐滞后特征、滚动均值和滚动标准差。

## 模型文件

```text
models/load_mean_Stacking.joblib
models/load_max_Stacking.joblib
models/load_min_Stacking.joblib
models/project2_model_manifest.csv
```

`project2_model_manifest.csv` 中记录了每个目标模型需要的特征列表。
