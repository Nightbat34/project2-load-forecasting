# 电力系统短期负荷预测系统

这是项目二的正式成品交付包，按对接要求拆分为三类文件：

```text
Short-term Power Load Forecasting System/
  frontend/   前端展示网页
  backend/    后端模型调用服务
  models/     训练完成的 Stacking 模型与结果文件
  docs/       过拟合分析、答辩说明等文档
```

## 1. 快速启动

Windows 双击：

```text
start_windows.bat
```

脚本会做两件事：

1. 启动后端 API：`http://127.0.0.1:5000`
2. 打开前端页面：`frontend/index.html`

如果只是查看展示网页，也可以直接打开：

```text
frontend/index.html
```

## 2. 设备与环境条件

推荐环境：

- Windows 10/11
- Python 3.10 或以上
- 已安装依赖：`flask`、`pandas`、`numpy`、`scikit-learn`、`joblib`、`xgboost`
- 本项目原训练环境：`D:\Anaconda\envs\pytorch_gpu`
- GPU 不是运行成品网页的必须条件；GPU 只用于训练阶段的 XGBoost / PyTorch 探索。

安装后端依赖：

```powershell
cd "Short-term Power Load Forecasting System\backend"
pip install -r requirements.txt
```

## 3. 前端说明

入口：

```text
frontend/index.html
```

前端是纯静态页面，不依赖外网 CDN。主要展示：

- 2014 年负荷持续曲线
- 2015-01-11 至 2015-01-17 预测结果
- 气象因素与负荷关系
- 验证期真实值与预测值
- Stacking 模型效果
- 过拟合分析入口

前端数据文件：

```text
frontend/assets/data/dashboard_data.js
frontend/assets/data/dashboard_data.json
```

## 4. 后端说明

后端入口：

```text
backend/app.py
```

启动：

```powershell
cd "Short-term Power Load Forecasting System"
python backend/app.py
```

接口：

```text
GET  /api/health
GET  /api/final-forecast
GET  /api/model-manifest
POST /api/predict
```

注意：`POST /api/predict` 接收的是已经完成特征工程的特征行，不是单纯未来天气。因为 Stacking 模型需要滞后负荷、滚动均值、滚动标准差等时序特征。

## 5. 模型说明

模型文件：

```text
models/load_mean_Stacking.joblib
models/load_max_Stacking.joblib
models/load_min_Stacking.joblib
```

三个模型分别预测：

- 日平均负荷
- 日最高负荷
- 日最低负荷

模型结构：

- 基学习器：岭回归 Ridge、随机森林 RandomForest、极端梯度提升 XGBoost、支持向量回归 SVR、K近邻回归 KNN
- 元学习器：岭回归 Ridge(alpha=10.0)

验证集平均 R2：`0.8546`

## 6. 迁移说明

整个文件夹可以直接复制到其他电脑。迁移后需要注意：

1. 保持目录结构不变。
2. 如果 Python 环境不是 `D:\Anaconda\envs\pytorch_gpu`，`start_windows.bat` 会自动尝试使用系统 `python`。
3. 如果后端启动失败，先进入 `backend` 安装 `requirements.txt`。
4. 前端展示不依赖后端；后端用于后续模型调用和系统融合。

## 7. 对接建议

如果其他组员已有统一前端，可以只接入：

```text
frontend/assets/data/dashboard_data.json
backend/app.py
models/
```

如果其他组员已有统一后端，可以直接复用：

```text
models/*.joblib
models/project2_model_manifest.csv
backend/feature_engineering_reference.py
```

推荐对接流程：

```text
前端输入或选择预测条件
        ↓
后端补齐滞后负荷与滚动特征
        ↓
加载 models/*.joblib 预测三个目标
        ↓
执行物理约束纠偏
        ↓
返回日最低 / 日平均 / 日最高负荷
```

物理约束：

```text
日最低负荷 <= 日平均负荷 <= 日最高负荷
所有预测负荷 > 0
```
