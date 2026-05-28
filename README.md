# 项目二：电力系统短期负荷预测

本项目围绕“电力系统短期负荷预测”展开：使用 2012-01-01 至 2014-12-31 的负荷数据和气象因素数据，分析日最高负荷、日最低负荷、日平均负荷与气象因素之间的关系，并预测 2015-01-11 至 2015-01-17 共 7 天的负荷。

仓库地址：

[Nightbat34/project2-load-forecasting](https://github.com/Nightbat34/project2-load-forecasting)

## 一、项目目标

1. 分析 2014 年全年负荷数据，统计日最高、日最低、日平均负荷，并绘制负荷持续曲线。
2. 根据 2012-01-01 至 2014-12-31 的负荷与气象因素数据，建立预测模型。
3. 预测 2015-01-11 至 2015-01-17 的日最高、日最低、日平均负荷。
4. 在报告中体现数据处理、特征工程、模型训练、模型评估、模型选择、统计检验和最终预测全过程。

## 二、目录结构

```text
process/
  project2_forecast.py                  # 主程序：数据处理、训练、预测、生成报告
  requirements.txt                      # Python 依赖
  README.md                             # 项目说明、运行方法、汇报方法
  output/
    project2/
      project2_report.html              # 网页汇报，重点查看
      project2_final_prediction_2015_01_11_17.csv
      project2_model_performance.csv
      project2_training_log.csv
      project2_model_selection.csv
      project2_statistical_tests.csv
      project2_model_manifest.csv
      project2_model_manifest.json
      project2_regression_summary.csv
      project2_weather_correlations.csv
      project2_validation_predictions.csv
      models/
        load_mean_RandomForest.joblib
        load_max_Stacking.joblib
        load_min_RandomForest.joblib
      01_weather_regression.png
      02_model_training_comparison.png
      03_validation_fit.png
      04_final_prediction.png
      05_history_forecast_context.png
```

## 三、环境准备

建议使用 Python 3.10 或以上版本。

安装依赖：

```powershell
pip install -r requirements.txt
```

依赖说明：

```text
pandas        表格数据读取、清洗、聚合
numpy         数值计算
matplotlib    绘图
seaborn       统计图表
statsmodels   线性回归与统计建模
scikit-learn  机器学习模型与评估
xgboost       XGBoost 梯度提升模型
openpyxl      读取 Excel 文件
scipy         T 检验等统计检验
joblib        保存训练后的模型文件
```

## 四、数据文件位置

脚本默认读取：

```text
../Data/附件1-电网负荷数据.xlsx
```

也就是说，推荐目录结构是：

```text
Data Mining Practicum/
  Data/
    附件1-电网负荷数据.xlsx
  process/
    project2_forecast.py
```

如果数据文件放在其他位置，可以运行前设置环境变量：

```powershell
$env:LOAD_DATA_PATH="F:\Practicum\Data Mining Practicum\Data\附件1-电网负荷数据.xlsx"
python project2_forecast.py
```

## 五、怎么运行

在 `process` 文件夹中运行：

```powershell
python project2_forecast.py
```

默认情况下，脚本会让 XGBoost 使用 GPU：

```text
PROJECT2_XGB_DEVICE=cuda
```

如果某台电脑没有可用 NVIDIA GPU，或者 CUDA 环境不稳定，可以强制使用 CPU：

```powershell
$env:PROJECT2_XGB_DEVICE="cpu"
python project2_forecast.py
```

运行完成后会生成：

```text
output/project2/
```

重点查看：

```text
output/project2/project2_report.html
```

这个 HTML 是最终网页汇报文件，包含过程解释、公式推导、模型训练过程、模型选择、统计检验和最终预测结果。

## 六、怎么演示

建议演示顺序如下，控制在 5-8 分钟比较合适。

### 1. 先说明业务背景

可以这样讲：

> 本项目要解决的是电力系统短期负荷预测问题。短期负荷预测对机组调度、经济运行和安全校核很重要。题目给出了 2009-2015 年的负荷数据，以及 2012-2015 年的气象数据。我们重点负责项目二，也就是分析气象因素与负荷的关系，并预测 2015-01-11 至 2015-01-17 的负荷。

### 2. 再说明数据处理

打开网页报告第 1 节。

讲法：

> 原始负荷数据是宽表，每天 96 个 15 分钟采样点。我先把一天的 96 个点聚合为三个日指标：日最高负荷、日最低负荷、日平均负荷。2015-01-11 至 2015-01-17 的负荷数据全部缺失，正好是预测目标；但气象数据完整，可以作为预测输入。

核心公式：

```text
日最高负荷 = max(一天96个采样点)
日最低负荷 = min(一天96个采样点)
日平均负荷 = 96个采样点的平均值
```

### 3. 讲特征工程

打开网页报告第 2 节。

讲法：

> 负荷不仅受气象影响，还受星期周期和历史负荷惯性影响，所以我构造了三类特征：气象特征、周期特征和滞后特征。

重点解释：

```text
temp_range：温差
HDD：供热度日，低温越明显，该值越大
CDD：供冷度日，高温越明显，该值越大
lag_1：昨天负荷
lag_7：上周同日负荷
roll_mean_7：过去7天平均负荷
```

### 4. 讲 GLM 和气象回归

打开网页报告第 3 节。

讲法：

> 广义线性模型 GLM 的核心思想是用联系函数把响应变量的期望和线性预测器连接起来。本题预测的是连续负荷值，因此可以近似采用高斯分布和恒等联系函数，退化为线性回归/Ridge 这类模型。

专业词汇备注：

```text
GLM：Generalized Linear Model，广义线性模型
Gaussian：高斯分布/正态分布
identity link：恒等联系函数
OLS：Ordinary Least Squares，普通最小二乘
Ridge：岭回归，在线性回归基础上加入 L2 正则化
```

### 5. 讲模型训练过程

打开网页报告第 4 节和图 `02_model_training_comparison.png`。

讲法：

> 我没有只训练一个模型，而是训练了 Ridge、RandomForest、XGBoost、Stacking 四类候选模型。每个目标变量，即日平均、日最高、日最低，分别独立建模。模型评估不仅用了留出法，还用了时间序列交叉验证和自助法袋外误差估计。

对应方法：

```text
Hold-out：留出法，固定训练集和验证集
TimeSeriesSplit：时间序列交叉验证，保证训练数据早于验证数据
Bootstrap：自助法，有放回抽样，用袋外样本估计泛化误差
```

### 6. 讲模型选择和模型文件

打开网页报告第 5 节。

讲法：

> 最终模型选择规则是：优先看验证集 RMSE，越小越好；如果结果接近，再参考交叉验证 RMSE 的均值和稳定性。最终入选模型保存为 joblib 文件，方便后续复用。

最终模型：

```text
日平均负荷：RandomForest
日最高负荷：Stacking
日最低负荷：RandomForest
```

模型文件：

```text
output/project2/models/load_mean_RandomForest.joblib
output/project2/models/load_max_Stacking.joblib
output/project2/models/load_min_RandomForest.joblib
```

### 7. 讲统计检验

打开网页报告第 5 节的统计检验表。

讲法：

> 为了判断模型差异是不是偶然造成的，我加入了配对样本 T 检验。它比较同一验证日期上两个模型的误差差异。McNemar 检验本来用于分类任务，这里我把误差是否小于实际值 5% 转化为正确/错误，再比较两个模型的犯错模式。

### 8. 讲最终预测结果

打开网页报告第 7 节和图 `04_final_prediction.png`。

讲法：

> 最终预测采用递推方式。2015-01-11 使用 2015-01-10 及以前的真实负荷；从 2015-01-12 开始，前一天的预测值会作为 lag_1 输入下一天模型，避免使用目标期缺失的真实负荷。

最终预测 CSV：

```text
output/project2/project2_final_prediction_2015_01_11_17.csv
```

## 七、核心代码怎么读

主代码文件：

```text
project2_forecast.py
```

建议按下面顺序读。

### 1. 数据读取与日指标构造

```python
load_daily_data()
```

作用：

```text
读取 Excel 的 Area_Load 和 Area_Weather 两个 Sheet。
将每天 96 个负荷点聚合成日最高、日最低、日平均负荷。
合并气象数据。
```

### 2. 特征工程

```python
add_features()
make_target_features()
```

作用：

```text
add_features：构造气象衍生特征和周期特征。
make_target_features：为每个预测目标构造 lag_1、lag_7、lag_14、滚动均值和滚动标准差。
```

### 3. 气象回归分析

```python
build_weather_regression()
```

作用：

```text
计算 Pearson 相关系数。
建立 GLM/OLS 回归模型，输出系数、p 值、R²。
```

### 4. 候选模型定义

```python
candidate_models()
```

包含：

```text
Ridge：线性基线模型，可解释性强
RandomForest：Bagging 思想，降低方差
XGBoost：Boosting 思想，逐步修正残差；本项目默认 device="cuda"，可使用 GPU 加速
Stacking：多模型融合，用元学习器整合多个模型输出
```

### 5. 模型训练与评估

```python
fit_and_evaluate()
```

作用：

```text
对日平均、日最高、日最低三个目标分别训练模型。
计算留出法指标、时间序列交叉验证指标、自助法袋外误差。
生成训练日志、性能对比表、模型选择表。
保存最终模型文件。
```

### 6. 显著性检验

```python
statistical_tests()
```

作用：

```text
比较最优模型和次优模型。
使用 paired T-test 判断误差差异是否显著。
使用 McNemar test 分析预测正确/错误模式差异。
```

### 7. 最终递推预测

```python
forecast_recursive()
```

作用：

```text
逐日预测 2015-01-11 到 2015-01-17。
后一天会使用前一天预测值构造 lag_1。
保证不会把预测目标期的缺失真实负荷泄漏进模型。
```

### 8. 生成图表和网页报告

```python
save_figures()
generate_report()
save_outputs()
```

作用：

```text
保存图表、CSV、模型清单和网页报告。
```

## 八、输出文件怎么解释

```text
project2_report.html
```

最终网页汇报，答辩时优先打开。

```text
project2_final_prediction_2015_01_11_17.csv
```

最终 7 天预测表。

```text
project2_model_performance.csv
```

模型性能对比，包括 RMSE、MAE、MAPE、R²、交叉验证、自助法结果。

```text
project2_training_log.csv
```

训练过程记录，包括模型函数名、中文解释、参数、训练耗时、交叉验证每折结果。

```text
project2_model_selection.csv
```

最终模型选择结果。

```text
project2_statistical_tests.csv
```

配对 T 检验和 McNemar 检验结果。

```text
project2_model_manifest.csv/json
```

最终模型清单，包括模型文件路径和特征数量。

```text
models/*.joblib
```

训练好的最终模型文件，可复用。

## 九、GitHub 版本管理方法

查看当前改动：

```powershell
git status
```

提交一版：

```powershell
git add .
git commit -m "Update project2 report and documentation"
```

上传到 GitHub：

```powershell
git push
```

如果命令行 push 网络不稳定，可以使用 GitHub Desktop，点击 `Push origin`。

## 十、汇报时的简短总结

可以用这段作为结尾：

> 本项目从原始 15 分钟负荷数据出发，先聚合为日最高、日最低、日平均三个预测目标，再结合气象因素、周期特征和历史滞后负荷构造特征。模型训练阶段对 Ridge、RandomForest、XGBoost 和 Stacking 进行对比，并使用留出法、时间序列交叉验证、自助法和统计检验综合评估。最终选择 RandomForest 和 Stacking 作为不同目标的最优模型，并保存模型文件，完成了 2015-01-11 至 2015-01-17 的短期负荷预测。
