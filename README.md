# 项目二：电力系统短期负荷预测

本仓库只保留项目二内容：根据 2012-01-01 至 2014-12-31 的电力负荷与气象数据，分析负荷与气象因素的关系，并预测 2015-01-11 至 2015-01-17 共 7 天的日最高、日最低、日平均负荷。

## 目录结构

```text
process/
  project2_forecast.py
  requirements.txt
  output/
    project2/
      project2_report.html
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
        load_mean_*.joblib
        load_max_*.joblib
        load_min_*.joblib
      01_weather_regression.png
      02_model_training_comparison.png
      03_validation_fit.png
      04_final_prediction.png
      05_history_forecast_context.png
```

## 数据文件

脚本默认读取：

```text
../Data/附件1-电网负荷数据.xlsx
```

如果你的数据文件在其他位置，可以在运行前设置环境变量：

```powershell
$env:LOAD_DATA_PATH="F:\Practicum\Data Mining Practicum\Data\附件1-电网负荷数据.xlsx"
python project2_forecast.py
```

## 运行方法

```powershell
pip install -r requirements.txt
python project2_forecast.py
```

运行后结果会生成在：

```text
output/project2/
```

最重要的两个文件是：

- `output/project2/project2_report.html`：项目二完整报告
- `output/project2/project2_final_prediction_2015_01_11_17.csv`：最终 7 天预测结果
- `output/project2/models/`：最终选中模型文件（joblib 格式）

## GitHub 上传流程

第一次上传：

```powershell
git init
git add .
git commit -m "Initial project2 load forecasting"
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

后续每次修改后保存版本：

```powershell
git status
git add .
git commit -m "Update project2 analysis"
git push
```

建议每次我们完成一轮修改，就执行一次 `git add`、`git commit`、`git push`，这样 GitHub 上会有历史版本，不容易丢。
