# 第三部分对接说明：成品展示、网页设计与模型调用

## 1. 推荐成品展示口径

第二部分最终结论是：`Stacking` 作为正式预测主模型，`GRU/LSTM/Attention-GRU` 作为 PyTorch + GPU 深度学习探索。第三部分展示时建议把主界面分成三块：

1. 数据概览：展示原始负荷、天气、特征工程和异常值处理摘要。
2. 模型结果：展示 Stacking 最终预测、验证集拟合、误差指标、统计检验。
3. 深度学习探索：展示 GRU/LSTM/Attention-GRU 的训练过程、GPU 信息、loss 曲线和与 Stacking 的对比。

## 2. 网页设计建议

- 首页不要做营销页，直接做项目仪表盘。
- 顶部放 7 天预测结果卡片：日期、日平均、日最高、日最低。
- 中部放图表：验证拟合、最终预测曲线、模型 RMSE 对比、深度学习 loss 曲线。
- 底部放方法说明：为什么最终选 Stacking、为什么尝试 PyTorch GRU/LSTM、Attention 结果如何解释。

## 3. 可直接读取的数据文件

正式主模型预测：

- `V6/output/project2/project2_final_prediction_2015_01_11_17.csv`

深度学习探索预测：

- `V6/output/project2/project2_deep_learning_forecast_2015_01_11_17.csv`

模型性能：

- `V6/output/project2/project2_model_performance.csv`
- `V6/output/project2/project2_deep_learning_performance.csv`
- `V6/output/project2/project2_rolling_4week_validation.csv`

前端展示图：

- `03_validation_fit.png`
- `04_final_prediction.png`
- `10_deep_learning_loss.png`
- `11_deep_learning_validation_fit.png`
- `12_attention_weights.png`
- `13_deep_learning_vs_stacking.png`

## 4. 模型调用建议

当前最稳的调用方式是直接读取已经生成的最终预测 CSV。若需要在线调用模型，优先加载 Stacking 模型：

- `models/load_mean_Stacking.joblib`
- `models/load_max_Stacking.joblib`
- `models/load_min_Stacking.joblib`

深度学习模型用于展示和扩展：

- `deep_learning/GRU.pt`
- `deep_learning/LSTM.pt`
- `deep_learning/Attention_GRU.pt`
- `deep_learning/deep_learning_scalers.joblib`

## 5. 与其他项目融合

如果要和别的项目融合，建议输出统一 JSON：

```json
{
  "date": "2015-01-11",
  "weekday": "周日",
  "main_model": "Stacking",
  "load_mean_mw": 6098.66,
  "load_max_mw": 7656.09,
  "load_min_mw": 4677.13,
  "deep_learning_reference": {
    "model": "GRU",
    "load_mean_mw": 5824.39,
    "load_max_mw": 7223.65,
    "load_min_mw": 4409.29
  }
}
```

对接方只需要知道：主预测采用 `Stacking`，深度学习结果是参考/探索结果，不替代最终主模型。
