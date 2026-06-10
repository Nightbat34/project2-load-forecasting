# 后端 API 使用说明

启动：

```powershell
python backend/app.py
```

默认地址：

```text
http://127.0.0.1:5000
```

## 健康检查

```powershell
curl http://127.0.0.1:5000/api/health
```

## 获取最终预测结果

```powershell
curl http://127.0.0.1:5000/api/final-forecast
```

返回内容为 2015-01-11 至 2015-01-17 的最终预测表。

## 获取模型特征清单

```powershell
curl http://127.0.0.1:5000/api/model-manifest
```

返回每个目标模型的特征列表。

## 调用模型预测

```http
POST /api/predict
Content-Type: application/json
```

请求格式：

```json
{
  "target": "load_mean",
  "rows": [
    {
      "temp_max": 12.0,
      "temp_min": 4.0,
      "temp_avg": 8.0,
      "humidity": 70.0,
      "rainfall": 0.0,
      "temp_range": 8.0,
      "hdd": 10.0,
      "cdd": 0.0,
      "month_sin": 0.5,
      "month_cos": 0.866,
      "dow_sin": 0.0,
      "dow_cos": 1.0,
      "doy_sin": 0.2,
      "doy_cos": 0.98,
      "is_weekend": 0,
      "load_mean_lag_1": 6500.0,
      "load_mean_lag_7": 6300.0,
      "load_mean_lag_14": 6200.0,
      "load_mean_roll_mean_7": 6400.0,
      "load_mean_roll_std_7": 300.0,
      "load_mean_roll_mean_14": 6350.0,
      "load_mean_roll_std_14": 350.0
    }
  ]
}
```

`target` 可选：

```text
load_mean
load_max
load_min
```

注意：示例数值只展示字段格式，不代表真实业务输入。真实调用时必须由后端根据历史负荷数据构造滞后特征和滚动特征。
