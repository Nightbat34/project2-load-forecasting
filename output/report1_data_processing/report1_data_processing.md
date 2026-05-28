# 第一次汇报：数据处理与特征构建

本次汇报只覆盖项目二的数据处理部分，不展开模型训练。

## 1. 原始数据

- `Area_Load`：每天一行，96 个 15 分钟负荷采样点。
- `Area_Weather`：每天一行，包含最高温、最低温、平均温、湿度、降雨量。
- 预测目标期：`2015-01-11` 至 `2015-01-17`。

| 检查项 | 结果 | 说明 |
| --- | --- | --- |
| Area_Load 原始负荷表 | 2208 行 x 97 列 | 每天一行，96 个 15 分钟采样点。 |
| Area_Weather 原始天气表 | 1113 行 x 6 列 | 每天一行，包含最高/最低/平均温度、湿度、降雨量。 |
| 负荷采样列数 | 96 个 | 96 = 24 小时 x 每小时 4 个 15 分钟点。 |
| 预测目标期负荷缺失 | 672 个单元格 | 正好 7 天 x 96 点，是题目要求预测的目标，不作为异常删除。 |
| 目标期天气是否完整 | 0 个缺失 | 递推预测时可以使用目标期已给定天气。 |
| 建模日粒度表 | 2208 行 x 23 列 | 由 15 分钟负荷聚合为日最大、日最小、日平均。 |

## 2. 数据处理流程

1. 读取 Excel 的两个工作表。
2. 将负荷宽表通过 `melt` 转成长表。
3. 按日期聚合为 `load_max`、`load_min`、`load_mean`。
4. 按 `YMD` 合并天气数据。
5. 构造天气、日历、周期、滞后和滚动特征。

![数据处理主线](assets/03_data_pipeline.png)

## 3. 核心代码：读取与日粒度聚合

![读取、重构、聚合、合并代码截图](assets/09_code_load_daily_data.png)

```python
def load_daily_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    # 原始负荷表是“每天一行、96个15分钟采样点”的宽表；建模前先压缩成日粒度。
    # 这样能直接对应题目要求的日最高、日最低、日平均三个预测目标。
    load_raw = pd.read_excel(DATA_PATH, sheet_name="Area_Load")
    weather_raw = pd.read_excel(DATA_PATH, sheet_name="Area_Weather")
    weather_raw.columns = ["YMD", "temp_max", "temp_min", "temp_avg", "humidity", "rainfall"]

    time_cols = [col for col in load_raw.columns if col != "YMD"]
    long_df = load_raw.melt(id_vars="YMD", value_vars=time_cols, var_name="time_slot", value_name="load")
    daily = (
        long_df.groupby("YMD", as_index=False)
        .agg(load_max=("load", "max"), load_min=("load", "min"), load_mean=("load", "mean"))
        .merge(weather_raw, on="YMD", how="left")
    )
    daily["date"] = pd.to_datetime(daily["YMD"].astype(str), format="%Y%m%d")
    return add_features(daily.sort_values("date").reset_index(drop=True))
```

![缺失值检查](assets/04_missingness_check.png)

## 4. 核心代码：基础特征工程

![日历周期与天气衍生特征代码截图](assets/10_code_add_features.png)

```python
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # 周期特征不能直接用“月份=12、1”这种数值距离，因为12月和1月在时间上相邻。
    # sin/cos 编码把周期映射到圆上，保留“首尾相接”的季节/星期规律。
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # HDD/CDD 是电力负荷常用气象衍生变量：低温带来供热需求，高温带来制冷需求。
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["hdd"] = np.maximum(18 - df["temp_avg"], 0)
    df["cdd"] = np.maximum(df["temp_avg"] - 26, 0)
    return df
```

![2014 年日负荷曲线](assets/05_2014_daily_load.png)

## 5. 核心代码：滞后与滚动特征

![特征工程结构](assets/08_feature_engineering_blocks.png)

![滞后与滚动特征代码截图](assets/11_code_make_target_features.png)

```python
def make_target_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    lag_features: list[str] = []

    # 负荷序列有强惯性和周周期：昨天、上周同日、两周前同日通常是强预测信号。
    # 每个目标单独构造滞后项，避免用“日均负荷”的历史去预测“日最高/最低”时混淆目标。
    for lag in [1, 7, 14]:
        col = f"{target}_lag_{lag}"
        out[col] = out[target].shift(lag)
        lag_features.append(col)

    # 滚动均值/标准差描述近期负荷水平和波动程度，是短期预测里很实用的平滑特征。
    for win in [7, 14]:
        mean_col = f"{target}_roll_mean_{win}"
        std_col = f"{target}_roll_std_{win}"
        out[mean_col] = out[target].shift(1).rolling(win).mean()
        out[std_col] = out[target].shift(1).rolling(win).std()
        lag_features.extend([mean_col, std_col])
    return out, MODEL_WEATHER + CALENDAR_FEATURES + lag_features
```

## 6. 汇报结论

处理后的数据表已经把原始 15 分钟负荷转换成项目要求的日最高、日最低、日平均三个预测目标，并补充了天气、日历周期和历史负荷特征。第二次汇报即可进入算法模型训练与对比。
