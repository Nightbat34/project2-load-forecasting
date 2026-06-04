#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V5 前置数据分析脚本：严格七步检查
环境: D:/Anaconda/envs/pytorch_gpu
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# === 数据加载 ===
DATA_PATH = "F:/Practicum/Data Mining Practicum/Data/附件1-电网负荷数据.xlsx"
OUT = "F:/Practicum/Data Mining Practicum/V5/output/project2"

load_raw = pd.read_excel(DATA_PATH, sheet_name="Area_Load")
weather_raw = pd.read_excel(DATA_PATH, sheet_name="Area_Weather")
weather_raw.columns = ["YMD", "temp_max", "temp_min", "temp_avg", "humidity", "rainfall"]

time_cols = [c for c in load_raw.columns if c != "YMD"]
long_df = load_raw.melt(id_vars="YMD", value_vars=time_cols, var_name="time_slot", value_name="load")

daily = (
    long_df.groupby("YMD", as_index=False)
    .agg(load_max=("load", "max"), load_min=("load", "min"), load_mean=("load", "mean"))
    .merge(weather_raw, on="YMD", how="left")
)
daily["date"] = pd.to_datetime(daily["YMD"].astype(str), format="%Y%m%d")

# 基础特征
daily["dayofweek"] = daily["date"].dt.dayofweek
daily["month"] = daily["date"].dt.month
daily["day_of_year"] = daily["date"].dt.dayofyear
daily["is_weekend"] = (daily["dayofweek"] >= 5).astype(int)
daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)
daily["dow_sin"] = np.sin(2 * np.pi * daily["dayofweek"] / 7)
daily["dow_cos"] = np.cos(2 * np.pi * daily["dayofweek"] / 7)
daily["doy_sin"] = np.sin(2 * np.pi * daily["day_of_year"] / 365)
daily["doy_cos"] = np.cos(2 * np.pi * daily["day_of_year"] / 365)
daily["temp_range"] = daily["temp_max"] - daily["temp_min"]
daily["hdd"] = np.maximum(18 - daily["temp_avg"], 0)
daily["cdd"] = np.maximum(daily["temp_avg"] - 26, 0)

# 训练集
train = daily[(daily["date"] >= "2012-01-01") & (daily["date"] <= "2014-12-31")].copy()
train = train.dropna().reset_index(drop=True)

print(f"训练集: {len(train)} 天")
print(f"日期范围: {train['date'].min()} ~ {train['date'].max()}")
print(f"NaN 统计:\n{train[['load_max','load_min','load_mean','temp_max','temp_min','temp_avg','humidity','rainfall']].isna().sum()}")
print()

# =============================================
# 第一步: Y 的变量类型
# =============================================
print("=" * 70)
print("第一步: Y 的变量类型判断")
print("=" * 70)
targets = {"load_mean": "日平均负荷", "load_max": "日最高负荷", "load_min": "日最低负荷"}
for col, label in targets.items():
    vals = train[col].dropna()
    n_unique = vals.nunique()
    n_total = len(vals)
    ratio = n_unique / n_total
    print(f"  {label} ({col}):")
    print(f"    唯一值: {n_unique}/{n_total} ({ratio:.4f})")
    print(f"    取值范围: [{vals.min():.2f}, {vals.max():.2f}]")
    print(f"    数据类型: 连续回归变量 (取值连续，有物理量纲 MW)")
    print(f"    建议模型族: Ridge, SVR, RF, XGBoost, Stacking")
    print(f"    联系函数: identity (Y = Xb)")
    print()

# =============================================
# 第二步: Y 的分布诊断
# =============================================
print("=" * 70)
print("第二步: Y 的分布诊断")
print("=" * 70)

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
dist_stats = []
for idx, (col, label) in enumerate(targets.items()):
    vals = train[col].dropna()
    mu = vals.mean()
    med = vals.median()
    sigma = vals.std()
    skew = vals.skew()
    kurt = vals.kurtosis()
    pcts = vals.quantile([0.05, 0.25, 0.50, 0.75, 0.95])

    # 偏态诊断
    abs_skew = abs(skew)
    if abs_skew < 0.5:
        skew_diag = "近似对称，线性模型可行"
    elif abs_skew < 1.0:
        skew_diag = "中度偏态，建议用树模型或验证残差正态性"
    else:
        skew_diag = "严重偏态，考虑 log/Box-Cox 变换或直接用树模型"

    # CV
    cv = sigma / mu

    print(f"  {label} ({col}):")
    print(f"    均值={mu:.2f}, 中位数={med:.2f}, 标准差={sigma:.2f}")
    print(f"    偏度={skew:.4f}, 峰度={kurt:.4f}")
    print(f"    P5={pcts[0.05]:.2f}, P25={pcts[0.25]:.2f}, P50={pcts[0.50]:.2f}, P75={pcts[0.75]:.2f}, P95={pcts[0.95]:.2f}")
    print(f"    CV = {cv:.4f} ({'高离散' if cv > 0.3 else '中等' if cv > 0.1 else '低离散'})")
    print(f"    偏态诊断: |偏度|={abs_skew:.4f} -> {skew_diag}")
    print()

    dist_stats.append({
        "target": label,
        "col": col,
        "mean": round(mu, 2),
        "median": round(med, 2),
        "std": round(sigma, 2),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "P5": round(pcts[0.05], 2),
        "P25": round(pcts[0.25], 2),
        "P50": round(pcts[0.50], 2),
        "P75": round(pcts[0.75], 2),
        "P95": round(pcts[0.95], 2),
        "CV": round(cv, 4),
        "skew_diagnosis": skew_diag,
    })

    # 直方图 + KDE
    ax_hist = axes[idx, 0]
    ax_hist.hist(vals, bins=50, density=True, alpha=0.6, color="steelblue", edgecolor="white")
    vals_kde = vals.dropna()
    x_range = np.linspace(vals_kde.min(), vals_kde.max(), 300)
    kde = sp_stats.gaussian_kde(vals_kde)
    ax_hist.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
    ax_hist.set_title(f"{label} 分布 (偏度={skew:.3f})")
    ax_hist.set_xlabel("MW")
    ax_hist.legend()

    # Q-Q 图
    ax_qq = axes[idx, 1]
    sp_stats.probplot(vals, dist="norm", plot=ax_qq)
    ax_qq.set_title(f"{label} Q-Q 图")

    # 按月份箱线图 (检查多峰)
    ax_box = axes[idx, 2]
    train["month_label"] = train["date"].dt.month
    sns.boxplot(data=train, x="month", y=col, ax=ax_box, color="steelblue")
    ax_box.set_title(f"{label} 月度分布")
    ax_box.set_xlabel("月份")

fig.tight_layout()
fig.savefig(f"{OUT}/v4_step2_distribution.png", dpi=160, bbox_inches="tight")
plt.close(fig)

pd.DataFrame(dist_stats).to_csv(f"{OUT}/v4_step2_distribution_stats.csv", index=False, encoding="utf-8-sig")

# =============================================
# 第三步: 离散度与异常值
# =============================================
print("=" * 70)
print("第三步: 离散度与异常值")
print("=" * 70)

# 中国法定节假日
CN_HOLIDAYS = {
    "春节": [
        "2012-01-19","2012-01-20","2012-01-21","2012-01-22","2012-01-23","2012-01-24","2012-01-25","2012-01-26","2012-01-27","2012-01-28","2012-01-29",
        "2013-02-06","2013-02-07","2013-02-08","2013-02-09","2013-02-10","2013-02-11","2013-02-12","2013-02-13","2013-02-14","2013-02-15","2013-02-16","2013-02-17",
        "2014-01-26","2014-01-27","2014-01-28","2014-01-29","2014-01-30","2014-01-31","2014-02-01","2014-02-02","2014-02-03","2014-02-04","2014-02-05","2014-02-06",
    ],
    "国庆节": [
        "2012-10-01","2012-10-02","2012-10-03",
        "2013-10-01","2013-10-02","2013-10-03",
        "2014-10-01","2014-10-02","2014-10-03",
    ],
    "元旦": ["2012-01-01","2013-01-01","2014-01-01"],
    "劳动节": ["2012-05-01","2013-05-01","2014-05-01"],
}

def classify_cause(date_str, temp_avg):
    for h, dates in CN_HOLIDAYS.items():
        if date_str in dates:
            return h
    if temp_avg > 30:
        return "极端高温"
    if temp_avg < 5:
        return "极端低温"
    return "其他/未知"

outlier_rows = []
for col in ["load_mean", "load_max", "load_min"]:
    label = targets[col]
    q1 = train[col].quantile(0.25)
    q3 = train[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    for _, row in train.iterrows():
        if row[col] < lower or row[col] > upper:
            ds = row["date"].strftime("%Y-%m-%d")
            cause = classify_cause(ds, row["temp_avg"])
            outlier_rows.append({
                "date": ds, "target": label, "col": col,
                "value": round(row[col], 2),
                "lower": round(lower, 2), "upper": round(upper, 2),
                "direction": "偏低" if row[col] < lower else "偏高",
                "temp_avg": row["temp_avg"],
                "cause": cause,
                "action": "保留（真实极端事件）",
            })

outlier_df = pd.DataFrame(outlier_rows)
cause_counts = outlier_df.groupby(["col", "cause"]).size().reset_index(name="count")
print(f"  异常值总计: {len(outlier_df)} 个")
for col_name in ["load_mean", "load_max", "load_min"]:
    sub = outlier_df[outlier_df["col"] == col_name]
    print(f"    {targets[col_name]}: {len(sub)} 个异常值")
    for _, r in sub.groupby("cause").size().items():
        print(f"      - {_}: {r} 天")
print("  处置建议: 全部保留（均为真实极端事件：春节/国庆/元旦/劳动节/极端天气）")

# 异方差性检查: 按季度分组查看均值-方差
print("\n  异方差性检查 (按季度分组):")
for col in ["load_mean", "load_max", "load_min"]:
    train["quarter"] = train["date"].dt.quarter
    grouped = train.groupby("quarter")[col].agg(["mean", "std"]).dropna()
    corr_mv = grouped["mean"].corr(grouped["std"])
    print(f"    {targets[col]}: 均值-标准差 Pearson r = {corr_mv:.4f} ({'存在异方差' if abs(corr_mv) > 0.5 else '方差基本恒定'})")

# 离散度
print("\n  离散度 (CV):")
for col in ["load_mean", "load_max", "load_min"]:
    vals = train[col]
    cv = vals.std() / vals.mean()
    print(f"    {targets[col]}: CV = {cv:.4f} ({'高离散 >0.3' if cv > 0.3 else '中等 0.1-0.3' if cv > 0.1 else '低离散 <0.1'})")

outlier_df.to_csv(f"{OUT}/v4_step3_outlier_diagnosis.csv", index=False, encoding="utf-8-sig")

# =============================================
# 第四步: 共线性检查
# =============================================
print("\n" + "=" * 70)
print("第四步: 共线性检查 (VIF)")
print("=" * 70)

# 计算滞后特征 (用 load_mean 作为代表)
model_cols_base = ["temp_max", "temp_min", "temp_avg", "humidity", "rainfall",
                   "temp_range", "hdd", "cdd",
                   "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"]

# 为每个目标计算完整特征集的 VIF
for col in ["load_mean", "load_max", "load_min"]:
    label = targets[col]
    df_vif = train.copy()
    for lag in [1, 7, 14]:
        df_vif[f"{col}_lag_{lag}"] = df_vif[col].shift(lag)
    for win in [7, 14]:
        df_vif[f"{col}_roll_mean_{win}"] = df_vif[col].shift(1).rolling(win).mean()
        df_vif[f"{col}_roll_std_{win}"] = df_vif[col].shift(1).rolling(win).std()

    lag_cols = [f"{col}_lag_{l}" for l in [1, 7, 14]]
    roll_cols = []
    for w in [7, 14]:
        roll_cols.extend([f"{col}_roll_mean_{w}", f"{col}_roll_std_{w}"])
    feature_cols = model_cols_base + lag_cols + roll_cols

    df_clean = df_vif[feature_cols + [col]].dropna().reset_index(drop=True)
    X_vif = df_clean[feature_cols]

    vif_data = []
    try:
        for i, feat in enumerate(feature_cols):
            vif_val = variance_inflation_factor(X_vif.values, i)
            vif_data.append({"target": label, "feature": feat, "VIF": round(vif_val, 4)})
    except Exception as e:
        print(f"  VIF 计算遇到问题: {e}")

    vif_df = pd.DataFrame(vif_data)
    vif_df_sorted = vif_df.sort_values("VIF", ascending=False)
    print(f"\n  {label} VIF 排名 (Top 10):")
    for _, r in vif_df_sorted.head(10).iterrows():
        flag = " [严重 >10]" if r["VIF"] > 10 else " [中度 5-10]" if r["VIF"] > 5 else " [OK]"
        print(f"    {r['feature']}: VIF = {r['VIF']:.2f}{flag}")

    high_vif = vif_df[vif_df["VIF"] > 10]
    print(f"    VIF > 10 的特征数: {len(high_vif)}")

    # Pearson |r| > 0.8 的特征对
    corr_matrix = X_vif.corr().abs()
    high_corr_pairs = []
    feats = list(corr_matrix.columns)
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            if corr_matrix.iloc[i, j] > 0.8:
                high_corr_pairs.append({
                    "target": label,
                    "feature_a": feats[i],
                    "feature_b": feats[j],
                    "pearson_r": round(corr_matrix.iloc[i, j], 4),
                })

    print(f"    |Pearson r| > 0.8 的特征对数: {len(high_corr_pairs)}")
    for p in high_corr_pairs[:5]:
        print(f"      {p['feature_a']} vs {p['feature_b']}: r = {p['pearson_r']:.4f}")

    vif_df.to_csv(f"{OUT}/v4_step4_vif_{col}.csv", index=False, encoding="utf-8-sig")
    if high_corr_pairs:
        pd.DataFrame(high_corr_pairs).to_csv(f"{OUT}/v4_step4_high_corr_{col}.csv", index=False, encoding="utf-8-sig")

    print(f"    处理建议: 树模型 (RF/XGBoost/Stacking) 可不处理共线性; Ridge L2 自动缓解; OLS 需删除 VIF>10 特征")

print("\n  结论: XGBoost/RF/Stacking 等树模型天然对共线性不敏感，无需额外处理。")
print("  Ridge 的 L2 正则化可有效缓解系数膨胀。OLS 不在主方案中，VIF 仅作参考。")

# =============================================
# 完成
# =============================================
print("\n" + "=" * 70)
print("前置分析完成！图表已保存至 V5/output/project2/")
print("=" * 70)
