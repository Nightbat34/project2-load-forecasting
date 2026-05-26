#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project 2: short-term power load forecasting with weather factors.

This focused pipeline follows the assignment requirement:
1. Use 2012-01-01 to 2014-12-31 to regress daily max/min/mean load
   against weather factors.
2. Build a short-term forecasting method and predict 2015-01-11 to
   2015-01-17 daily max/min/mean load.
"""

from __future__ import annotations

import base64
import html
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("LOAD_DATA_PATH", ROOT.parent / "Data" / "附件1-电网负荷数据.xlsx"))
OUTPUT_DIR = ROOT / "output" / "project2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")


TARGETS = {
    "load_max": "日最高负荷",
    "load_min": "日最低负荷",
    "load_mean": "日平均负荷",
}
BASE_WEATHER = ["temp_max", "temp_min", "temp_avg", "humidity", "rainfall"]
REG_WEATHER = ["temp_avg", "temp_range", "humidity", "rainfall", "hdd", "cdd"]
MODEL_WEATHER = ["temp_max", "temp_min", "temp_avg", "humidity", "rainfall", "temp_range", "hdd", "cdd"]
CALENDAR_FEATURES = [
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
]
WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


@dataclass
class TargetFit:
    target: str
    model_name: str
    estimator: object
    features: list[str]
    validation_pred: np.ndarray
    metrics: dict[str, float]


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_daily_data() -> pd.DataFrame:
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
    daily = add_calendar_and_weather_features(daily)
    return daily.sort_values("date").reset_index(drop=True)


def add_calendar_and_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["hdd"] = np.maximum(18 - df["temp_avg"], 0)
    df["cdd"] = np.maximum(df["temp_avg"] - 26, 0)
    return df


def make_target_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    lag_features: list[str] = []
    for lag in [1, 7, 14]:
        name = f"{target}_lag_{lag}"
        out[name] = out[target].shift(lag)
        lag_features.append(name)
    for win in [7, 14]:
        mean_col = f"{target}_roll_mean_{win}"
        std_col = f"{target}_roll_std_{win}"
        out[mean_col] = out[target].shift(1).rolling(win).mean()
        out[std_col] = out[target].shift(1).rolling(win).std()
        lag_features.extend([mean_col, std_col])

    features = MODEL_WEATHER + CALENDAR_FEATURES + lag_features
    return out, features


def build_relationship_tables(train_2012_2014: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_rows: list[dict[str, object]] = []
    for target in TARGETS:
        for feature in MODEL_WEATHER:
            corr_rows.append(
                {
                    "target": TARGETS[target],
                    "weather_factor": feature,
                    "pearson_r": train_2012_2014[[target, feature]].corr().iloc[0, 1],
                }
            )
    corr_df = pd.DataFrame(corr_rows)

    ols_rows: list[dict[str, object]] = []
    for target in TARGETS:
        model_df = train_2012_2014[[target] + REG_WEATHER].dropna()
        X = sm.add_constant(model_df[REG_WEATHER])
        y = model_df[target]
        result = sm.OLS(y, X).fit()
        for term in ["const"] + REG_WEATHER:
            ols_rows.append(
                {
                    "target": TARGETS[target],
                    "term": term,
                    "coef": result.params[term],
                    "p_value": result.pvalues[term],
                    "r_squared": result.rsquared,
                    "adj_r_squared": result.rsquared_adj,
                }
            )
    ols_df = pd.DataFrame(ols_rows)
    return corr_df, ols_df


def validation_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "MAPE(%)": round(float(np.mean(np.abs((y_true.to_numpy() - y_pred) / y_true.to_numpy())) * 100), 3),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def candidate_models() -> dict[str, object]:
    return {
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=20.0))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            tree_method="hist",
            verbosity=0,
        ),
    }


def fit_models(daily: pd.DataFrame) -> tuple[dict[str, TargetFit], pd.DataFrame, pd.DataFrame]:
    fits: dict[str, TargetFit] = {}
    performance_rows: list[dict[str, object]] = []
    validation_frames: list[pd.DataFrame] = []

    for target, label in TARGETS.items():
        feature_df, features = make_target_features(daily, target)
        model_df = feature_df[
            (feature_df["date"] >= "2012-01-01")
            & (feature_df["date"] <= "2014-12-31")
            & feature_df[target].notna()
        ].dropna(subset=features + [target])

        train_mask = model_df["date"] <= "2014-10-31"
        val_mask = model_df["date"] >= "2014-11-01"
        X_train, y_train = model_df.loc[train_mask, features], model_df.loc[train_mask, target]
        X_val, y_val = model_df.loc[val_mask, features], model_df.loc[val_mask, target]

        best_name = ""
        best_estimator = None
        best_pred: np.ndarray | None = None
        best_rmse = float("inf")

        for name, estimator in candidate_models().items():
            fitted = clone(estimator)
            fitted.fit(X_train, y_train)
            pred = fitted.predict(X_val)
            metrics = validation_metrics(y_val, pred)
            performance_rows.append({"target": label, "model": name, **metrics})
            if metrics["RMSE"] < best_rmse:
                best_name, best_estimator, best_pred, best_rmse = name, fitted, pred, metrics["RMSE"]

        final_estimator = clone(candidate_models()[best_name])
        final_estimator.fit(model_df[features], model_df[target])
        metrics = validation_metrics(y_val, best_pred)
        fits[target] = TargetFit(target, best_name, final_estimator, features, best_pred, metrics)

        validation_frames.append(
            pd.DataFrame(
                {
                    "date": model_df.loc[val_mask, "date"].to_numpy(),
                    "target": label,
                    "actual": y_val.to_numpy(),
                    "predicted": best_pred,
                    "model": best_name,
                }
            )
        )

    return fits, pd.DataFrame(performance_rows), pd.concat(validation_frames, ignore_index=True)


def forecast_recursive(daily: pd.DataFrame, fits: dict[str, TargetFit]) -> pd.DataFrame:
    pred_dates = pd.date_range("2015-01-11", "2015-01-17", freq="D")
    known_daily = daily[daily["date"] <= "2015-01-10"].copy()
    history = {
        target: dict(zip(known_daily["date"], known_daily[target]))
        for target in TARGETS
    }
    weather_lookup = daily.set_index("date")[MODEL_WEATHER + CALENDAR_FEATURES].to_dict("index")
    rows: list[dict[str, object]] = []

    for date in pred_dates:
        row: dict[str, object] = {"date": date, "weekday": WEEKDAY_CN[date.dayofweek]}
        for target, fit in fits.items():
            feature_values: dict[str, float] = {}
            feature_values.update(weather_lookup[date])

            values_7 = []
            values_14 = []
            for i in range(1, 15):
                hist_date = date - pd.Timedelta(days=i)
                value = history[target].get(hist_date, np.nan)
                if i in [1, 7, 14]:
                    feature_values[f"{target}_lag_{i}"] = value
                if i <= 7:
                    values_7.append(value)
                values_14.append(value)

            arr7 = np.array(values_7, dtype=float)
            arr14 = np.array(values_14, dtype=float)
            feature_values[f"{target}_roll_mean_7"] = float(np.nanmean(arr7))
            feature_values[f"{target}_roll_std_7"] = float(np.nanstd(arr7, ddof=1))
            feature_values[f"{target}_roll_mean_14"] = float(np.nanmean(arr14))
            feature_values[f"{target}_roll_std_14"] = float(np.nanstd(arr14, ddof=1))

            X = pd.DataFrame([feature_values])[fit.features]
            pred = float(fit.estimator.predict(X)[0])
            history[target][date] = pred
            row[f"pred_{target}"] = pred
            row[f"model_{target}"] = fit.model_name

        rows.append(row)

    pred_df = pd.DataFrame(rows)
    fit_base = daily[(daily["date"] >= "2012-01-01") & (daily["date"] <= "2014-12-31")]
    max_gap = float((fit_base["load_max"] - fit_base["load_mean"]).median())
    min_gap = float((fit_base["load_mean"] - fit_base["load_min"]).median())

    low_bad = pred_df["pred_load_min"] > pred_df["pred_load_mean"]
    high_bad = pred_df["pred_load_max"] < pred_df["pred_load_mean"]
    pred_df.loc[low_bad, "pred_load_min"] = pred_df.loc[low_bad, "pred_load_mean"] - min_gap
    pred_df.loc[high_bad, "pred_load_max"] = pred_df.loc[high_bad, "pred_load_mean"] + max_gap
    return pred_df


def save_figures(
    train_2012_2014: pd.DataFrame,
    corr_df: pd.DataFrame,
    ols_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> None:
    corr_pivot = corr_df.pivot(index="target", columns="weather_factor", values="pearson_r")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(corr_pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[0])
    axes[0].set_title("2012-2014 负荷与气象因素 Pearson 相关系数")
    axes[0].set_xlabel("气象因素")
    axes[0].set_ylabel("负荷指标")

    coef_df = ols_df[ols_df["term"].isin(REG_WEATHER)].copy()
    coef_df["coef_scaled"] = coef_df.groupby("target")["coef"].transform(
        lambda s: s / max(np.nanmax(np.abs(s)), 1)
    )
    sns.barplot(data=coef_df, y="term", x="coef_scaled", hue="target", ax=axes[1])
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_title("多元线性回归系数方向（按目标归一化）")
    axes[1].set_xlabel("归一化系数")
    axes[1].set_ylabel("解释变量")
    axes[1].legend(title="")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_weather_regression.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, (target, label) in zip(axes, TARGETS.items()):
        data = validation_df[validation_df["target"] == label]
        ax.plot(data["date"], data["actual"], "o-", label="实际值", color="#111827", linewidth=2)
        ax.plot(data["date"], data["predicted"], "s--", label="验证预测", color="#2563eb", linewidth=1.8)
        ax.set_ylabel("MW")
        ax.set_title(f"{label}：2014-11-01 至 2014-12-31 验证集")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_validation_fit.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(pred_df))
    ax.fill_between(
        x,
        pred_df["pred_load_min"],
        pred_df["pred_load_max"],
        color="#93c5fd",
        alpha=0.35,
        label="预测日最低-日最高区间",
    )
    ax.plot(x, pred_df["pred_load_mean"], "o-", color="#1d4ed8", linewidth=2.5, label="预测日平均负荷")
    ax.plot(x, pred_df["pred_load_max"], "^--", color="#dc2626", linewidth=1.5, label="预测日最高负荷")
    ax.plot(x, pred_df["pred_load_min"], "v--", color="#16a34a", linewidth=1.5, label="预测日最低负荷")
    for idx, value in enumerate(pred_df["pred_load_mean"]):
        ax.text(idx, value + 80, f"{value:.0f}", ha="center", fontsize=9, fontweight="bold")
    labels = [f"{d:%m-%d}\n{w}" for d, w in zip(pred_df["date"], pred_df["weekday"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("负荷 (MW)")
    ax.set_title("2015-01-11 至 2015-01-17 电力负荷预测")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_final_prediction.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    history = train_2012_2014[train_2012_2014["date"] >= "2014-10-01"]
    ax.plot(history["date"], history["load_mean"], color="#64748b", label="2014年末日平均负荷")
    ax.plot(pred_df["date"], pred_df["pred_load_mean"], "o-", color="#1d4ed8", label="2015年1月预测日平均")
    ax.axvline(pd.Timestamp("2015-01-11"), color="#dc2626", linestyle="--", linewidth=1)
    ax.set_title("预测窗口与历史短期趋势衔接")
    ax.set_ylabel("日平均负荷 (MW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_history_forecast_context.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def format_table(df: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    def fmt(value: object) -> object:
        if isinstance(value, float):
            return float_format.format(value)
        return value

    formatted = df.map(fmt)
    return formatted.to_html(index=False, escape=False, border=0, classes="data-table")


def generate_report(
    daily: pd.DataFrame,
    corr_df: pd.DataFrame,
    ols_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> None:
    best_rows = (
        performance_df.sort_values(["target", "RMSE"])
        .groupby("target", as_index=False)
        .first()
        .sort_values("target")
    )
    corr_rank = (
        corr_df.assign(abs_r=lambda d: d["pearson_r"].abs())
        .sort_values(["target", "abs_r"], ascending=[True, False])
        .groupby("target")
        .head(5)
        [["target", "weather_factor", "pearson_r"]]
    )
    final_table = pred_df[
        ["date", "weekday", "pred_load_mean", "pred_load_max", "pred_load_min", "model_load_mean", "model_load_max", "model_load_min"]
    ].copy()
    final_table["date"] = final_table["date"].dt.strftime("%Y-%m-%d")
    final_table.columns = ["日期", "星期", "预测日平均(MW)", "预测日最高(MW)", "预测日最低(MW)", "均值模型", "最高模型", "最低模型"]

    target_missing_daily = daily[(daily["date"] >= "2015-01-11") & (daily["date"] <= "2015-01-17")][
        ["load_max", "load_min", "load_mean"]
    ].isna().sum().sum()
    target_missing_points = int(target_missing_daily / 3 * 96)
    weather_full = daily[(daily["date"] >= "2015-01-11") & (daily["date"] <= "2015-01-17")][BASE_WEATHER].notna().all().all()

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>项目二：电力系统短期负荷预测</title>
<style>
body {{ margin: 0; background: #f6f8fb; color: #1f2937; font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif; line-height: 1.72; }}
header {{ background: #12355b; color: white; padding: 42px 56px; }}
header h1 {{ margin: 0 0 10px; font-size: 30px; }}
header p {{ margin: 0; color: #dbeafe; }}
main {{ max-width: 1120px; margin: 28px auto 56px; padding: 0 24px; }}
section {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 26px 30px; margin-bottom: 22px; box-shadow: 0 1px 5px rgba(15,23,42,.05); }}
h2 {{ margin: 0 0 14px; font-size: 21px; color: #12355b; }}
h3 {{ margin: 20px 0 10px; font-size: 16px; color: #2563eb; }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 16px 0; }}
.card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; background: #f8fafc; }}
.label {{ color: #64748b; font-size: 12px; }}
.value {{ color: #0f172a; font-size: 22px; font-weight: 800; }}
.note {{ border-left: 4px solid #2563eb; background: #eff6ff; padding: 12px 16px; border-radius: 0 8px 8px 0; }}
.warn {{ border-left-color: #d97706; background: #fffbeb; }}
img {{ max-width: 100%; border-radius: 8px; border: 1px solid #e5e7eb; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 10px 0 16px; }}
th {{ background: #1f3a5f; color: white; text-align: left; padding: 8px 10px; }}
td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; }}
tr:nth-child(even) td {{ background: #f8fafc; }}
code {{ background: #eef2ff; padding: 2px 6px; border-radius: 4px; }}
@media (max-width: 820px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
</style>
</head>
<body>
<header>
  <h1>项目二：电力系统短期负荷预测</h1>
  <p>基于 2012-01-01 至 2014-12-31 的负荷与气象数据进行回归分析，并预测 2015-01-11 至 2015-01-17 的日最高、日最低、日平均负荷。</p>
</header>
<main>
<section>
  <h2>1. 数据边界与质量核对</h2>
  <div class="grid">
    <div class="card"><div class="label">负荷原始天数</div><div class="value">{len(daily)}</div></div>
    <div class="card"><div class="label">建模训练期</div><div class="value">2012-2014</div></div>
    <div class="card"><div class="label">预测目标缺失点</div><div class="value">{target_missing_points}</div></div>
    <div class="card"><div class="label">目标期气象完整</div><div class="value">{"是" if weather_full else "否"}</div></div>
  </div>
  <p class="note">2015-01-11 至 2015-01-17 的 7 天负荷数据在原始 Excel 中为空，正好对应待预测目标；同一时间段的最高温、最低温、平均温、湿度、降雨量均已给出，可用于预测。</p>
</section>

<section>
  <h2>2. 气象因素回归分析</h2>
  <p>先计算各气象因素与三类日负荷指标的 Pearson 相关系数，再用 <code>temp_avg + temp_range + humidity + rainfall + hdd + cdd</code> 建立多元线性回归，观察方向、显著性与解释度。</p>
  <img src="{image_data_uri(OUTPUT_DIR / "01_weather_regression.png")}" alt="气象回归分析">
  <h3>相关性最强的气象因素 Top 5</h3>
  {format_table(corr_rank, "{:.4f}")}
  <h3>多元线性回归摘要</h3>
  {format_table(ols_df[ols_df["term"].isin(REG_WEATHER)].copy(), "{:.4f}")}
  <p class="note">从相关性和回归结果看，温度类变量是主要气象驱动因素；湿度与降雨量的边际贡献较弱。短期预测不能只依赖气象变量，还需要加入负荷滞后项和星期周期来刻画电力负荷惯性。</p>
</section>

<section>
  <h2>3. 预测方法设计</h2>
  <p>为每个目标变量分别训练模型：日最高、日最低、日平均负荷各自使用本目标的 <code>lag_1</code>、<code>lag_7</code>、<code>lag_14</code>、7/14 日滚动均值与标准差，再叠加气象因素和星期/年周期特征。验证集设为 2014-11-01 至 2014-12-31，最终模型在 2012-2014 全量数据上重训。</p>
  <h3>验证集模型表现</h3>
  {format_table(performance_df.sort_values(["target", "RMSE"]), "{:.3f}")}
  <h3>每个目标采用的最优模型</h3>
  {format_table(best_rows, "{:.3f}")}
  <img src="{image_data_uri(OUTPUT_DIR / "02_validation_fit.png")}" alt="验证集拟合">
  <p class="note warn">预测 7 天时采用递推法：2015-01-11 使用 2015-01-10 及以前真实负荷；2015-01-12 起，前一日滞后项会回填上一天预测值，避免把缺失目标期负荷当成已知值。</p>
</section>

<section>
  <h2>4. 2015-01-11 至 2015-01-17 预测结果</h2>
  <img src="{image_data_uri(OUTPUT_DIR / "03_final_prediction.png")}" alt="最终预测">
  <h3>最终预测表</h3>
  {format_table(final_table, "{:.2f}")}
  <img src="{image_data_uri(OUTPUT_DIR / "04_history_forecast_context.png")}" alt="历史趋势衔接">
</section>

<section>
  <h2>5. 结论</h2>
  <p>项目二建议优先使用温度相关特征（平均温、温差、供热度日 HDD、供冷度日 CDD）并结合滞后负荷进行短期预测。气象变量能够解释季节性与冷热负荷变化，滞后变量则补足短期惯性和周周期，因此组合模型比单纯气象回归更适合 7 天短期预测。</p>
</section>
</main>
</body>
</html>
"""
    (OUTPUT_DIR / "project2_report.html").write_text(html_text, encoding="utf-8")


def save_outputs(
    corr_df: pd.DataFrame,
    ols_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> None:
    corr_df.to_csv(OUTPUT_DIR / "project2_weather_correlations.csv", index=False, encoding="utf-8-sig")
    ols_df.to_csv(OUTPUT_DIR / "project2_regression_summary.csv", index=False, encoding="utf-8-sig")
    performance_df.to_csv(OUTPUT_DIR / "project2_model_performance.csv", index=False, encoding="utf-8-sig")
    validation_df.to_csv(OUTPUT_DIR / "project2_validation_predictions.csv", index=False, encoding="utf-8-sig")

    final_csv = pred_df[
        ["date", "weekday", "pred_load_mean", "pred_load_max", "pred_load_min", "model_load_mean", "model_load_max", "model_load_min"]
    ].copy()
    final_csv["date"] = final_csv["date"].dt.strftime("%Y-%m-%d")
    for col in ["pred_load_mean", "pred_load_max", "pred_load_min"]:
        final_csv[col] = final_csv[col].round(2)
    final_csv.columns = ["日期", "星期", "预测日平均(MW)", "预测日最高(MW)", "预测日最低(MW)", "均值模型", "最高模型", "最低模型"]
    final_csv.to_csv(OUTPUT_DIR / "project2_final_prediction_2015_01_11_17.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    daily = load_daily_data()
    train_2012_2014 = daily[(daily["date"] >= "2012-01-01") & (daily["date"] <= "2014-12-31")].copy()

    corr_df, ols_df = build_relationship_tables(train_2012_2014)
    fits, performance_df, validation_df = fit_models(daily)
    pred_df = forecast_recursive(daily, fits)

    save_figures(train_2012_2014, corr_df, ols_df, validation_df, pred_df)
    save_outputs(corr_df, ols_df, performance_df, validation_df, pred_df)
    generate_report(daily, corr_df, ols_df, performance_df, validation_df, pred_df)

    print("Project 2 outputs saved to:", OUTPUT_DIR)
    print((OUTPUT_DIR / "project2_final_prediction_2015_01_11_17.csv").read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    main()
