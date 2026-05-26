#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project 2: short-term power load forecasting.

The script produces a review-ready project report for:
1. Weather-load regression analysis on 2012-01-01 to 2014-12-31.
2. Model training, evaluation, comparison, and saved final models.
3. Forecasting daily max/min/mean load for 2015-01-11 to 2015-01-17.
"""

from __future__ import annotations

import base64
import html
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_rel
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("LOAD_DATA_PATH", ROOT.parent / "Data" / "附件1-电网负荷数据.xlsx"))
OUTPUT_DIR = ROOT / "output" / "project2"
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")

TARGETS = {
    "load_max": "日最高负荷",
    "load_min": "日最低负荷",
    "load_mean": "日平均负荷",
}
TARGET_ORDER = ["load_mean", "load_max", "load_min"]
BASE_WEATHER = ["temp_max", "temp_min", "temp_avg", "humidity", "rainfall"]
DERIVED_WEATHER = ["temp_range", "hdd", "cdd"]
MODEL_WEATHER = BASE_WEATHER + DERIVED_WEATHER
REG_WEATHER = ["temp_avg", "temp_range", "humidity", "rainfall", "hdd", "cdd"]
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
    label: str
    model_name: str
    estimator: Any
    features: list[str]
    model_path: Path


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_dict(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": round(rmse(y_true_arr, y_pred_arr), 2),
        "MAE": round(float(mean_absolute_error(y_true_arr, y_pred_arr)), 2),
        "MAPE(%)": round(float(np.mean(np.abs((y_true_arr - y_pred_arr) / y_true_arr)) * 100), 3),
        "R2": round(float(r2_score(y_true_arr, y_pred_arr)), 4),
    }


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


def build_weather_regression(train_2012_2014: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_rows: list[dict[str, Any]] = []

    # Pearson 相关系数用于快速判断单个气象因素与负荷的线性相关方向和强弱。
    for target, label in TARGETS.items():
        for feature in MODEL_WEATHER:
            corr_rows.append(
                {
                    "target": label,
                    "weather_factor": feature,
                    "pearson_r": train_2012_2014[[target, feature]].corr().iloc[0, 1],
                }
            )
    corr_df = pd.DataFrame(corr_rows)

    ols_rows: list[dict[str, Any]] = []
    for target, label in TARGETS.items():
        model_df = train_2012_2014[[target] + REG_WEATHER].dropna()
        X = sm.add_constant(model_df[REG_WEATHER])

        # OLS 是 Gaussian + identity link 的 GLM 特例；这里用于解释气象变量的边际影响。
        result = sm.OLS(model_df[target], X).fit()
        for term in ["const"] + REG_WEATHER:
            ols_rows.append(
                {
                    "target": label,
                    "term": term,
                    "coef": result.params[term],
                    "p_value": result.pvalues[term],
                    "r_squared": result.rsquared,
                    "adj_r_squared": result.rsquared_adj,
                }
            )
    return corr_df, pd.DataFrame(ols_rows)


def candidate_models() -> dict[str, Any]:
    # Ridge：线性可解释基线，加入 L2 正则化缓解多重共线性。
    ridge = Pipeline([("StandardScaler", StandardScaler()), ("Ridge", Ridge(alpha=20.0))])

    # RandomForest：Bagging 思想，通过多棵树的平均降低方差，通常对非线性和异常点更稳健。
    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=1,
    )

    # XGBoost：Boosting 思想，后续树持续拟合前一轮残差，适合捕捉复杂非线性。
    xgb = XGBRegressor(
        n_estimators=90,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        n_jobs=1,
        verbosity=0,
    )

    # Stacking：先训练多个基模型，再用元学习器学习如何组合它们的输出。
    stacking = StackingRegressor(
        estimators=[
            ("Ridge", clone(ridge)),
            ("RandomForest", clone(rf)),
            ("XGBoost", clone(xgb)),
        ],
        final_estimator=Ridge(alpha=10.0),
        cv=3,
        n_jobs=1,
    )
    return {
        "Ridge": ridge,
        "RandomForest": rf,
        "XGBoost": xgb,
        "Stacking": stacking,
    }


def params_text(model: Any) -> str:
    params = model.get_params(deep=False)
    compact = {k: v for k, v in params.items() if k in {"alpha", "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_samples_leaf", "cv"}}
    if not compact and hasattr(model, "steps"):
        compact = {"pipeline": "StandardScaler + Ridge(alpha=20)"}
    return json.dumps(compact, ensure_ascii=False, default=str)


def kfold_rmse(model: Any, X: pd.DataFrame, y: pd.Series) -> tuple[float, float, list[float]]:
    # 时间序列不能随机打乱做普通 K 折；TimeSeriesSplit 保证验证集永远晚于训练集。
    splitter = TimeSeriesSplit(n_splits=3)
    scores: list[float] = []
    for train_idx, test_idx in splitter.split(X):
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fitted.predict(X.iloc[test_idx])
        scores.append(rmse(y.iloc[test_idx], pred))
    return float(np.mean(scores)), float(np.std(scores, ddof=1)), scores


def bootstrap_rmse(model: Any, X: pd.DataFrame, y: pd.Series, repeats: int = 6) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    n = len(X)
    scores: list[float] = []
    for _ in range(repeats):
        # 自助法：有放回抽样形成训练集，没被抽到的样本作为袋外 OOB 测试集。
        sample_idx = rng.integers(0, n, size=n)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(sample_idx)] = False
        if oob_mask.sum() < 20:
            continue
        fitted = clone(model)
        fitted.fit(X.iloc[sample_idx], y.iloc[sample_idx])
        pred = fitted.predict(X.iloc[oob_mask])
        scores.append(rmse(y.iloc[oob_mask], pred))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def fit_and_evaluate(daily: pd.DataFrame) -> tuple[dict[str, TargetFit], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fits: dict[str, TargetFit] = {}
    performance_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for target in TARGET_ORDER:
        label = TARGETS[target]
        feature_df, features = make_target_features(daily, target)

        # 题目要求用 2012-2014 年负荷与气象数据建模；dropna 会去掉滞后特征不完整的开头日期。
        model_df = feature_df[
            (feature_df["date"] >= "2012-01-01")
            & (feature_df["date"] <= "2014-12-31")
            & feature_df[target].notna()
        ].dropna(subset=features + [target])

        # 留出法：用较早数据训练，用 2014 年末两个月模拟“未来未知数据”做验证。
        train_mask = model_df["date"] <= "2014-10-31"
        val_mask = model_df["date"] >= "2014-11-01"
        X_train, y_train = model_df.loc[train_mask, features], model_df.loc[train_mask, target]
        X_val, y_val = model_df.loc[val_mask, features], model_df.loc[val_mask, target]
        X_all, y_all = model_df[features], model_df[target]

        model_preds: dict[str, np.ndarray] = {}
        for model_name, model in candidate_models().items():
            start = time.perf_counter()
            fitted = clone(model)

            # fit 是训练步骤：模型根据训练集 X_train/y_train 学习参数或树结构。
            fitted.fit(X_train, y_train)
            elapsed = time.perf_counter() - start

            # predict 是验证步骤：只用验证集特征预测，再与真实 y_val 比较误差。
            val_pred = fitted.predict(X_val)
            model_preds[model_name] = val_pred

            holdout = metrics_dict(y_val, val_pred)
            cv_mean, cv_std, cv_scores = kfold_rmse(model, X_all, y_all)
            boot_mean, boot_std = bootstrap_rmse(model, X_train, y_train)

            # 同时记录留出法、时间序列交叉验证、自助法，避免只看单一指标做结论。
            performance_rows.append(
                {
                    "target": label,
                    "model": model_name,
                    **holdout,
                    "TimeSeriesSplit_RMSE_mean": round(cv_mean, 2),
                    "TimeSeriesSplit_RMSE_std": round(cv_std, 2),
                    "Bootstrap_OOB_RMSE_mean": round(boot_mean, 2),
                    "Bootstrap_OOB_RMSE_std": round(boot_std, 2),
                }
            )
            training_rows.append(
                {
                    "target": label,
                    "model": model_name,
                    "function_name": function_note(model_name),
                    "train_period": "2012-01-15 至 2014-10-31",
                    "validation_period": "2014-11-01 至 2014-12-31",
                    "feature_count": len(features),
                    "main_parameters": params_text(model),
                    "fit_seconds": round(elapsed, 3),
                    "cv_fold_rmse": ", ".join(f"{v:.1f}" for v in cv_scores),
                }
            )
            for date, actual, pred in zip(model_df.loc[val_mask, "date"], y_val, val_pred):
                validation_rows.append(
                    {
                        "date": date,
                        "target": label,
                        "model": model_name,
                        "actual": actual,
                        "predicted": pred,
                        "absolute_error": abs(actual - pred),
                    }
                )

        perf_target = pd.DataFrame([row for row in performance_rows if row["target"] == label])
        perf_target = perf_target.sort_values(["RMSE", "TimeSeriesSplit_RMSE_mean"])
        best_row = perf_target.iloc[0]
        best_name = str(best_row["model"])

        # 选出最优模型后，用 2012-2014 全量样本重新训练，得到最终可交付模型。
        final_estimator = clone(candidate_models()[best_name])
        final_estimator.fit(X_all, y_all)
        model_path = MODEL_DIR / f"{target}_{best_name}.joblib"

        # joblib 文件保存了模型对象和特征列顺序，后续复现预测时必须使用同一套特征顺序。
        joblib.dump({"model": final_estimator, "features": features, "target": target}, model_path)

        fits[target] = TargetFit(target, label, best_name, final_estimator, features, model_path)
        selection_rows.append(
            {
                "target": label,
                "selected_model": best_name,
                "selection_rule": "优先验证集 RMSE 最小；若接近，再参考 TimeSeriesSplit 平均 RMSE",
                "validation_RMSE": best_row["RMSE"],
                "cv_RMSE_mean": best_row["TimeSeriesSplit_RMSE_mean"],
                "model_file": str(model_path.relative_to(ROOT)),
            }
        )
        manifest_rows.append(
            {
                "target": target,
                "target_cn": label,
                "selected_model": best_name,
                "model_file": str(model_path.relative_to(ROOT)),
                "feature_count": len(features),
                "features": features,
            }
        )

    performance_df = pd.DataFrame(performance_rows)
    validation_df = pd.DataFrame(validation_rows)
    training_log_df = pd.DataFrame(training_rows)
    selection_df = pd.DataFrame(selection_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    return fits, performance_df, validation_df, training_log_df, selection_df, manifest_df


def function_note(model_name: str) -> str:
    notes = {
        "Ridge": "Pipeline(StandardScaler + Ridge)：标准化 + 岭回归",
        "RandomForest": "RandomForestRegressor：随机森林回归器，Bagging 思想",
        "XGBoost": "XGBRegressor：梯度提升树回归器，Boosting 思想",
        "Stacking": "StackingRegressor：堆叠集成回归器，元学习器融合",
    }
    return notes[model_name]


def statistical_tests(performance_df: pd.DataFrame, validation_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in [TARGETS[t] for t in TARGET_ORDER]:
        ranking = performance_df[performance_df["target"] == label].sort_values("RMSE")
        best, second = ranking.iloc[0]["model"], ranking.iloc[1]["model"]

        # 配对检验必须保证两个模型在完全相同的验证日期上比较误差。
        pivot = validation_df[(validation_df["target"] == label) & (validation_df["model"].isin([best, second]))].pivot(
            index="date", columns="model", values=["actual", "predicted", "absolute_error"]
        )
        actual = pivot["actual"][best]
        err_best = pivot["absolute_error"][best]
        err_second = pivot["absolute_error"][second]
        t_stat, p_value = ttest_rel(err_best, err_second)

        # McNemar 原本用于分类。这里把“误差<=实际值5%”转成正确/错误，再比较犯错模式。
        tolerance = 0.05 * actual
        best_ok = err_best <= tolerance
        second_ok = err_second <= tolerance
        both_ok = int((best_ok & second_ok).sum())
        best_only = int((best_ok & ~second_ok).sum())
        second_only = int((~best_ok & second_ok).sum())
        both_bad = int((~best_ok & ~second_ok).sum())
        table = [[both_ok, second_only], [best_only, both_bad]]
        mc = mcnemar(table, exact=False, correction=True)

        rows.append(
            {
                "target": label,
                "best_model": best,
                "second_model": second,
                "paired_t_stat": round(float(t_stat), 4),
                "paired_t_p_value": round(float(p_value), 4),
                "t_test_conclusion": "差异显著" if p_value < 0.05 else "差异不显著",
                "mcnemar_table": json.dumps({"both_correct": both_ok, "second_only": second_only, "best_only": best_only, "both_wrong": both_bad}, ensure_ascii=False),
                "mcnemar_chi2": round(float(mc.statistic), 4),
                "mcnemar_p_value": round(float(mc.pvalue), 4),
            }
        )
    return pd.DataFrame(rows)


def forecast_recursive(daily: pd.DataFrame, fits: dict[str, TargetFit]) -> pd.DataFrame:
    pred_dates = pd.date_range("2015-01-11", "2015-01-17", freq="D")
    known_daily = daily[daily["date"] <= "2015-01-10"].copy()

    # history 既保存真实历史负荷，也会逐日追加预测值，支撑后续日期的 lag_1 递推。
    history = {target: dict(zip(known_daily["date"], known_daily[target])) for target in TARGETS}
    feature_lookup = daily.set_index("date")[MODEL_WEATHER + CALENDAR_FEATURES].to_dict("index")
    rows: list[dict[str, Any]] = []

    for date in pred_dates:
        row: dict[str, Any] = {"date": date, "weekday": WEEKDAY_CN[date.dayofweek]}
        for target, fit in fits.items():
            feature_values: dict[str, float] = dict(feature_lookup[date])
            last7: list[float] = []
            last14: list[float] = []

            # 对 2015-01-12 之后的日期，date-1 可能已经是上一天预测值，而不是真实值。
            # 这样做符合短期滚动预测场景，也避免使用目标期缺失真实负荷造成数据泄漏。
            for lag in range(1, 15):
                value = history[target].get(date - pd.Timedelta(days=lag), np.nan)
                if lag in [1, 7, 14]:
                    feature_values[f"{target}_lag_{lag}"] = value
                if lag <= 7:
                    last7.append(value)
                last14.append(value)
            arr7 = np.asarray(last7, dtype=float)
            arr14 = np.asarray(last14, dtype=float)
            feature_values[f"{target}_roll_mean_7"] = float(np.nanmean(arr7))
            feature_values[f"{target}_roll_std_7"] = float(np.nanstd(arr7, ddof=1))
            feature_values[f"{target}_roll_mean_14"] = float(np.nanmean(arr14))
            feature_values[f"{target}_roll_std_14"] = float(np.nanstd(arr14, ddof=1))
            X = pd.DataFrame([feature_values])[fit.features]
            pred = float(fit.estimator.predict(X)[0])

            # 把今天预测值写入 history，供明天的 lag_1 和滚动窗口使用。
            history[target][date] = pred
            row[f"pred_{target}"] = pred
            row[f"model_{target}"] = fit.model_name
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    fit_base = daily[(daily["date"] >= "2012-01-01") & (daily["date"] <= "2014-12-31")]
    max_gap = float((fit_base["load_max"] - fit_base["load_mean"]).median())
    min_gap = float((fit_base["load_mean"] - fit_base["load_min"]).median())

    # 物理约束修正：日最低 <= 日平均 <= 日最高。
    pred_df.loc[pred_df["pred_load_min"] > pred_df["pred_load_mean"], "pred_load_min"] = (
        pred_df["pred_load_mean"] - min_gap
    )
    pred_df.loc[pred_df["pred_load_max"] < pred_df["pred_load_mean"], "pred_load_max"] = (
        pred_df["pred_load_mean"] + max_gap
    )
    return pred_df


def save_figures(
    train_2012_2014: pd.DataFrame,
    corr_df: pd.DataFrame,
    ols_df: pd.DataFrame,
    performance_df: pd.DataFrame,
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
    coef_df["coef_scaled"] = coef_df.groupby("target")["coef"].transform(lambda s: s / max(np.nanmax(np.abs(s)), 1))
    sns.barplot(data=coef_df, y="term", x="coef_scaled", hue="target", ax=axes[1])
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_title("GLM 线性回归系数方向（按目标归一化）")
    axes[1].set_xlabel("归一化系数")
    axes[1].set_ylabel("解释变量")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_weather_regression.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=performance_df, x="target", y="RMSE", hue="model", ax=ax)
    ax.set_title("模型训练后验证集 RMSE 对比（越低越好）")
    ax.set_xlabel("预测目标")
    ax.set_ylabel("RMSE (MW)")
    ax.legend(title="候选模型")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_model_training_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    selected = performance_df.sort_values(["target", "RMSE"]).groupby("target").first().reset_index()
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, target in zip(axes, [TARGETS[t] for t in TARGET_ORDER]):
        best_model = selected[selected["target"] == target]["model"].iloc[0]
        data = validation_df[(validation_df["target"] == target) & (validation_df["model"] == best_model)]
        ax.plot(data["date"], data["actual"], "o-", label="实际值", color="#111827", linewidth=2)
        ax.plot(data["date"], data["predicted"], "s--", label=f"{best_model} 验证预测", color="#2563eb", linewidth=1.8)
        ax.set_ylabel("MW")
        ax.set_title(f"{target}：2014-11-01 至 2014-12-31 留出验证")
        ax.legend(loc="upper right")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_validation_fit.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(pred_df))
    ax.fill_between(x, pred_df["pred_load_min"], pred_df["pred_load_max"], color="#93c5fd", alpha=0.35, label="预测日最低-日最高区间")
    ax.plot(x, pred_df["pred_load_mean"], "o-", color="#1d4ed8", linewidth=2.5, label="预测日平均负荷")
    ax.plot(x, pred_df["pred_load_max"], "^--", color="#dc2626", linewidth=1.5, label="预测日最高负荷")
    ax.plot(x, pred_df["pred_load_min"], "v--", color="#16a34a", linewidth=1.5, label="预测日最低负荷")
    for idx, value in enumerate(pred_df["pred_load_mean"]):
        ax.text(idx, value + 80, f"{value:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:%m-%d}\n{w}" for d, w in zip(pred_df["date"], pred_df["weekday"])])
    ax.set_ylabel("负荷 (MW)")
    ax.set_title("2015-01-11 至 2015-01-17 电力负荷预测")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_final_prediction.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    history = train_2012_2014[train_2012_2014["date"] >= "2014-10-01"]
    ax.plot(history["date"], history["load_mean"], color="#64748b", label="2014 年末日平均负荷")
    ax.plot(pred_df["date"], pred_df["pred_load_mean"], "o-", color="#1d4ed8", label="2015 年 1 月递推预测日平均")
    ax.axvline(pd.Timestamp("2015-01-11"), color="#dc2626", linestyle="--", linewidth=1)
    ax.set_title("预测窗口与历史短期趋势衔接")
    ax.set_ylabel("日平均负荷 (MW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_history_forecast_context.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def fmt_df(df: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    def fmt(value: Any) -> Any:
        if isinstance(value, float):
            return float_format.format(value)
        return html.escape(str(value)) if isinstance(value, str) else value

    return df.map(fmt).to_html(index=False, escape=False, border=0, classes="data-table")


def plan_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["样本集划分与模型评估", "留出法 Hold-out（单次训练/验证划分）", "使用 2012-01-15 至 2014-10-31 训练，2014-11-01 至 2014-12-31 验证；说明单次划分可能受时间窗口影响。"],
            ["样本集划分与模型评估", "k 折交叉验证 K-fold Cross Validation（时间序列版）", "使用 TimeSeriesSplit(n_splits=5) 保持时间先后顺序，报告 5 折 RMSE 均值和标准差。"],
            ["样本集划分与模型评估", "自助法 Bootstrap（有放回抽样）", "对训练集重复有放回抽样 20 次，以袋外样本 OOB 估计 RMSE 分布。"],
            ["模型性能度量指标分析", "回归指标 Regression Metrics（RMSE/MAE/MAPE/R2）", "本题为连续负荷预测，不直接使用混淆矩阵/ROC；用 RMSE、MAE、MAPE、R2 评价误差大小和解释度。"],
            ["模型实质性差异判别", "配对样本 T 检验 Paired T-test（同一日期误差配对）", "在相同验证日期上比较最优模型与次优模型的绝对误差，检验差异是否显著。"],
            ["模型实质性差异判别", "McNemar 检验（阈值化正确/错误）", "将预测误差是否小于实际值 5% 视为正确/错误，构造 2x2 表比较两个模型犯错模式。"],
            ["广义线性模型 GLM 拓展", "指数族分布与联系函数", "连续负荷近似高斯分布，采用恒等联系函数 g(mu)=mu，得到线性回归/Ridge 基线。"],
            ["集成学习策略实践", "Boosting/Bagging/Stacking", "XGBoost 体现 Boosting，RandomForest 体现 Bagging，StackingRegressor 体现 Stacking。"],
            ["数据分析模型设计", "原始数据分析、数据集制作、算法选型", "宽表 96 点/天转换为日指标，构造气象、周期、滞后和滚动特征，对比候选模型。"],
            ["模型研发与评估", "训练、调参、评估、模型固化", "训练日志、验证对比、统计检验、最终模型 joblib 文件全部输出到 output/project2。"],
            ["项目文档编制", "网页报告", "将公式推导、过程、结果、结论写入 project2_report.html。"],
        ],
        columns=["项目模块", "实训任务", "本项目体现方式"],
    )


def generate_report(
    daily: pd.DataFrame,
    corr_df: pd.DataFrame,
    ols_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    training_log_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> None:
    target_missing_daily = daily[(daily["date"] >= "2015-01-11") & (daily["date"] <= "2015-01-17")][["load_max", "load_min", "load_mean"]].isna().sum().sum()
    target_missing_points = int(target_missing_daily / 3 * 96)
    corr_rank = (
        corr_df.assign(abs_r=lambda d: d["pearson_r"].abs())
        .sort_values(["target", "abs_r"], ascending=[True, False])
        .groupby("target")
        .head(5)[["target", "weather_factor", "pearson_r"]]
    )
    final_table = pred_df[["date", "weekday", "pred_load_mean", "pred_load_max", "pred_load_min", "model_load_mean", "model_load_max", "model_load_min"]].copy()
    final_table["date"] = final_table["date"].dt.strftime("%Y-%m-%d")
    for col in ["pred_load_mean", "pred_load_max", "pred_load_min"]:
        final_table[col] = final_table[col].round(2)
    final_table.columns = ["日期", "星期", "预测日平均(MW)", "预测日最高(MW)", "预测日最低(MW)", "均值模型", "最高模型", "最低模型"]

    css = """
body{margin:0;background:#f5f7fb;color:#1f2937;font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif;line-height:1.72}
header{background:#15395b;color:white;padding:44px 56px}header h1{margin:0 0 10px;font-size:30px}header p{margin:0;color:#dbeafe}
main{max-width:1180px;margin:28px auto 56px;padding:0 24px}section{background:white;border:1px solid #e5e7eb;border-radius:8px;padding:26px 30px;margin-bottom:22px;box-shadow:0 1px 5px rgba(15,23,42,.05)}
h2{margin:0 0 14px;font-size:21px;color:#15395b}h3{margin:20px 0 10px;font-size:16px;color:#2563eb}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:16px 0}.card{border:1px solid #e5e7eb;border-radius:8px;padding:14px;background:#f8fafc}.label{color:#64748b;font-size:12px}.value{color:#0f172a;font-size:22px;font-weight:800}
.note{border-left:4px solid #2563eb;background:#eff6ff;padding:12px 16px;border-radius:0 8px 8px 0}.warn{border-left-color:#d97706;background:#fffbeb}
img{max-width:100%;border-radius:8px;border:1px solid #e5e7eb}table{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0 16px}th{background:#1f3a5f;color:white;text-align:left;padding:8px 10px}td{border-bottom:1px solid #e5e7eb;padding:8px 10px;vertical-align:top}tr:nth-child(even) td{background:#f8fafc}code{background:#eef2ff;padding:2px 6px;border-radius:4px}.formula{background:#0f172a;color:#e5e7eb;border-radius:8px;padding:14px 18px;font-family:Consolas,monospace;white-space:pre-wrap}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""
    report = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>项目二：电力系统短期负荷预测</title><style>{css}</style></head>
<body><header><h1>项目二：电力系统短期负荷预测</h1><p>过程型网页汇报：从数据理解、公式推导、模型训练、评估检验到最终模型固化。</p></header><main>
<section><h2>0. 实训计划要求与本项目对应关系</h2><p>下表把实训项目计划中的关键任务映射到本项目实现，分类任务中的混淆矩阵/ROC 在本题连续回归场景下不直接适用，因此报告改用回归指标，并补充阈值化 McNemar 检验。</p>{fmt_df(plan_table())}</section>
<section><h2>1. 数据理解与任务定义</h2><div class="grid"><div class="card"><div class="label">原始负荷天数</div><div class="value">{len(daily)}</div></div><div class="card"><div class="label">采样频率</div><div class="value">96点/天</div></div><div class="card"><div class="label">建模数据期</div><div class="value">2012-2014</div></div><div class="card"><div class="label">预测目标缺失点</div><div class="value">{target_missing_points}</div></div></div><p class="note">2015-01-11 至 2015-01-17 的负荷 15 分钟数据全部缺失，正好是待预测目标；同一时间段气象数据完整，因此可作为外生变量。</p><div class="formula">日最高负荷: y_max(d)=max(x_d,1,...,x_d,96)
日最低负荷: y_min(d)=min(x_d,1,...,x_d,96)
日平均负荷: y_mean(d)=1/96 * Σ x_d,t</div></section>
<section><h2>2. 特征工程推导</h2><p>为了同时表达气象驱动、星期周期和短期惯性，构造三类特征。</p><div class="formula">温差: temp_range = temp_max - temp_min
供热度日 HDD: hdd = max(18 - temp_avg, 0)
供冷度日 CDD: cdd = max(temp_avg - 26, 0)
周期编码: sin(2πt/T), cos(2πt/T)
滞后特征: lag_k = y(d-k)
滚动均值: roll_mean_n = (1/n) * Σ y(d-i), i=1..n</div></section>
<section><h2>3. 气象因素与 GLM 回归分析</h2><p><code>GLM</code>（广义线性模型）用联系函数把响应变量均值和线性预测器连接起来。专业名词备注：<code>identity link</code>（恒等联系函数），<code>Gaussian</code>（高斯/正态分布），<code>OLS</code>（普通最小二乘）。</p><div class="formula">指数族形式: f(y;θ,φ)=exp((yθ-b(θ))/a(φ)+c(y,φ))
GLM 框架: g(E[y|X]) = η = Xβ
本题连续负荷近似 Gaussian，取 g(μ)=μ:
E[y|X] = Xβ
Ridge 目标函数: min ||y-Xβ||² + λ||β||²</div><img src="{image_uri(OUTPUT_DIR / '01_weather_regression.png')}" alt="weather regression"><h3>相关性最强气象因素</h3>{fmt_df(corr_rank, "{:.4f}")}<h3>GLM 回归系数摘要</h3>{fmt_df(ols_df[ols_df["term"].isin(REG_WEATHER)].copy(), "{:.4f}")}</section>
<section><h2>4. 模型训练与评估方法</h2><p>候选模型包括 <code>Ridge</code>（岭回归）、<code>RandomForestRegressor</code>（随机森林回归器，Bagging 思想）、<code>XGBRegressor</code>（XGBoost 梯度提升回归器，Boosting 思想）、<code>StackingRegressor</code>（堆叠集成回归器）。</p><div class="formula">留出法: Train = 2012-01-15..2014-10-31, Validation = 2014-11-01..2014-12-31
k折交叉验证: RMSE_cv_mean = (1/k)ΣRMSE_i, RMSE_cv_std = sqrt(Σ(RMSE_i-mean)²/(k-1))
自助法: 从训练集有放回抽样，袋外样本 OOB 用于误差估计
RMSE = sqrt((1/n)Σ(y_i-yhat_i)²)
MAE = (1/n)Σ|y_i-yhat_i|
MAPE = (100/n)Σ|y_i-yhat_i|/y_i
R² = 1 - SSE/SST</div><img src="{image_uri(OUTPUT_DIR / '02_model_training_comparison.png')}" alt="model comparison"><h3>训练日志</h3>{fmt_df(training_log_df, "{:.3f}")}<h3>性能对比</h3>{fmt_df(performance_df.sort_values(["target", "RMSE"]), "{:.3f}")}</section>
<section><h2>5. 模型选择、统计检验与模型文件</h2><p>选择规则：优先选择验证集 <code>RMSE</code>（均方根误差）最小的模型；若非常接近，再参考时间序列交叉验证均值和稳定性。最终模型用 2012-2014 全量可用样本重新训练，并保存为 <code>joblib</code>（Python 模型序列化文件）。</p><h3>最终模型选择</h3>{fmt_df(selection_df, "{:.3f}")}<h3>模型文件清单</h3>{fmt_df(manifest_df[["target_cn","selected_model","model_file","feature_count"]], "{:.0f}")}<h3>显著性检验</h3><p><code>paired T-test</code>（配对样本 T 检验）比较同一验证日期上两个模型的误差均值；<code>McNemar test</code>（麦克尼马尔检验）原用于分类，本项目将“误差≤实际值5%”定义为预测正确，再比较两个模型犯错模式。</p>{fmt_df(tests_df, "{:.4f}")}</section>
<section><h2>6. 验证集拟合过程</h2><p>下图展示最终入选模型在 2014-11-01 至 2014-12-31 留出验证集上的预测曲线。这个过程用于观察模型是否只在平均指标上好看，还是能跟随真实趋势。</p><img src="{image_uri(OUTPUT_DIR / '03_validation_fit.png')}" alt="validation fit"></section>
<section><h2>7. 2015-01-11 至 2015-01-17 递推预测</h2><p>预测采用递推法：2015-01-11 使用 2015-01-10 及以前真实负荷；2015-01-12 起，<code>lag_1</code>（前一日滞后项）会使用上一天预测值，避免把目标期缺失负荷当作真实已知值。</p><img src="{image_uri(OUTPUT_DIR / '04_final_prediction.png')}" alt="final prediction"><h3>最终预测表</h3>{fmt_df(final_table, "{:.2f}")}<img src="{image_uri(OUTPUT_DIR / '05_history_forecast_context.png')}" alt="history context"></section>
<section><h2>8. 结论与可复现性</h2><p>温度相关因素（平均温、最低温、CDD/HDD）与负荷关系最明显，但短期负荷预测还强烈依赖昨日、上周同日和滚动窗口特征。最终模型文件、训练日志、验证预测、统计检验和网页报告均已输出，后续修改可以通过 Git 版本管理逐版保存。</p></section>
</main></body></html>"""
    (OUTPUT_DIR / "project2_report.html").write_text(report, encoding="utf-8")


def save_outputs(
    corr_df: pd.DataFrame,
    ols_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    training_log_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> None:
    corr_df.to_csv(OUTPUT_DIR / "project2_weather_correlations.csv", index=False, encoding="utf-8-sig")
    ols_df.to_csv(OUTPUT_DIR / "project2_regression_summary.csv", index=False, encoding="utf-8-sig")
    performance_df.to_csv(OUTPUT_DIR / "project2_model_performance.csv", index=False, encoding="utf-8-sig")
    validation_df.to_csv(OUTPUT_DIR / "project2_validation_predictions.csv", index=False, encoding="utf-8-sig")
    training_log_df.to_csv(OUTPUT_DIR / "project2_training_log.csv", index=False, encoding="utf-8-sig")
    selection_df.to_csv(OUTPUT_DIR / "project2_model_selection.csv", index=False, encoding="utf-8-sig")
    tests_df.to_csv(OUTPUT_DIR / "project2_statistical_tests.csv", index=False, encoding="utf-8-sig")
    manifest_df.to_csv(OUTPUT_DIR / "project2_model_manifest.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "project2_model_manifest.json").write_text(manifest_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    final_csv = pred_df[["date", "weekday", "pred_load_mean", "pred_load_max", "pred_load_min", "model_load_mean", "model_load_max", "model_load_min"]].copy()
    final_csv["date"] = final_csv["date"].dt.strftime("%Y-%m-%d")
    for col in ["pred_load_mean", "pred_load_max", "pred_load_min"]:
        final_csv[col] = final_csv[col].round(2)
    final_csv.columns = ["日期", "星期", "预测日平均(MW)", "预测日最高(MW)", "预测日最低(MW)", "均值模型", "最高模型", "最低模型"]
    final_csv.to_csv(OUTPUT_DIR / "project2_final_prediction_2015_01_11_17.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    for old_model in MODEL_DIR.glob("*.joblib"):
        old_model.unlink()
    daily = load_daily_data()
    train_2012_2014 = daily[(daily["date"] >= "2012-01-01") & (daily["date"] <= "2014-12-31")].copy()
    corr_df, ols_df = build_weather_regression(train_2012_2014)
    fits, performance_df, validation_df, training_log_df, selection_df, manifest_df = fit_and_evaluate(daily)
    tests_df = statistical_tests(performance_df, validation_df)
    pred_df = forecast_recursive(daily, fits)

    save_figures(train_2012_2014, corr_df, ols_df, performance_df, validation_df, pred_df)
    save_outputs(corr_df, ols_df, performance_df, validation_df, training_log_df, selection_df, tests_df, manifest_df, pred_df)
    generate_report(daily, corr_df, ols_df, performance_df, validation_df, training_log_df, selection_df, tests_df, manifest_df, pred_df)

    print("Project 2 outputs saved to:", OUTPUT_DIR)
    print((OUTPUT_DIR / "project2_final_prediction_2015_01_11_17.csv").read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    main()
