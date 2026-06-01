#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project 2: short-term power load forecasting.

This source-code version keeps the data processing, model training,
evaluation, model saving, chart exporting, and final prediction workflow.
HTML/web report generation is intentionally excluded; webpage presentation
materials are maintained by separate report scripts.
"""

from __future__ import annotations

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
XGB_DEVICE = os.environ.get("PROJECT2_XGB_DEVICE", "cuda")

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
        device=XGB_DEVICE,
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
    compact = {k: v for k, v in params.items() if k in {"alpha", "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_samples_leaf", "cv", "device"}}
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

    print("Project 2 outputs saved to:", OUTPUT_DIR)
    print((OUTPUT_DIR / "project2_final_prediction_2015_01_11_17.csv").read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    main()
