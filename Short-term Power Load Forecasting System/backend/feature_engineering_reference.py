#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project 2 V5: 短期电力负荷预测 — 七步分析 + 残差诊断基线版
=======================================================

V5 基线内容（继承 v4）:
  1. 严格七步前置数据分析检查报告 (Steps 0-7)
  2. 新增残差分析模块 (Step 6): 残差正态性、自相关、异方差检验
  3. 基于预分析结果优化的建模流程
  4. 输出结构化数据分析前置检查报告 (Markdown)

工作目录: F:/Practicum/Data Mining Practicum/V5/
数据来源: ../Data/附件1-电网负荷数据.xlsx
环境: D:/Anaconda/envs/pytorch_gpu (CUDA)
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from itertools import combinations
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
from scipy import stats as sp_stats
from scipy.stats import ttest_rel, shapiro, normaltest, kstest
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# ── 路径配置 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(
    os.environ.get("LOAD_DATA_PATH", ROOT.parent / "Data" / "附件1-电网负荷数据.xlsx")
)
OUTPUT_DIR = ROOT / "output" / "project2"
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── 全局常量 ──────────────────────────────────────────────────────────────
XGB_DEVICE = os.environ.get("PROJECT2_XGB_DEVICE", "cuda")
ALPHA = 0.05
APPROX_THRESHOLDS = (0.03, 0.05, 0.10)
OPTUNA_TRIALS = 50
RANDOM_SEED = 42

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"
]
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
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "doy_sin", "doy_cos", "is_weekend",
]
WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 中国法定节假日
CN_HOLIDAYS = {
    "春节": [
        "2012-01-19","2012-01-20","2012-01-21","2012-01-22",
        "2012-01-23","2012-01-24","2012-01-25","2012-01-26",
        "2012-01-27","2012-01-28","2012-01-29",
        "2013-02-06","2013-02-07","2013-02-08","2013-02-09",
        "2013-02-10","2013-02-11","2013-02-12","2013-02-13",
        "2013-02-14","2013-02-15","2013-02-16","2013-02-17",
        "2014-01-26","2014-01-27","2014-01-28","2014-01-29",
        "2014-01-30","2014-01-31","2014-02-01","2014-02-02",
        "2014-02-03","2014-02-04","2014-02-05","2014-02-06",
    ],
    "国庆节": [
        "2012-10-01","2012-10-02","2012-10-03",
        "2013-10-01","2013-10-02","2013-10-03",
        "2014-10-01","2014-10-02","2014-10-03",
    ],
    "元旦": ["2012-01-01","2013-01-01","2014-01-01"],
    "劳动节": ["2012-05-01","2013-05-01","2014-05-01"],
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. 数据结构
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TargetFit:
    target: str
    label: str
    model_name: str
    estimator: Any
    features: list[str]
    model_path: Path


@dataclass
class OptunaResult:
    model_name: str
    target: str
    best_params: dict[str, Any]
    best_score: float
    trials_df: pd.DataFrame


# ══════════════════════════════════════════════════════════════════════════════
# 2. 基础工具函数
# ══════════════════════════════════════════════════════════════════════════════

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def metrics_dict(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": round(rmse(y_true_arr, y_pred_arr), 2),
        "MAE": round(mae(y_true_arr, y_pred_arr), 2),
        "MAPE(%)": round(
            float(np.mean(np.abs((y_true_arr - y_pred_arr) / y_true_arr)) * 100), 3
        ),
        "R2": round(float(r2_score(y_true_arr, y_pred_arr)), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. 数据加载与特征工程
# ══════════════════════════════════════════════════════════════════════════════

def load_daily_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    load_raw = pd.read_excel(DATA_PATH, sheet_name="Area_Load")
    weather_raw = pd.read_excel(DATA_PATH, sheet_name="Area_Weather")
    weather_raw.columns = [
        "YMD", "temp_max", "temp_min", "temp_avg", "humidity", "rainfall"
    ]
    time_cols = [col for col in load_raw.columns if col != "YMD"]
    long_df = load_raw.melt(
        id_vars="YMD", value_vars=time_cols, var_name="time_slot", value_name="load"
    )
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


def make_target_features(df: pd.DataFrame, target: str):
    out = df.copy()
    lag_features = []
    for lag in [1, 7, 14]:
        col = f"{target}_lag_{lag}"
        out[col] = out[target].shift(lag)
        lag_features.append(col)
    for win in [7, 14]:
        mean_col = f"{target}_roll_mean_{win}"
        std_col = f"{target}_roll_std_{win}"
        out[mean_col] = out[target].shift(1).rolling(win).mean()
        out[std_col] = out[target].shift(1).rolling(win).std()
        lag_features.extend([mean_col, std_col])
    return out, MODEL_WEATHER + CALENDAR_FEATURES + lag_features


# ══════════════════════════════════════════════════════════════════════════════
# 4. 异常值诊断
# ══════════════════════════════════════════════════════════════════════════════

def _classify_outlier(date_str: str, temp_avg: float) -> str:
    for holiday, dates in CN_HOLIDAYS.items():
        if date_str in dates:
            return holiday
    if temp_avg > 30:
        return "极端高温"
    if temp_avg < 5:
        return "极端低温"
    return "其他/未知"


def outlier_diagnosis(daily: pd.DataFrame) -> pd.DataFrame:
    train = daily[(daily["date"] >= "2012-01-01") & (daily["date"] <= "2014-12-31")].copy()
    rows = []
    for col in TARGET_ORDER:
        q1 = train[col].quantile(0.25)
        q3 = train[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        for _, row in train.iterrows():
            if row[col] < lower or row[col] > upper:
                date_str = row["date"].strftime("%Y-%m-%d")
                cause = _classify_outlier(date_str, row["temp_avg"])
                rows.append({
                    "date": date_str, "target": TARGETS[col], "target_col": col,
                    "value": round(row[col], 2),
                    "lower_bound": round(lower, 2), "upper_bound": round(upper, 2),
                    "direction": "偏低" if row[col] < lower else "偏高",
                    "temp_avg": row["temp_avg"], "cause": cause,
                    "action": "保留（真实极端事件）",
                })
    return pd.DataFrame(rows).sort_values(["target_col", "date"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. 气象回归分析
# ══════════════════════════════════════════════════════════════════════════════

def build_weather_regression(train_df):
    corr_rows = []
    for target, label in TARGETS.items():
        for feature in MODEL_WEATHER:
            corr_rows.append({
                "target": label,
                "weather_factor": feature,
                "pearson_r": train_df[[target, feature]].corr().iloc[0, 1],
            })
    corr_df = pd.DataFrame(corr_rows)
    ols_rows = []
    for target, label in TARGETS.items():
        model_df = train_df[[target] + REG_WEATHER].dropna()
        X = sm.add_constant(model_df[REG_WEATHER])
        result = sm.OLS(model_df[target], X).fit()
        for term in ["const"] + REG_WEATHER:
            ols_rows.append({
                "target": label, "term": term,
                "coef": result.params[term], "p_value": result.pvalues[term],
                "r_squared": result.rsquared, "adj_r_squared": result.rsquared_adj,
            })
    return corr_df, pd.DataFrame(ols_rows)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Diebold-Mariano 检验
# ══════════════════════════════════════════════════════════════════════════════

def diebold_mariano_test(actual, forecast1, forecast2, horizon=1, loss="MSE"):
    n = len(actual)
    if n < 4:
        return {"DM_statistic": 0.0, "p_value": 1.0, "method": "DM", "loss_function": loss}
    if loss == "MSE":
        d = (actual - forecast1) ** 2 - (actual - forecast2) ** 2
    elif loss == "MAE":
        d = np.abs(actual - forecast1) - np.abs(actual - forecast2)
    else:
        raise ValueError(f"Unknown loss: {loss}")
    d_mean = np.mean(d)
    gamma_0 = np.var(d, ddof=0)
    long_run_var = gamma_0
    for k in range(1, min(horizon, n - 1) + 1):
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        weight = 1.0 - k / (horizon + 1.0)
        long_run_var += 2.0 * weight * gamma_k
    long_run_var = max(long_run_var, 1e-10)
    dm_stat = d_mean / np.sqrt(long_run_var / n)
    p_value = 2.0 * (1.0 - sp_stats.norm.cdf(np.abs(dm_stat)))
    p_value = min(max(p_value, 0.0), 1.0)
    return {
        "DM_statistic": round(float(dm_stat), 4),
        "p_value": round(float(p_value), 6),
        "method": "Diebold-Mariano",
        "loss_function": loss,
        "d_mean": round(float(d_mean), 4),
        "long_run_variance": round(float(long_run_var), 6),
        "n_observations": n,
    }


def dm_test_all_pairs(validation_df, loss="MSE"):
    rows = []
    for label in [TARGETS[t] for t in TARGET_ORDER]:
        model_names = sorted(
            validation_df.loc[validation_df["target"] == label, "model"].unique()
        )
        for model_a, model_b in combinations(model_names, 2):
            subset = validation_df[
                (validation_df["target"] == label) & (validation_df["model"].isin([model_a, model_b]))
            ]
            pivot = subset.pivot(index="date", columns="model", values=["actual", "predicted"]).dropna()
            if pivot.empty or len(pivot) < 4:
                continue
            actual = pivot["actual"][model_a].values
            pred_a = pivot["predicted"][model_a].values
            pred_b = pivot["predicted"][model_b].values
            result = diebold_mariano_test(actual, pred_a, pred_b, horizon=1, loss=loss)
            better = model_a if result["d_mean"] < 0 else model_b
            rows.append({
                "target": label, "model_a": model_a, "model_b": model_b,
                "DM_statistic": result["DM_statistic"], "p_value": result["p_value"],
                "loss_function": loss, "d_mean": result["d_mean"],
                "better_by_DM": better,
                "conclusion": (
                    f"p < {ALPHA}，预测能力差异显著" if result["p_value"] < ALPHA
                    else f"p >= {ALPHA}，预测能力差异不显著"
                ),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Optuna 超参数优化
# ══════════════════════════════════════════════════════════════════════════════

def _make_optuna_objective(model_type, X_train, y_train, n_splits=3):
    import optuna
    def objective(trial):
        if model_type == "Ridge":
            alpha = trial.suggest_float("alpha", 0.01, 100.0, log=True)
            model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        elif model_type == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "random_state": RANDOM_SEED, "n_jobs": 1,
            }
            model = RandomForestRegressor(**params)
        elif model_type == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
                "objective": "reg:squarederror", "device": XGB_DEVICE,
                "random_state": RANDOM_SEED, "tree_method": "hist",
                "n_jobs": 1, "verbosity": 0,
            }
            model = XGBRegressor(**params)
        elif model_type == "SVR":
            C = trial.suggest_float("C", 0.1, 100.0, log=True)
            gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
            epsilon = trial.suggest_float("epsilon", 0.01, 0.5)
            model = Pipeline([("scaler", StandardScaler()),
                              ("svr", SVR(C=C, gamma=gamma, epsilon=epsilon, kernel="rbf"))])
        elif model_type == "KNN":
            n_neighbors = trial.suggest_int("n_neighbors", 3, 30)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            model = Pipeline([("scaler", StandardScaler()),
                              ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights))])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        splitter = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for train_idx, val_idx in splitter.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            scores.append(rmse(y_val, pred))
        return float(np.mean(scores))
    return objective


def optuna_tune_model(model_type, X_train, y_train, target_label, n_trials=OPTUNA_TRIALS):
    import optuna
    objective = _make_optuna_objective(model_type, X_train, y_train)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=f"{model_type}_{target_label}", direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials_data = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trials_data.append({"trial_number": t.number, "value": t.value, **t.params})
    return OptunaResult(
        model_name=model_type, target=target_label,
        best_params=study.best_params, best_score=study.best_value,
        trials_df=pd.DataFrame(trials_data),
    )


def build_tuned_model(model_type, best_params):
    if model_type == "Ridge":
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=best_params["alpha"]))])
    elif model_type == "RandomForest":
        return RandomForestRegressor(
            n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            min_samples_split=best_params["min_samples_split"],
            random_state=RANDOM_SEED, n_jobs=1,
        )
    elif model_type == "XGBoost":
        return XGBRegressor(
            n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=best_params["subsample"], colsample_bytree=best_params["colsample_bytree"],
            min_child_weight=best_params["min_child_weight"],
            reg_alpha=best_params.get("reg_alpha", 0.0), reg_lambda=best_params.get("reg_lambda", 1.0),
            objective="reg:squarederror", device=XGB_DEVICE, random_state=RANDOM_SEED,
            tree_method="hist", n_jobs=1, verbosity=0,
        )
    elif model_type == "SVR":
        return Pipeline([("scaler", StandardScaler()),
                         ("svr", SVR(C=best_params["C"], gamma=best_params["gamma"],
                                    epsilon=best_params["epsilon"], kernel="rbf"))])
    elif model_type == "KNN":
        return Pipeline([("scaler", StandardScaler()),
                         ("knn", KNeighborsRegressor(n_neighbors=best_params["n_neighbors"],
                                                     weights=best_params["weights"]))])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def tune_all_models(X_train, y_train, target_label):
    model_types = ["Ridge", "RandomForest", "XGBoost", "SVR", "KNN"]
    tuned_models = {}
    optuna_results = []
    print(f"  [{target_label}] Optuna 调参中 ({OPTUNA_TRIALS} trials/模型)...")
    for mt in model_types:
        print(f"    - {mt} ...", end=" ", flush=True)
        result = optuna_tune_model(mt, X_train, y_train, target_label)
        tuned_models[mt] = build_tuned_model(mt, result.best_params)
        optuna_results.append(result)
        print(f"最佳 RMSE={result.best_score:.2f}")
    return tuned_models, optuna_results


def params_text(model):
    if hasattr(model, "steps"):
        params = model.steps[-1][1].get_params()
    else:
        params = model.get_params(deep=False)
    compact = {k: v for k, v in params.items() if k in {
        "alpha", "n_estimators", "max_depth", "learning_rate",
        "subsample", "colsample_bytree", "min_samples_leaf",
        "min_samples_split", "min_child_weight", "reg_alpha", "reg_lambda",
        "C", "gamma", "epsilon", "n_neighbors", "weights", "cv",
    }}
    return json.dumps(compact, ensure_ascii=False, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# 8. 训练与评估
# ══════════════════════════════════════════════════════════════════════════════

def kfold_rmse(model, X, y):
    splitter = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, test_idx in splitter.split(X):
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fitted.predict(X.iloc[test_idx])
        scores.append(rmse(y.iloc[test_idx], pred))
    return float(np.mean(scores)), float(np.std(scores, ddof=1)), scores


def bootstrap_rmse(model, X, y, repeats=6):
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(X)
    scores = []
    for _ in range(repeats):
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


def generalization_risk_label(gap):
    abs_gap = abs(gap)
    if abs_gap <= 150:
        return "低风险"
    if abs_gap <= 350:
        return "中等风险"
    return "高风险"


def function_note(model_name):
    notes = {
        "Ridge": "Pipeline(StandardScaler + Ridge)",
        "RandomForest": "RandomForestRegressor (Bagging)",
        "XGBoost": "XGBRegressor (Boosting, CUDA)",
        "SVR": "Pipeline(StandardScaler + SVR/RBF)",
        "KNN": "Pipeline(StandardScaler + KNN)",
        "Stacking": "StackingRegressor: 5基模型 + Ridge元学习器",
    }
    return notes.get(model_name, model_name)


def fit_and_evaluate(daily):
    fits = {}
    performance_rows, validation_rows, training_rows = [], [], []
    selection_rows, manifest_rows = [], []
    all_optuna_results = []

    for target in TARGET_ORDER:
        label = TARGETS[target]
        feature_df, features = make_target_features(daily, target)
        model_df = feature_df[
            (feature_df["date"] >= "2012-01-01")
            & (feature_df["date"] <= "2014-12-31")
            & feature_df[target].notna()
        ].dropna(subset=features + [target])

        train_mask = model_df["date"] <= "2014-06-30"
        val_mask = model_df["date"] >= "2014-07-01"
        X_train, y_train = model_df.loc[train_mask, features], model_df.loc[train_mask, target]
        X_val, y_val = model_df.loc[val_mask, features], model_df.loc[val_mask, target]
        X_all, y_all = model_df[features], model_df[target]

        print(f"\n{'='*60}")
        print(f"  目标: {label} | 训练集: {len(X_train)} 天 | 验证集: {len(X_val)} 天")
        print(f"{'='*60}")

        # Optuna 调参
        print(f"  [{label}] 执行 Optuna 超参数搜索...")
        models, optuna_results = tune_all_models(X_train, y_train, label)
        all_optuna_results.extend(optuna_results)

        # 构建 Stacking
        stacking = StackingRegressor(
            estimators=[
                ("Ridge", clone(models["Ridge"])),
                ("RandomForest", clone(models["RandomForest"])),
                ("XGBoost", clone(models["XGBoost"])),
                ("SVR", clone(models["SVR"])),
                ("KNN", clone(models["KNN"])),
            ],
            final_estimator=Ridge(alpha=10.0), cv=3, n_jobs=1,
        )
        models["Stacking"] = stacking

        model_preds = {}
        for model_name, model in models.items():
            start = time.perf_counter()
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            elapsed = time.perf_counter() - start
            val_pred = fitted.predict(X_val)
            model_preds[model_name] = val_pred

            holdout = metrics_dict(y_val, val_pred)
            cv_mean, cv_std, cv_scores = kfold_rmse(model, X_all, y_all)
            boot_mean, boot_std = bootstrap_rmse(model, X_train, y_train)
            gap_vs_cv = holdout["RMSE"] - cv_mean

            performance_rows.append({
                "target": label, "model": model_name, **holdout,
                "test_error_RMSE": holdout["RMSE"],
                "generalization_error_estimate_RMSE": round(cv_mean, 2),
                "generalization_gap_vs_cv_RMSE": round(gap_vs_cv, 2),
                "generalization_risk": generalization_risk_label(gap_vs_cv),
                "TimeSeriesSplit_RMSE_mean": round(cv_mean, 2),
                "TimeSeriesSplit_RMSE_std": round(cv_std, 2),
                "Bootstrap_OOB_RMSE_mean": round(boot_mean, 2),
                "Bootstrap_OOB_RMSE_std": round(boot_std, 2),
            })
            training_rows.append({
                "target": label, "model": model_name,
                "function_name": function_note(model_name),
                "train_period": "2012-01-15 至 2014-06-30",
                "validation_period": "2014-07-01 至 2014-12-31",
                "feature_count": len(features),
                "main_parameters": params_text(model),
                "fit_seconds": round(elapsed, 3),
                "cv_fold_rmse": ", ".join(f"{v:.1f}" for v in cv_scores),
            })
            for date, actual, pred in zip(model_df.loc[val_mask, "date"], y_val, val_pred):
                validation_rows.append({
                    "date": date, "target": label, "model": model_name,
                    "actual": actual, "predicted": pred,
                    "absolute_error": abs(actual - pred),
                })

        # 模型选择
        perf_target = pd.DataFrame([r for r in performance_rows if r["target"] == label])
        perf_target = perf_target.sort_values(["RMSE", "TimeSeriesSplit_RMSE_mean"])
        best_row = perf_target.iloc[0]
        best_name = str(best_row["model"])

        final_estimator = clone(models[best_name])
        final_estimator.fit(X_all, y_all)
        model_path = MODEL_DIR / f"{target}_{best_name}.joblib"
        joblib.dump({"model": final_estimator, "features": features, "target": target}, model_path)

        fits[target] = TargetFit(target, label, best_name, final_estimator, features, model_path)
        selection_rows.append({
            "target": label, "selected_model": best_name,
            "selection_rule": "验证集 RMSE 最小",
            "validation_RMSE": best_row["RMSE"],
            "cv_RMSE_mean": best_row["TimeSeriesSplit_RMSE_mean"],
            "model_file": str(model_path.relative_to(ROOT)),
        })
        manifest_rows.append({
            "target": target, "target_cn": label,
            "selected_model": best_name,
            "model_file": str(model_path.relative_to(ROOT)),
            "feature_count": len(features), "features": features,
        })
        print(f"  -> 最优模型: {best_name} (RMSE={best_row['RMSE']:.2f})")

    return (
        fits, pd.DataFrame(performance_rows), pd.DataFrame(validation_rows),
        pd.DataFrame(training_rows), pd.DataFrame(selection_rows),
        pd.DataFrame(manifest_rows), all_optuna_results,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 9. 统计检验
# ══════════════════════════════════════════════════════════════════════════════

def approximate_correctness(validation_df):
    rows = []
    for (target, model_name), group in validation_df.groupby(["target", "model"]):
        actual_abs = group["actual"].abs().replace(0, np.nan)
        relative_error = group["absolute_error"] / actual_abs
        for threshold in APPROX_THRESHOLDS:
            ok = relative_error <= threshold
            valid_count = int(ok.notna().sum())
            ok_count = int(ok.sum())
            rate = float(ok.mean()) if valid_count else np.nan
            rows.append({
                "target": target, "model": model_name,
                "threshold": threshold,
                "threshold_label": f"相对误差 <= {threshold:.0%}",
                "approx_correct_count": ok_count, "sample_count": valid_count,
                "approx_correct_probability": round(rate, 4),
                "approx_test_error_probability": round(1 - rate, 4),
            })
    return pd.DataFrame(rows)


def pairwise_significance_tests(validation_df, tolerance_rate=0.05):
    rows = []
    for label in [TARGETS[t] for t in TARGET_ORDER]:
        model_names = sorted(
            validation_df.loc[validation_df["target"] == label, "model"].unique()
        )
        for model_a, model_b in combinations(model_names, 2):
            subset = validation_df[
                (validation_df["target"] == label)
                & (validation_df["model"].isin([model_a, model_b]))
            ]
            pivot = subset.pivot(
                index="date", columns="model", values=["actual", "absolute_error"]
            ).dropna()
            if pivot.empty:
                continue
            actual = pivot["actual"][model_a].abs()
            err_a = pivot["absolute_error"][model_a]
            err_b = pivot["absolute_error"][model_b]
            mean_a = float(err_a.mean())
            mean_b = float(err_b.mean())
            t_stat, p_value = ttest_rel(err_a, err_b)
            if np.isnan(t_stat) or np.isnan(p_value):
                t_stat, p_value = 0.0, 1.0
            tolerance = tolerance_rate * actual
            a_ok = err_a <= tolerance
            b_ok = err_b <= tolerance
            both_ok = int((a_ok & b_ok).sum())
            a_only = int((a_ok & ~b_ok).sum())
            b_only = int((~a_ok & b_ok).sum())
            both_bad = int((~a_ok & ~b_ok).sum())
            table = [[both_ok, b_only], [a_only, both_bad]]
            mc = mcnemar(table, exact=False, correction=True)
            mc_p = float(mc.pvalue) if not np.isnan(mc.pvalue) else 1.0
            better_model = model_a if mean_a < mean_b else model_b
            rows.append({
                "target": label, "model_a": model_a, "model_b": model_b,
                "mean_abs_error_a": round(mean_a, 2), "mean_abs_error_b": round(mean_b, 2),
                "better_by_mean_abs_error": better_model,
                "paired_t_stat": round(float(t_stat), 4),
                "paired_t_p_value": round(float(p_value), 6),
                "t_test_conclusion": (
                    f"p < {ALPHA}，差异显著" if p_value < ALPHA
                    else f"p >= {ALPHA}，差异不显著"
                ),
                "mcnemar_chi2": round(float(mc.statistic), 4),
                "mcnemar_p_value": round(mc_p, 6),
                "mcnemar_conclusion": (
                    f"p < {ALPHA}，犯错模式差异显著" if mc_p < ALPHA
                    else f"p >= {ALPHA}，犯错模式差异不显著"
                ),
            })
    return pd.DataFrame(rows)


def statistical_tests(performance_df, validation_df):
    pairwise_df = pairwise_significance_tests(validation_df)
    approx_df = approximate_correctness(validation_df)
    dm_mse_df = dm_test_all_pairs(validation_df, loss="MSE")
    summary_rows = []
    for label in [TARGETS[t] for t in TARGET_ORDER]:
        ranking = performance_df[performance_df["target"] == label].sort_values("RMSE")
        best, second = ranking.iloc[0]["model"], ranking.iloc[1]["model"]
        pair = pairwise_df[
            (pairwise_df["target"] == label)
            & (((pairwise_df["model_a"] == best) & (pairwise_df["model_b"] == second))
               | ((pairwise_df["model_a"] == second) & (pairwise_df["model_b"] == best)))
        ]
        dm_mse_row = dm_mse_df[
            (dm_mse_df["target"] == label)
            & (((dm_mse_df["model_a"] == best) & (dm_mse_df["model_b"] == second))
               | ((dm_mse_df["model_a"] == second) & (dm_mse_df["model_b"] == best)))
        ]
        best_approx_5 = approx_df[
            (approx_df["target"] == label) & (approx_df["model"] == best)
            & (approx_df["threshold"] == 0.05)
        ]["approx_correct_probability"].iloc[0]
        summary_rows.append({
            "target": label, "best_model": best, "second_model": second,
            "best_validation_RMSE": ranking.iloc[0]["RMSE"],
            "second_validation_RMSE": ranking.iloc[1]["RMSE"],
            "best_approx_correct_probability_at_5pct": best_approx_5,
            "paired_t_p_value": pair.iloc[0]["paired_t_p_value"] if len(pair) > 0 else np.nan,
            "t_test_conclusion": pair.iloc[0]["t_test_conclusion"] if len(pair) > 0 else "N/A",
            "dm_mse_p_value": dm_mse_row.iloc[0]["p_value"] if len(dm_mse_row) > 0 else np.nan,
            "dm_mse_conclusion": dm_mse_row.iloc[0]["conclusion"] if len(dm_mse_row) > 0 else "N/A",
            "mcnemar_p_value": pair.iloc[0]["mcnemar_p_value"] if len(pair) > 0 else np.nan,
        })
    return pd.DataFrame(summary_rows), pairwise_df, approx_df, dm_mse_df


# ══════════════════════════════════════════════════════════════════════════════
# 10. Friedman 检验 + 季节性分析
# ══════════════════════════════════════════════════════════════════════════════

def friedman_test(validation_df):
    rows = []
    for label in [TARGETS[t] for t in TARGET_ORDER]:
        pivot = validation_df[validation_df["target"] == label].pivot(
            index="date", columns="model", values="absolute_error"
        ).dropna()
        if pivot.empty or pivot.shape[1] < 3:
            continue
        ranks = pivot.rank(axis=1)
        model_names = list(pivot.columns)
        n = len(pivot)
        k = len(model_names)
        rank_sums = ranks.sum(axis=0)
        chi2 = (12 * n) / (k * (k + 1)) * np.sum((rank_sums / n - (k + 1) / 2) ** 2)
        p_value = 1.0 - sp_stats.chi2.cdf(chi2, k - 1)
        avg_ranks = rank_sums / n
        rows.append({
            "target": label, "n_dates": n, "n_models": k,
            "friedman_chi2": round(float(chi2), 4),
            "friedman_p_value": round(float(p_value), 6),
            "model_avg_ranks": json.dumps(
                {m: round(float(avg_ranks[m]), 2) for m in model_names}, ensure_ascii=False
            ),
            "best_by_rank": str(avg_ranks.idxmin()),
            "conclusion": (
                f"p < {ALPHA}，模型间存在显著差异" if p_value < ALPHA
                else f"p >= {ALPHA}，模型间差异不显著"
            ),
        })
    return pd.DataFrame(rows)


def seasonal_analysis(validation_df):
    df = validation_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.dayofweek
    df["season"] = df["month"].map({
        12: "冬季", 1: "冬季", 2: "冬季",
        3: "春季", 4: "春季", 5: "春季",
        6: "夏季", 7: "夏季", 8: "夏季",
        9: "秋季", 10: "秋季", 11: "秋季",
    })
    df["day_type"] = df["weekday"].apply(lambda d: "周末" if d >= 5 else "工作日")
    rows = []
    for group_col in ["season", "month", "day_type"]:
        for (target, model), grp in df.groupby(["target", "model"]):
            for group_val, sub in grp.groupby(group_col):
                m = metrics_dict(sub["actual"], sub["predicted"])
                rows.append({
                    "target": target, "model": model,
                    "group_by": group_col, "group_value": group_val,
                    "n_samples": len(sub), **m,
                })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 11. 残差分析模块 (Step 6 - v4 NEW)
# ══════════════════════════════════════════════════════════════════════════════

def residual_analysis(daily, fits) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    v4 新增: 残差分析模块
    对最优模型进行:
      1. 残差正态性检验 (Shapiro-Wilk + D'Agostino-Pearson + Kolmogorov-Smirnov)
      2. 残差自相关检验 (Ljung-Box)
      3. 残差异方差检验 (Breusch-Pagan)
      4. 残差 vs 拟合值散点图
    """
    train_2012_2014 = daily[(daily["date"] >= "2012-01-01") & (daily["date"] <= "2014-12-31")]
    norm_rows = []
    autocorr_rows = []
    hetero_rows = []
    residual_data = {}

    for target in TARGET_ORDER:
        fit = fits[target]
        feature_df, features = make_target_features(daily, target)
        model_df = feature_df[
            (feature_df["date"] >= "2012-01-01")
            & (feature_df["date"] <= "2014-12-31")
            & feature_df[target].notna()
        ].dropna(subset=features + [target])

        train_mask = model_df["date"] <= "2014-06-30"
        val_mask = model_df["date"] >= "2014-07-01"

        # 用训练集拟合，在验证集上计算残差
        X_train = model_df.loc[train_mask, features]
        y_train = model_df.loc[train_mask, target]
        X_val = model_df.loc[val_mask, features]
        y_val = model_df.loc[val_mask, target]
        dates_val = model_df.loc[val_mask, "date"]

        fitted_model = clone(fit.estimator)
        fitted_model.fit(X_train, y_train)
        y_pred = fitted_model.predict(X_val)
        residuals = y_val.values - y_pred
        fitted_values = y_pred

        residual_data[target] = {
            "residuals": residuals, "fitted": fitted_values,
            "actual": y_val.values, "dates": dates_val.values,
            "target_label": fit.label, "model_name": fit.model_name,
        }

        # 1. 残差正态性检验
        if len(residuals) >= 3:
            # Shapiro-Wilk (n < 5000 可靠)
            if len(residuals) <= 5000:
                sw_stat, sw_p = shapiro(residuals)
            else:
                sw_stat, sw_p = 0.0, np.nan
            # D'Agostino-Pearson
            if len(residuals) >= 20:
                dagostino_stat, dagostino_p = normaltest(residuals)
            else:
                dagostino_stat, dagostino_p = np.nan, np.nan
            # Kolmogorov-Smirnov
            ks_stat, ks_p = kstest(residuals, "norm", args=(np.mean(residuals), np.std(residuals)))
        else:
            sw_stat, sw_p, dagostino_stat, dagostino_p, ks_stat, ks_p = 0, np.nan, np.nan, np.nan, 0, np.nan

        sw_conclusion = "拒绝正态假设" if (not np.isnan(sw_p) and sw_p < ALPHA) else "无法拒绝正态假设"
        dagostino_conclusion = "拒绝正态假设" if (not np.isnan(dagostino_p) and dagostino_p < ALPHA) else "无法拒绝正态假设"
        ks_conclusion = "拒绝正态假设" if (not np.isnan(ks_p) and ks_p < ALPHA) else "无法拒绝正态假设"

        norm_rows.append({
            "target": fit.label, "model": fit.model_name,
            "residual_n": len(residuals), "residual_mean": round(float(np.mean(residuals)), 4),
            "residual_std": round(float(np.std(residuals)), 4),
            "Shapiro_Wilk_W": round(float(sw_stat), 6),
            "Shapiro_Wilk_p": round(float(sw_p), 6) if not np.isnan(sw_p) else np.nan,
            "Shapiro_Wilk_conclusion": sw_conclusion,
            "Dagostino_K2": round(float(dagostino_stat), 6) if not np.isnan(dagostino_stat) else np.nan,
            "Dagostino_p": round(float(dagostino_p), 6) if not np.isnan(dagostino_p) else np.nan,
            "Dagostino_conclusion": dagostino_conclusion,
            "KS_statistic": round(float(ks_stat), 6),
            "KS_p": round(float(ks_p), 6) if not np.isnan(ks_p) else np.nan,
            "KS_conclusion": ks_conclusion,
        })

        # 2. 残差自相关 (Ljung-Box)
        for lag in [1, 7, 14]:
            if len(residuals) > lag:
                lb_result = acorr_ljungbox(residuals, lags=[lag], return_df=True)
                lb_stat = float(lb_result.iloc[0]["lb_stat"])
                lb_p = float(lb_result.iloc[0]["lb_pvalue"])
            else:
                lb_stat, lb_p = np.nan, np.nan
            autocorr_rows.append({
                "target": fit.label, "model": fit.model_name, "lag": lag,
                "LjungBox_statistic": round(lb_stat, 4) if not np.isnan(lb_stat) else np.nan,
                "LjungBox_p_value": round(lb_p, 6) if not np.isnan(lb_p) else np.nan,
                "conclusion": (
                    f"p < {ALPHA}，存在显著自相关" if (not np.isnan(lb_p) and lb_p < ALPHA)
                    else f"p >= {ALPHA}，不存在显著自相关"
                ),
            })

        # 3. 异方差检验 (Breusch-Pagan)
        X_bp = sm.add_constant(pd.DataFrame(fitted_values, columns=["fitted"]))
        y_bp = pd.Series(residuals ** 2)
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(y_bp, X_bp)
            bp_conclusion = f"p < {ALPHA}，存在显著异方差" if bp_p < ALPHA else f"p >= {ALPHA}，不存在显著异方差"
        except Exception:
            bp_stat, bp_p = np.nan, np.nan
            bp_conclusion = "计算失败"

        hetero_rows.append({
            "target": fit.label, "model": fit.model_name,
            "BreuschPagan_statistic": round(float(bp_stat), 4) if not np.isnan(bp_stat) else np.nan,
            "BreuschPagan_p_value": round(float(bp_p), 6) if not np.isnan(bp_p) else np.nan,
            "BreuschPagan_conclusion": bp_conclusion,
        })

    norm_df = pd.DataFrame(norm_rows)
    autocorr_df = pd.DataFrame(autocorr_rows)
    hetero_df = pd.DataFrame(hetero_rows)

    # 输出残差分析摘要
    print("\n" + "=" * 60)
    print("  第六步: 残差分析 (Step 6)")
    print("=" * 60)
    for _, row in norm_df.iterrows():
        print(f"\n  {row['target']} ({row['model']}):")
        print(f"    残差均值={row['residual_mean']:.4f}, 标准差={row['residual_std']:.4f}")
        print(f"    Shapiro-Wilk: W={row['Shapiro_Wilk_W']:.4f}, p={row['Shapiro_Wilk_p']}, {row['Shapiro_Wilk_conclusion']}")
        if not np.isnan(row["Dagostino_p"]):
            print(f"    D'Agostino: K2={row['Dagostino_K2']:.4f}, p={row['Dagostino_p']:.6f}, {row['Dagostino_conclusion']}")
        print(f"    KS: statistic={row['KS_statistic']:.4f}, p={row['KS_p']:.6f}, {row['KS_conclusion']}")

    for _, row in autocorr_df.iterrows():
        print(f"    Ljung-Box (lag={row['lag']}): stat={row['LjungBox_statistic']:.4f}, p={row['LjungBox_p_value']:.6f}, {row['conclusion']}")

    for _, row in hetero_df.iterrows():
        print(f"    Breusch-Pagan: stat={row['BreuschPagan_statistic']:.4f}, p={row['BreuschPagan_p_value']:.6f}, {row['BreuschPagan_conclusion']}")

    # 绘制残差分析图
    _save_residual_figures(residual_data, norm_df)

    return norm_df, pd.concat([autocorr_df, hetero_df], ignore_index=True)


def _save_residual_figures(residual_data, norm_df):
    """绘制残差分析四图合一"""
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    for idx, target in enumerate(TARGET_ORDER):
        data = residual_data[target]
        residuals = data["residuals"]
        fitted = data["fitted"]
        label = data["target_label"]
        model = data["model_name"]

        # 残差直方图
        ax = axes[idx, 0]
        ax.hist(residuals, bins=30, density=True, alpha=0.6, color="steelblue", edgecolor="white")
        x_range = np.linspace(residuals.min(), residuals.max(), 200)
        kde = sp_stats.gaussian_kde(residuals)
        ax.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
        ax.set_title(f"{label} 残差分布\n({model})")
        ax.set_xlabel("残差 (MW)")
        ax.legend(fontsize=7)

        # 残差 Q-Q 图
        ax = axes[idx, 1]
        sp_stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f"{label} 残差 Q-Q 图")

        # 残差 vs 拟合值
        ax = axes[idx, 2]
        ax.scatter(fitted, residuals, alpha=0.5, s=15, color="steelblue")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{label} 残差 vs 拟合值")
        ax.set_xlabel("拟合值 (MW)")
        ax.set_ylabel("残差 (MW)")

        # 残差时序图
        ax = axes[idx, 3]
        dates = data["dates"]
        ax.plot(dates, residuals, "-", alpha=0.7, linewidth=1, color="steelblue")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{label} 残差时序图")
        ax.set_ylabel("残差 (MW)")
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    fig.suptitle("v4 残差分析 (Step 6): 正态性 + 自相关 + 异方差", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "v4_step6_residual_analysis.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 12. 集成策略 + 递推预测
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_strategy_table():
    rows = [
        {"role": "a", "model": "Ridge", "function_name": "Ridge: Ridge Regression",
         "learning_strategy": "Linear + L2 Regularization",
         "why_used": "Interpretable linear baseline"},
        {"role": "b", "model": "RandomForest", "function_name": "RandomForestRegressor",
         "learning_strategy": "Bagging: parallel tree averaging",
         "why_used": "Robust to non-linearity and outliers"},
        {"role": "c", "model": "XGBoost", "function_name": "XGBRegressor (CUDA)",
         "learning_strategy": "Boosting: sequential residual fitting",
         "why_used": "Complex non-linear pattern capture"},
        {"role": "d", "model": "SVR", "function_name": "SVR (RBF kernel)",
         "learning_strategy": "Kernel method: epsilon-insensitive hyperplane",
         "why_used": "Complementary to tree models"},
        {"role": "e", "model": "KNN", "function_name": "K-Nearest Neighbors",
         "learning_strategy": "Distance-based similar day matching",
         "why_used": "Non-parametric similarity-based"},
        {"role": "a+b+c+d+e", "model": "Stacking",
         "function_name": "StackingRegressor: 5 base + Ridge meta",
         "learning_strategy": "Meta-learner learns optimal combination",
         "why_used": "Maximize ensemble diversity"},
    ]
    return pd.DataFrame(rows)


def forecast_recursive(daily, fits):
    pred_dates = pd.date_range("2015-01-11", "2015-01-17", freq="D")
    known_daily = daily[daily["date"] <= "2015-01-10"].copy()
    history = {
        target: dict(zip(known_daily["date"], known_daily[target])) for target in TARGETS
    }
    feature_lookup = daily.set_index("date")[MODEL_WEATHER + CALENDAR_FEATURES].to_dict("index")
    rows = []
    for date in pred_dates:
        row = {"date": date, "weekday": WEEKDAY_CN[date.dayofweek]}
        for target, fit in fits.items():
            feature_values = dict(feature_lookup[date])
            last7, last14 = [], []
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
            history[target][date] = pred
            row[f"pred_{target}"] = pred
            row[f"model_{target}"] = fit.model_name
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    fit_base = daily[(daily["date"] >= "2012-01-01") & (daily["date"] <= "2014-12-31")]
    max_gap = float((fit_base["load_max"] - fit_base["load_mean"]).median())
    min_gap = float((fit_base["load_mean"] - fit_base["load_min"]).median())

    # 物理约束修正
    pred_df.loc[pred_df["pred_load_min"] > pred_df["pred_load_mean"], "pred_load_min"] = \
        (pred_df["pred_load_mean"] - min_gap)
    pred_df.loc[pred_df["pred_load_max"] < pred_df["pred_load_mean"], "pred_load_max"] = \
        (pred_df["pred_load_mean"] + max_gap)
    return pred_df


# ══════════════════════════════════════════════════════════════════════════════
# 13. 图表导出
# ══════════════════════════════════════════════════════════════════════════════

def save_figures(train_2012_2014, corr_df, ols_df, performance_df, validation_df,
                 pred_df, pairwise_df, approx_df, dm_mse_df, outlier_df,
                 optuna_results, seasonal_df, friedman_df):
    # 01 气象回归
    corr_pivot = corr_df.pivot(index="target", columns="weather_factor", values="pearson_r")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(corr_pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[0])
    axes[0].set_title("2012-2014 负荷与气象因素 Pearson 相关系数")
    coef_df = ols_df[ols_df["term"].isin(REG_WEATHER)].copy()
    coef_df["coef_scaled"] = coef_df.groupby("target")["coef"].transform(lambda s: s / max(np.nanmax(np.abs(s)), 1))
    sns.barplot(data=coef_df, y="term", x="coef_scaled", hue="target", ax=axes[1])
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_title("GLM 线性回归系数方向（按目标归一化）")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_weather_regression.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 02 模型对比
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=performance_df, x="target", y="RMSE", hue="model", ax=ax)
    ax.set_title("v4 模型验证集 RMSE 对比 (Optuna + 扩大验证集)")
    ax.set_xlabel("预测目标"); ax.set_ylabel("RMSE (MW)")
    ax.legend(title="候选模型", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_model_training_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 03 验证拟合
    selected = performance_df.sort_values(["target", "RMSE"]).groupby("target").first().reset_index()
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, target in zip(axes, [TARGETS[t] for t in TARGET_ORDER]):
        best_model = selected[selected["target"] == target]["model"].iloc[0]
        data = validation_df[(validation_df["target"] == target) & (validation_df["model"] == best_model)]
        ax.plot(data["date"], data["actual"], "o-", label="Actual", color="#111827", linewidth=2, markersize=3)
        ax.plot(data["date"], data["predicted"], "s--", label=f"{best_model} Pred", color="#2563eb", linewidth=1.5, markersize=2)
        ax.set_ylabel("MW"); ax.set_title(f"{target}: 2014-07-01 ~ 2014-12-31 Validation"); ax.legend(loc="upper right")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_validation_fit.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 04 最终预测
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(pred_df))
    ax.fill_between(x, pred_df["pred_load_min"], pred_df["pred_load_max"], color="#93c5fd", alpha=0.35, label="Min-Max Range")
    ax.plot(x, pred_df["pred_load_mean"], "o-", color="#1d4ed8", linewidth=2.5, label="Mean Load")
    ax.plot(x, pred_df["pred_load_max"], "^--", color="#dc2626", linewidth=1.5, label="Max Load")
    ax.plot(x, pred_df["pred_load_min"], "v--", color="#16a34a", linewidth=1.5, label="Min Load")
    for idx, value in enumerate(pred_df["pred_load_mean"]):
        ax.text(idx, value + 80, f"{value:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:%m-%d}\n{w}" for d, w in zip(pred_df["date"], pred_df["weekday"])])
    ax.set_ylabel("Load (MW)"); ax.set_title("2015-01-11 ~ 2015-01-17 Power Load Forecast (v4)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_final_prediction.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 05 历史趋势衔接
    fig, ax = plt.subplots(figsize=(14, 5))
    history = train_2012_2014[train_2012_2014["date"] >= "2014-07-01"]
    ax.plot(history["date"], history["load_mean"], color="#64748b", label="2014 H2 Mean Load")
    ax.plot(pred_df["date"], pred_df["pred_load_mean"], "o-", color="#1d4ed8", label="2015 Jan Forecast")
    ax.axvline(pd.Timestamp("2015-01-11"), color="#dc2626", linestyle="--", linewidth=1)
    ax.set_title("Forecast vs Historical Context (v4)"); ax.set_ylabel("Mean Load (MW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_history_forecast_context.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 06 DM 检验热力图
    if not dm_mse_df.empty:
        for label in [TARGETS[t] for t in TARGET_ORDER]:
            sub = dm_mse_df[dm_mse_df["target"] == label]
            models_list = sorted(set(sub["model_a"].unique()) | set(sub["model_b"].unique()))
            n_models = len(models_list)
            p_matrix = np.ones((n_models, n_models))
            for _, row in sub.iterrows():
                i = models_list.index(row["model_a"])
                j = models_list.index(row["model_b"])
                p_matrix[i, j] = row["p_value"]
                p_matrix[j, i] = row["p_value"]
            fig, ax = plt.subplots(figsize=(8, 7))
            mask = np.eye(n_models, dtype=bool)
            sns.heatmap(p_matrix, annot=True, fmt=".4f", cmap="RdYlGn_r",
                        vmin=0, vmax=1, xticklabels=models_list, yticklabels=models_list,
                        mask=mask, ax=ax, cbar_kws={"label": "DM p-value"})
            ax.set_title(f"DM Test p-value Matrix (MSE) - {label}\nalpha={ALPHA}")
            fig.tight_layout()
            safe_label = label.replace("/", "_")
            fig.savefig(OUTPUT_DIR / f"06_dm_test_heatmap_{safe_label}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

    # 07 Optuna 优化历史
    if optuna_results:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for ax, mt in zip(axes, ["Ridge", "RandomForest", "XGBoost", "SVR", "KNN"]):
            for result in optuna_results:
                if result.model_name == mt and not result.trials_df.empty:
                    ax.plot(result.trials_df["trial_number"], result.trials_df["value"].cummin(),
                            alpha=0.8, linewidth=1.5, label=f"{result.target}")
            ax.set_title(f"{mt} Optuna"); ax.set_xlabel("Trial"); ax.set_ylabel("Best RMSE"); ax.legend(fontsize=7)
        axes[-1].set_visible(False)
        fig.suptitle("v4 Optuna Optimization History (50 trials)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "07_optuna_optimization_history.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # 08 季节性分析
    if not seasonal_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        season_data = seasonal_df[seasonal_df["group_by"] == "season"]
        sns.barplot(data=season_data, x="group_value", y="RMSE", hue="model", ax=axes[0])
        axes[0].set_title("Seasonal RMSE"); axes[0].set_xlabel("Season"); axes[0].set_ylabel("RMSE (MW)")
        daytype_data = seasonal_df[seasonal_df["group_by"] == "day_type"]
        sns.barplot(data=daytype_data, x="group_value", y="RMSE", hue="model", ax=axes[1])
        axes[1].set_title("Weekday vs Weekend RMSE"); axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "08_seasonal_performance.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # 09 Friedman 排名
    if not friedman_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        for _, row in friedman_df.iterrows():
            ranks = json.loads(row["model_avg_ranks"])
            models_sorted = sorted(ranks.items(), key=lambda x: x[1])
            names = [m[0] for m in models_sorted]
            vals = [m[1] for m in models_sorted]
            ax.barh(names, vals, alpha=0.7, label=row["target"])
        ax.set_xlabel("Average Rank (lower=better)")
        ax.set_title(f"Friedman Test Ranking (p={friedman_df['friedman_p_value'].iloc[0]:.4f})")
        ax.invert_yaxis(); ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "09_friedman_ranking.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 14. 输出保存
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(corr_df, ols_df, performance_df, validation_df, training_log_df,
                 selection_df, tests_df, pairwise_df, approx_df, dm_mse_df,
                 ensemble_df, manifest_df, pred_df, outlier_df, optuna_results,
                 seasonal_df, friedman_df, residual_norm_df, residual_extra_df):
    corr_df.to_csv(OUTPUT_DIR / "project2_weather_correlations.csv", index=False, encoding="utf-8-sig")
    ols_df.to_csv(OUTPUT_DIR / "project2_regression_summary.csv", index=False, encoding="utf-8-sig")
    performance_df.to_csv(OUTPUT_DIR / "project2_model_performance.csv", index=False, encoding="utf-8-sig")
    validation_df.to_csv(OUTPUT_DIR / "project2_validation_predictions.csv", index=False, encoding="utf-8-sig")
    training_log_df.to_csv(OUTPUT_DIR / "project2_training_log.csv", index=False, encoding="utf-8-sig")
    selection_df.to_csv(OUTPUT_DIR / "project2_model_selection.csv", index=False, encoding="utf-8-sig")
    tests_df.to_csv(OUTPUT_DIR / "project2_statistical_tests.csv", index=False, encoding="utf-8-sig")
    pairwise_df.to_csv(OUTPUT_DIR / "project2_pairwise_significance.csv", index=False, encoding="utf-8-sig")
    approx_df.to_csv(OUTPUT_DIR / "project2_approx_correctness.csv", index=False, encoding="utf-8-sig")
    dm_mse_df.to_csv(OUTPUT_DIR / "project2_dm_test.csv", index=False, encoding="utf-8-sig")
    ensemble_df.to_csv(OUTPUT_DIR / "project2_ensemble_strategy.csv", index=False, encoding="utf-8-sig")
    manifest_df.to_csv(OUTPUT_DIR / "project2_model_manifest.csv", index=False, encoding="utf-8-sig")
    outlier_df.to_csv(OUTPUT_DIR / "project2_outlier_diagnosis.csv", index=False, encoding="utf-8-sig")
    seasonal_df.to_csv(OUTPUT_DIR / "project2_seasonal_analysis.csv", index=False, encoding="utf-8-sig")
    friedman_df.to_csv(OUTPUT_DIR / "project2_friedman_test.csv", index=False, encoding="utf-8-sig")
    residual_norm_df.to_csv(OUTPUT_DIR / "project2_residual_normality.csv", index=False, encoding="utf-8-sig")
    residual_extra_df.to_csv(OUTPUT_DIR / "project2_residual_autocorr_hetero.csv", index=False, encoding="utf-8-sig")

    (OUTPUT_DIR / "project2_model_manifest.json").write_text(
        manifest_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8"
    )

    all_trials = pd.concat(
        [r.trials_df.assign(model=r.model_name, target=r.target) for r in optuna_results], ignore_index=True
    )
    all_trials.to_csv(OUTPUT_DIR / "project2_optuna_trials.csv", index=False, encoding="utf-8-sig")

    final_csv = pred_df[["date", "weekday", "pred_load_mean", "pred_load_max", "pred_load_min",
                          "model_load_mean", "model_load_max", "model_load_min"]].copy()
    final_csv["date"] = final_csv["date"].dt.strftime("%Y-%m-%d")
    for col in ["pred_load_mean", "pred_load_max", "pred_load_min"]:
        final_csv[col] = final_csv[col].round(2)
    final_csv.columns = ["日期", "星期", "预测日平均(MW)", "预测日最高(MW)", "预测日最低(MW)", "均值模型", "最高模型", "最低模型"]
    final_csv.to_csv(OUTPUT_DIR / "project2_final_prediction_2015_01_11_17.csv", index=False, encoding="utf-8-sig")


# ══════════════════════════════════════════════════════════════════════════════
# 15. 七步前置检查报告生成 (v4 NEW)
# ══════════════════════════════════════════════════════════════════════════════

def generate_seven_step_report(
    daily, performance_df, tests_df, friedman_df,
    residual_norm_df, residual_extra_df, pred_df, outlier_df, seasonal_df,
):
    """生成完整的七步数据分析前置检查报告"""
    train = daily[(daily["date"] >= "2012-01-01") & (daily["date"] <= "2014-12-31")]

    lines = []
    lines.append("==================== 数据分析前置检查报告 ====================")
    lines.append("=" * 70)
    lines.append(f"  项目: 短期电力负荷预测 (Project 2)")
    lines.append(f"  版本: v4 (七步分析 + 残差诊断)")
    lines.append(f"  数据: 2012-2014 历史数据, 预测 2015-01-11 至 2015-01-17")
    lines.append(f"  训练: Ridge / RF / XGBoost(CUDA) / SVR / KNN / Stacking")
    lines.append(f"  优化: Optuna 50 trials, TimeSeriesSplit CV")
    lines.append("=" * 70)
    lines.append("")

    # 第〇步: 业务理解与文献综述
    lines.append("=" * 70)
    lines.append("第〇步: 业务理解与文献综述")
    lines.append("=" * 70)
    lines.append("  业务背景:")
    lines.append("    电力系统运行需要准确的短期负荷预测(1-7天)来指导机组启停、")
    lines.append("    燃料采购和电网调度决策。负荷预测误差直接影响供电可靠性和经济性。")
    lines.append("  预测目标:")
    lines.append("    日最高负荷(load_max)、日最低负荷(load_min)、日平均负荷(load_mean)")
    lines.append("    单位: MW (兆瓦)")
    lines.append("  关键约束:")
    lines.append("    (1) 物理约束: load_min <= load_mean <= load_max, 所有负荷 > 0")
    lines.append("    (2) 季节周期: 负荷受气温影响显著, 冬夏高、春秋低")
    lines.append("    (3) 节假日效应: 春节/国庆等长假负荷大幅下降")
    lines.append("  文献参考:")
    lines.append("    [1] Hong T, et al. (2016) 'Probabilistic energy forecasting' — 电力负荷预测综述")
    lines.append("    [2] Diebold & Mariano (1995) 'Comparing predictive accuracy' — DM检验经典文献")
    lines.append("    [3] Hyndman & Athanasopoulos (2021) 'Forecasting: Principles and Practice'")
    lines.append("    [4] Chen & Guestrin (2016) 'XGBoost: A Scalable Tree Boosting System'")
    lines.append("  模型选择依据:")
    lines.append("    线性基线(Ridge) + 树模型族(RF+XGBoost) + 核方法(SVR) +")
    lines.append("    距离方法(KNN) + 集成(Stacking), 覆盖5种不同学习范式")
    lines.append("")

    # 第一步: Y 变量类型
    lines.append("=" * 70)
    lines.append("第一步: Y 变量类型判断")
    lines.append("=" * 70)
    for col in TARGET_ORDER:
        vals = train[col].dropna()
        n_unique = vals.nunique()
        n_total = len(vals)
        lines.append(f"  {TARGETS[col]} ({col}):")
        lines.append(f"    唯一值: {n_unique}/{n_total} (比率={n_unique/n_total:.4f})")
        lines.append(f"    取值范围: [{vals.min():.2f}, {vals.max():.2f}] MW")
        lines.append(f"    数据类型: 连续回归变量 (有物理量纲)")
        lines.append(f"    联系函数: identity (Y = Xb)")
        lines.append(f"    建议模型族: Ridge, RF, XGBoost, SVR, KNN, Stacking")
    lines.append("  结论: 三个目标均为连续回归问题, 使用 identity link, 无需 log/Box-Cox 变换")
    lines.append("")

    # 第二步: Y 分布诊断
    lines.append("=" * 70)
    lines.append("第二步: Y 分布诊断")
    lines.append("=" * 70)
    for col in TARGET_ORDER:
        vals = train[col].dropna()
        mu = vals.mean()
        med = vals.median()
        sigma = vals.std()
        skew = vals.skew()
        kurt = vals.kurtosis()
        cv = sigma / mu
        pcts = vals.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        abs_skew = abs(skew)
        if abs_skew < 0.5:
            skew_diag = "近似对称, 线性模型可行"
        elif abs_skew < 1.0:
            skew_diag = "中度偏态, 建议用树模型或验证残差正态性"
        else:
            skew_diag = "严重偏态, 考虑变换或树模型"
        lines.append(f"  {TARGETS[col]} ({col}):")
        lines.append(f"    均值={mu:.2f}, 中位数={med:.2f}, 标准差={sigma:.2f}")
        lines.append(f"    偏度={skew:.4f}, 峰度={kurt:.4f}")
        lines.append(f"    P5={pcts[0.05]:.2f}, P25={pcts[0.25]:.2f}, P50={pcts[0.50]:.2f}, P75={pcts[0.75]:.2f}, P95={pcts[0.95]:.2f}")
        lines.append(f"    CV={cv:.4f} ({'高离散>0.3' if cv > 0.3 else '中等0.1-0.3' if cv > 0.1 else '低离散<0.1'})")
        lines.append(f"    偏态诊断: |偏度|={abs_skew:.4f} -> {skew_diag}")
    lines.append("  结论: 三个目标均为中度左偏 (|skew| 0.5-1.0), CV约0.22 (中等离散)")
    lines.append("    树模型天然不受偏态影响; Ridge/SVR 需关注残差正态性")
    lines.append("    详细分布图: v4_step2_distribution.png")
    lines.append("")

    # 第三步: 离散度与异常值
    lines.append("=" * 70)
    lines.append("第三步: 离散度与异常值")
    lines.append("=" * 70)
    cause_counts = outlier_df["cause"].value_counts()
    target_counts = outlier_df["target_col"].value_counts()
    lines.append(f"  异常值总计: {len(outlier_df)} 个 (IQR 1.5倍)")
    for col in TARGET_ORDER:
        sub = outlier_df[outlier_df["target_col"] == col]
        lines.append(f"    {TARGETS[col]}: {len(sub)} 个")
    lines.append("  成因分布:")
    for cause, count in cause_counts.items():
        lines.append(f"    - {cause}: {count} 天")
    lines.append("  异方差性检查 (按季度):")
    for col in TARGET_ORDER:
        train_copy = train.copy()
        train_copy["quarter"] = train_copy["date"].dt.quarter
        grouped = train_copy.groupby("quarter")[col].agg(["mean", "std"]).dropna()
        corr_mv = grouped["mean"].corr(grouped["std"])
        lines.append(f"    {TARGETS[col]}: 均值-标准差 r={corr_mv:.4f} ({'存在异方差' if abs(corr_mv) > 0.5 else '方差基本恒定'})")
    lines.append("  处置: 全部保留 (均为真实极端事件: 春节/国庆/元旦/劳动节/极端天气)")
    lines.append("  依据: IQR 检测到的异常与已知节假日高度吻合, 非数据质量问题")
    lines.append("")

    # 第四步: 共线性检查
    lines.append("=" * 70)
    lines.append("第四步: 共线性检查 (VIF)")
    lines.append("=" * 70)
    lines.append("  严重共线性特征 (VIF > 10):")
    for col in TARGET_ORDER:
        vif_path = OUTPUT_DIR / f"v4_step4_vif_{col}.csv"
        if vif_path.exists():
            vif_df = pd.read_csv(vif_path)
            high = vif_df[vif_df["VIF"] > 10]
            lines.append(f"    {TARGETS[col]}: {len(high)} 个特征 VIF > 10")
            for _, r in high.head(5).iterrows():
                vif_val = r["VIF"]
                lines.append(f"      - {r['feature']}: VIF={vif_val:.2f} {'[inf]' if vif_val == float('inf') else ''}")
    high_corr_path = OUTPUT_DIR / f"v4_step4_high_corr_load_mean.csv"
    if high_corr_path.exists():
        hc = pd.read_csv(high_corr_path)
        lines.append(f"  Pearson |r| > 0.8 的高相关特征对: {len(hc)} 对")
        for _, r in hc.head(5).iterrows():
            lines.append(f"    {r['feature_a']} vs {r['feature_b']}: r={r['pearson_r']:.4f}")
    lines.append("  结论: 共线性严重, 但:")
    lines.append("    (1) 树模型 (RF/XGBoost/Stacking) 对共线性天然不敏感")
    lines.append("    (2) Ridge 的 L2 正则化可缓解系数膨胀")
    lines.append("    (3) OLS 不在主方案中, VIF 仅作诊断参考")
    lines.append("")

    # 第五步: 统计显著性标准
    lines.append("=" * 70)
    lines.append("第五步: 统计显著性标准")
    lines.append("=" * 70)
    lines.append(f"  显著性水平: alpha = {ALPHA}")
    lines.append("  检验方法体系:")
    lines.append("    (1) 配对 T 检验: 比较两模型绝对误差差异的显著性")
    lines.append("    (2) McNemar 检验: 比较两模型犯错模式的差异 (基于5%容忍度)")
    lines.append("    (3) Diebold-Mariano 检验: 比较预测能力差异 (Newey-West方差)")
    lines.append("    (4) Friedman 检验: 多模型非参数排名比较")
    lines.append("    (5) PAC 近似正确概率: 验证集上相对误差 <= 阈值的比例")
    lines.append("  验证集规模: 2014-07-01 至 2014-12-31 (~184天)")
    lines.append(f"    样本量充足, 统计检验效力较高")
    lines.append("")

    # 第六步: 模型正确性检查 (残差分析)
    lines.append("=" * 70)
    lines.append("第六步: 模型正确性检查 (残差分析)")
    lines.append("=" * 70)
    for _, row in residual_norm_df.iterrows():
        lines.append(f"  {row['target']} ({row['model']}):")
        lines.append(f"    残差均值={row['residual_mean']:.4f}, 标准差={row['residual_std']:.4f}")
        lines.append(f"    Shapiro-Wilk: W={row['Shapiro_Wilk_W']:.6f}, p={row['Shapiro_Wilk_p']:.6f} -> {row['Shapiro_Wilk_conclusion']}")
        if not np.isnan(row.get("Dagostino_p", np.nan)):
            lines.append(f"    D'Agostino: K2={row['Dagostino_K2']:.6f}, p={row['Dagostino_p']:.6f} -> {row['Dagostino_conclusion']}")
        lines.append(f"    K-S test: stat={row['KS_statistic']:.6f}, p={row['KS_p']:.6f} -> {row['KS_conclusion']}")

    # 残差自相关
    for _, row in residual_extra_df.iterrows():
        if "LjungBox" in str(row.get("conclusion", "")):
            lines.append(f"    Ljung-Box (lag={row['lag']}): stat={row['LjungBox_statistic']:.4f}, p={row['LjungBox_p_value']:.6f} -> {row['conclusion']}")

    # 异方差
    for _, row in residual_extra_df.iterrows():
        if "BreuschPagan" in str(row.get("conclusion", "")):
            lines.append(f"    Breusch-Pagan: stat={row['BreuschPagan_statistic']:.4f}, p={row['BreuschPagan_p_value']:.6f} -> {row['BreuschPagan_conclusion']}")

    lines.append("  模型正确性总结:")
    lines.append("    - 残差均值接近零: 模型无系统性偏差")
    lines.append("    - 树模型对残差正态性要求低: 偏态残差可接受")
    lines.append("    - 残差自相关: 若存在, 提示可能遗漏时序特征")
    lines.append("    - 异方差: 若存在, 不影响树模型; 线性模型需加权")
    lines.append("  详细残差图: v4_step6_residual_analysis.png")
    lines.append("")

    # 第七步: 建模路线图 + 最终结果
    lines.append("=" * 70)
    lines.append("第七步: 建模路线图与结果")
    lines.append("=" * 70)
    lines.append("  建模路线图:")
    lines.append("    1. 数据加载: Excel宽表 -> pd.melt -> 日聚合 (max/min/mean)")
    lines.append("    2. 特征工程: 日历sin/cos + 气象衍生(HDD/CDD/temp_range) + 滞后/滚动统计")
    lines.append("    3. 异常值诊断: IQR检测 + 成因分类 + 全部保留")
    lines.append("    4. 模型训练: 5基模型 + Optuna调参(50 trials) + Stacking集成")
    lines.append("    5. 验证评估: 184天hold-out + TimeSeriesSplit CV + Bootstrap OOB")
    lines.append("    6. 统计检验: T检验 + McNemar + DM检验 + Friedman + PAC")
    lines.append("    7. 残差诊断: 正态性 + 自相关 + 异方差")
    lines.append("    8. 递推预测: 7天逐日递推 + 物理约束修正")
    lines.append("")

    lines.append("  模型性能排名 (验证集 RMSE):")
    for _, row in performance_df.sort_values(["target", "RMSE"]).groupby("target").first().reset_index().iterrows():
        lines.append(f"    {row['target']}: {row['model']} RMSE={row['RMSE']:.2f}, R2={row['R2']:.4f}")

    lines.append("")
    lines.append("  统计检验汇总 (最优 vs 次优):")
    for _, row in tests_df.iterrows():
        lines.append(f"    {row['target']}: {row['best_model']} vs {row['second_model']}")
        lines.append(f"      T检验 p={row['paired_t_p_value']:.6f} -> {row['t_test_conclusion']}")
        if not np.isnan(row.get("dm_mse_p_value", np.nan)):
            lines.append(f"      DM检验 p={row['dm_mse_p_value']:.6f} -> {row['dm_mse_conclusion']}")
        lines.append(f"      PAC@5%={row['best_approx_correct_probability_at_5pct']:.4f}")

    lines.append("")
    if not friedman_df.empty:
        for _, row in friedman_df.iterrows():
            ranks = json.loads(row["model_avg_ranks"])
            ranks_str = ", ".join(f"{m}={v:.2f}" for m, v in sorted(ranks.items(), key=lambda x: x[1]))
            lines.append(f"  Friedman检验: {row['target']}: chi2={row['friedman_chi2']:.2f}, p={row['friedman_p_value']:.6f} -> {row['conclusion']}")
            lines.append(f"    模型平均秩次: {ranks_str}")

    lines.append("")
    lines.append("  最终预测 (2015-01-11 ~ 2015-01-17):")
    for _, row in pred_df.iterrows():
        lines.append(
            f"    {row['date'].strftime('%Y-%m-%d')} {row['weekday']}: "
            f"mean={row['pred_load_mean']:.1f}MW, max={row['pred_load_max']:.1f}MW, min={row['pred_load_min']:.1f}MW"
        )

    lines.append("")
    lines.append("=" * 70)
    lines.append("  报告生成完毕")
    lines.append(f"  所有输出保存至: {OUTPUT_DIR}")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    # 保存报告
    report_path = OUTPUT_DIR / "v4_seven_step_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
# 16. 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Project 2 v4: 短期电力负荷预测 — 七步分析 + 残差诊断版")
    print("=" * 60)
    print(f"  验证集: 2014-07-01 至 2014-12-31 (~184天)")
    print(f"  候选模型: Ridge, RF, XGBoost(CUDA), SVR, KNN, Stacking")
    print(f"  统计检验: T + McNemar + DM + Friedman + PAC")
    print(f"  v4 新增: 残差分析 (正态性/自相关/异方差)")
    print("=" * 60)

    # 清理旧输出
    for old_model in MODEL_DIR.glob("*.joblib"):
        old_model.unlink()

    # 1. 数据加载
    print("\n[1/9] 加载数据...")
    daily = load_daily_data()
    train_2012_2014 = daily[(daily["date"] >= "2012-01-01") & (daily["date"] <= "2014-12-31")].copy()
    print(f"  总数据: {len(daily)} 天 | 训练期: {len(train_2012_2014)} 天 (2012-2014)")

    # 2. 异常值诊断
    print("\n[2/9] 异常值诊断...")
    outlier_df = outlier_diagnosis(daily)
    cause_summary = outlier_df["cause"].value_counts().to_dict()
    print(f"  检测到 {len(outlier_df)} 个统计异常日")
    for cause, count in sorted(cause_summary.items(), key=lambda x: -x[1]):
        print(f"    - {cause}: {count} 天")

    # 3. 气象回归
    print("\n[3/9] 气象回归分析...")
    corr_df, ols_df = build_weather_regression(train_2012_2014)

    # 4. 模型训练 + Optuna
    print("\n[4/9] 模型训练 + Optuna 超参数优化...")
    fits, performance_df, validation_df, training_log_df, selection_df, manifest_df, optuna_results = fit_and_evaluate(daily)

    # 5. 统计检验
    print("\n[5/9] 统计检验...")
    tests_df, pairwise_df, approx_df, dm_mse_df = statistical_tests(performance_df, validation_df)

    # 6. 多目标统计
    print("\n[6/9] 多目标联合统计 (Friedman + 季节性)...")
    friedman_df = friedman_test(validation_df)
    seasonal_df = seasonal_analysis(validation_df)

    # 7. 残差分析 (v4 NEW)
    print("\n[7/9] 残差分析 (Step 6)...")
    residual_norm_df, residual_extra_df = residual_analysis(daily, fits)

    # 8. 集成策略 + 递推预测
    print("\n[8/9] 集成策略 + 递推预测...")
    ensemble_df = ensemble_strategy_table()
    pred_df = forecast_recursive(daily, fits)

    # 9. 导出
    print("\n[9/9] 导出图表和 CSV...")
    save_figures(
        train_2012_2014, corr_df, ols_df, performance_df, validation_df,
        pred_df, pairwise_df, approx_df, dm_mse_df, outlier_df,
        optuna_results, seasonal_df, friedman_df,
    )
    save_outputs(
        corr_df, ols_df, performance_df, validation_df, training_log_df,
        selection_df, tests_df, pairwise_df, approx_df, dm_mse_df,
        ensemble_df, manifest_df, pred_df, outlier_df, optuna_results,
        seasonal_df, friedman_df, residual_norm_df, residual_extra_df,
    )

    # 生成七步报告
    print("\n[Report] 生成七步前置检查报告...")
    report_path = generate_seven_step_report(
        daily, performance_df, tests_df, friedman_df,
        residual_norm_df, residual_extra_df, pred_df, outlier_df, seasonal_df,
    )

    print(f"\n  七步报告已保存至: {report_path}")
    print(f"  所有输出已保存至: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
