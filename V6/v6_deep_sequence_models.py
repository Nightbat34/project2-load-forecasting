#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V6 PyTorch sequence-model exploration for Project 2.

This script trains small GPU-accelerated GRU/LSTM/Attention-GRU models as an
extension to the V5 Stacking solution. Stacking remains the final production
model; these neural networks are used for method exploration and reporting.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "Data" / "附件1-电网负荷数据.xlsx"
OUT = ROOT / "output" / "project2"
DL_DIR = OUT / "deep_learning"
DL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SEQ_LEN = 28
TRAIN_END = pd.Timestamp("2014-06-30")
VAL_START = pd.Timestamp("2014-07-01")
VAL_END = pd.Timestamp("2014-12-31")
FORECAST_START = pd.Timestamp("2015-01-11")
FORECAST_END = pd.Timestamp("2015-01-17")

TARGETS = ["load_mean", "load_max", "load_min"]
TARGET_LABELS = {
    "load_mean": "日平均负荷",
    "load_max": "日最高负荷",
    "load_min": "日最低负荷",
}
OUTPUT_LABELS = ["日平均负荷", "日最高负荷", "日最低负荷"]


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_daily_data() -> pd.DataFrame:
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


FEATURES = [
    "load_mean",
    "load_max",
    "load_min",
    "temp_max",
    "temp_min",
    "temp_avg",
    "humidity",
    "rainfall",
    "temp_range",
    "hdd",
    "cdd",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
]


def make_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    required = FEATURES + TARGETS
    known = df[df[required].notna().all(axis=1)].copy().reset_index(drop=True)
    X_rows, y_rows, date_rows = [], [], []
    values = known[FEATURES].to_numpy(dtype=np.float32)
    targets = known[TARGETS].to_numpy(dtype=np.float32)
    dates = known["date"].to_numpy()
    for i in range(seq_len, len(known)):
        X_rows.append(values[i - seq_len : i])
        y_rows.append(targets[i])
        date_rows.append(pd.Timestamp(dates[i]))
    return np.stack(X_rows), np.stack(y_rows), pd.to_datetime(date_rows)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE(%)": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


class SeqRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, rnn_type: str, attention: bool = False):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.rnn_type = rnn_type
        self.attention = attention
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        if attention:
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        if self.attention:
            score = self.attn(out).squeeze(-1)
            weight = torch.softmax(score, dim=1)
            context = torch.sum(out * weight.unsqueeze(-1), dim=1)
            return self.head(context), weight
        return self.head(out[:, -1, :]), None


@dataclass
class TrainResult:
    model_name: str
    model: nn.Module
    train_losses: list[float]
    val_losses: list[float]
    fit_seconds: float
    best_epoch: int


def train_model(
    model_name: str,
    rnn_type: str,
    attention: bool,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> TrainResult:
    model = SeqRegressor(X_train.shape[-1], hidden_dim=32, rnn_type=rnn_type, attention=attention).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.float32).to(device)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience = 18
    wait = 0
    train_losses, val_losses = [], []
    start = time.time()
    for epoch in range(1, 121):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        if not batch_losses:
            raise RuntimeError(f"{model_name} produced non-finite training loss for all batches.")
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(val_x)
            val_loss = float(loss_fn(val_pred, val_y).detach().cpu())
        if not np.isfinite(val_loss):
            val_loss = float("inf")
        train_loss = float(np.mean(batch_losses))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    fit_seconds = time.time() - start
    return TrainResult(model_name, model, train_losses, val_losses, fit_seconds, best_epoch)


def predict_model(model: nn.Module, X: np.ndarray, device: torch.device):
    model.eval()
    with torch.no_grad():
        pred, weight = model(torch.tensor(X, dtype=torch.float32).to(device))
    pred_np = pred.detach().cpu().numpy()
    pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=0.0, neginf=0.0)
    weight_np = None if weight is None else weight.detach().cpu().numpy()
    return pred_np, weight_np


def recursive_forecast(
    model: nn.Module,
    daily: pd.DataFrame,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    device: torch.device,
) -> pd.DataFrame:
    work = daily.copy().sort_values("date").reset_index(drop=True)
    forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    rows = []
    for date in forecast_dates:
        idx = work.index[work["date"] == date]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        hist = work.loc[i - SEQ_LEN : i - 1, FEATURES].copy()
        hist_scaled = feature_scaler.transform(hist.to_numpy(dtype=np.float32))
        pred_scaled, _ = predict_model(model, hist_scaled.reshape(1, SEQ_LEN, -1), device)
        pred = target_scaler.inverse_transform(pred_scaled)[0]
        mean_v, max_v, min_v = float(pred[0]), float(pred[1]), float(pred[2])
        min_v = max(min_v, 1.0)
        max_v = max(max_v, min_v + 1.0)
        mean_v = min(max(mean_v, min_v), max_v)
        work.loc[i, TARGETS] = [mean_v, max_v, min_v]
        rows.append(
            {
                "日期": date.strftime("%Y-%m-%d"),
                "星期": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][date.weekday()],
                "预测日平均(MW)": round(mean_v, 2),
                "预测日最高(MW)": round(max_v, 2),
                "预测日最低(MW)": round(min_v, 2),
            }
        )
    return pd.DataFrame(rows)


def plot_loss(history_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for row in history_rows:
        epochs = np.arange(1, len(row["train_losses"]) + 1)
        ax.plot(epochs, row["train_losses"], linestyle="--", alpha=0.75, label=f"{row['model']} train")
        ax.plot(epochs, row["val_losses"], linewidth=2, label=f"{row['model']} val")
    ax.set_title("V6 PyTorch Sequence Models: Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SmoothL1 loss (scaled target)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "10_deep_learning_loss.png", dpi=160)
    plt.close(fig)


def plot_validation(dates, y_true, pred_map: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for j, target in enumerate(TARGETS):
        ax = axes[j]
        ax.plot(dates, y_true[:, j], color="black", linewidth=2, label="True")
        for name, pred in pred_map.items():
            ax.plot(dates, pred[:, j], linewidth=1.4, alpha=0.9, label=name)
        ax.set_ylabel("MW")
        ax.set_title(OUTPUT_LABELS[j])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Validation date")
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "11_deep_learning_validation_fit.png", dpi=160)
    plt.close(fig)


def plot_attention(dates, weights: np.ndarray | None) -> None:
    if weights is None:
        return
    avg_weights = weights.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    lags = np.arange(SEQ_LEN, 0, -1)
    ax.bar(lags, avg_weights, color="#2563eb")
    ax.set_title("Attention-GRU: Average Temporal Attention Weight")
    ax.set_xlabel("Days before prediction")
    ax.set_ylabel("Average weight")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(OUT / "12_attention_weights.png", dpi=160)
    plt.close(fig)


def plot_comparison(v5_perf: pd.DataFrame, dl_perf: pd.DataFrame) -> None:
    best_stack = v5_perf[v5_perf["model"] == "Stacking"][["target", "RMSE"]].copy()
    best_stack["model"] = "Stacking(V5主模型)"
    deep = dl_perf[["target", "model", "RMSE"]].copy()
    comp = pd.concat([best_stack, deep], ignore_index=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = comp["target"].unique()
    x = np.arange(len(labels))
    models = comp["model"].unique()
    width = 0.18
    for i, model in enumerate(models):
        vals = [comp[(comp["target"] == t) & (comp["model"] == model)]["RMSE"].iloc[0] for t in labels]
        ax.bar(x + (i - (len(models)-1)/2) * width, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Validation RMSE (MW)")
    ax.set_title("V6 Deep Learning Exploration vs V5 Stacking")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "13_deep_learning_vs_stacking.png", dpi=160)
    plt.close(fig)


def rolling_window_eval(best_pred: pd.DataFrame, y_true: pd.DataFrame) -> pd.DataFrame:
    merged = best_pred.merge(y_true, on="date")
    rows = []
    for start in pd.date_range(VAL_START, VAL_END - pd.Timedelta(days=27), freq="14D"):
        end = start + pd.Timedelta(days=27)
        win = merged[(merged["date"] >= start) & (merged["date"] <= end)]
        if len(win) < 20:
            continue
        for target in TARGETS:
            rmse = np.sqrt(mean_squared_error(win[target], win[f"{target}_pred"]))
            rows.append({
                "window_start": start.strftime("%Y-%m-%d"),
                "window_end": end.strftime("%Y-%m-%d"),
                "target": TARGET_LABELS[target],
                "n_days": len(win),
                "RMSE": round(float(rmse), 2),
            })
    return pd.DataFrame(rows)


def main() -> None:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    daily = load_daily_data()
    X, y, dates = make_sequences(daily, SEQ_LEN)
    train_mask = dates <= TRAIN_END
    val_mask = (dates >= VAL_START) & (dates <= VAL_END)
    X_train_raw, y_train_raw = X[train_mask], y[train_mask]
    X_val_raw, y_val_raw = X[val_mask], y[val_mask]
    val_dates = dates[val_mask]

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    feature_scaler.fit(X_train_2d)
    target_scaler.fit(y_train_raw)
    X_train = feature_scaler.transform(X_train_2d).reshape(X_train_raw.shape)
    X_val = feature_scaler.transform(X_val_raw.reshape(-1, X_val_raw.shape[-1])).reshape(X_val_raw.shape)
    y_train = target_scaler.transform(y_train_raw)
    y_val = target_scaler.transform(y_val_raw)

    configs = [
        ("GRU", "GRU", False),
        ("LSTM", "LSTM", False),
        ("Attention-GRU", "GRU", True),
    ]
    perf_rows, pred_rows, history_rows = [], [], []
    pred_map = {}
    attn_weights = None
    trained = {}
    for name, rnn_type, attention in configs:
        result = train_model(name, rnn_type, attention, X_train, y_train, X_val, y_val, device)
        pred_scaled, weights = predict_model(result.model, X_val, device)
        pred = target_scaler.inverse_transform(pred_scaled)
        pred_map[name] = pred
        trained[name] = result.model
        if name == "Attention-GRU":
            attn_weights = weights
        for j, target in enumerate(TARGETS):
            m = metrics(y_val_raw[:, j], pred[:, j])
            perf_rows.append({
                "model": name,
                "target": TARGET_LABELS[target],
                "RMSE": round(m["RMSE"], 2),
                "MAE": round(m["MAE"], 2),
                "MAPE(%)": round(m["MAPE(%)"], 3),
                "R2": round(m["R2"], 4),
                "best_epoch": result.best_epoch,
                "fit_seconds": round(result.fit_seconds, 2),
                "device": str(device),
                "seq_len": SEQ_LEN,
            })
        for d, true, p in zip(val_dates, y_val_raw, pred):
            pred_rows.append({
                "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                "model": name,
                "true_load_mean": true[0],
                "true_load_max": true[1],
                "true_load_min": true[2],
                "pred_load_mean": p[0],
                "pred_load_max": p[1],
                "pred_load_min": p[2],
            })
        history_rows.append({
            "model": name,
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
            "best_epoch": result.best_epoch,
            "fit_seconds": result.fit_seconds,
        })
        torch.save(result.model.state_dict(), DL_DIR / f"{name.replace('-', '_')}.pt")

    perf_df = pd.DataFrame(perf_rows)
    pred_df = pd.DataFrame(pred_rows)
    perf_df.to_csv(OUT / "project2_deep_learning_performance.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUT / "project2_deep_learning_validation_predictions.csv", index=False, encoding="utf-8-sig")

    history_export = []
    for row in history_rows:
        for epoch, (tr, va) in enumerate(zip(row["train_losses"], row["val_losses"]), start=1):
            history_export.append({
                "model": row["model"],
                "epoch": epoch,
                "train_loss": tr,
                "val_loss": va,
                "best_epoch": row["best_epoch"],
                "fit_seconds": row["fit_seconds"],
            })
    pd.DataFrame(history_export).to_csv(OUT / "project2_deep_learning_training_history.csv", index=False, encoding="utf-8-sig")

    v5_perf = pd.read_csv(OUT / "project2_model_performance.csv", encoding="utf-8-sig")
    plot_loss(history_rows)
    plot_validation(val_dates, y_val_raw, pred_map)
    plot_attention(val_dates, attn_weights)
    plot_comparison(v5_perf, perf_df)

    best_model_name = perf_df.groupby("model")["RMSE"].mean().sort_values().index[0]
    best_model = trained[best_model_name]
    forecast_df = recursive_forecast(best_model, daily, feature_scaler, target_scaler, device)
    forecast_df.insert(2, "模型", best_model_name)
    forecast_df.to_csv(OUT / "project2_deep_learning_forecast_2015_01_11_17.csv", index=False, encoding="utf-8-sig")

    best_val_pred = pred_df[pred_df["model"] == best_model_name].copy()
    best_val_pred["date"] = pd.to_datetime(best_val_pred["date"])
    rolling_pred = best_val_pred.rename(columns={
        "pred_load_mean": "load_mean_pred",
        "pred_load_max": "load_max_pred",
        "pred_load_min": "load_min_pred",
    })[["date", "load_mean_pred", "load_max_pred", "load_min_pred"]]
    true_df = pd.DataFrame({
        "date": val_dates,
        "load_mean": y_val_raw[:, 0],
        "load_max": y_val_raw[:, 1],
        "load_min": y_val_raw[:, 2],
    })
    rolling_df = rolling_window_eval(rolling_pred, true_df)
    rolling_df.to_csv(OUT / "project2_rolling_4week_validation.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "version": "V6",
        "purpose": "PyTorch GRU/LSTM/Attention-GRU exploration; final main model remains V5 Stacking",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seq_len": SEQ_LEN,
        "train_period": f"<= {TRAIN_END.date()}",
        "validation_period": f"{VAL_START.date()} to {VAL_END.date()}",
        "models": [c[0] for c in configs],
        "best_deep_model_by_avg_rmse": best_model_name,
        "artifacts": {
            "performance": "project2_deep_learning_performance.csv",
            "validation_predictions": "project2_deep_learning_validation_predictions.csv",
            "training_history": "project2_deep_learning_training_history.csv",
            "forecast": "project2_deep_learning_forecast_2015_01_11_17.csv",
            "rolling_4week_validation": "project2_rolling_4week_validation.csv",
        },
    }
    (OUT / "project2_deep_learning_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"feature_scaler": feature_scaler, "target_scaler": target_scaler, "features": FEATURES, "targets": TARGETS}, DL_DIR / "deep_learning_scalers.joblib")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
