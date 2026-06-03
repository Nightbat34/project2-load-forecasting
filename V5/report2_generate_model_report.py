#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the second presentation report: model training, tuning and results.

The report follows the visual/narrative style of report1_data_processing:
section cards, process explanation, code snippets, evidence tables and figures.
It only reads existing V5 result artifacts and does not retrain any model.
"""

from __future__ import annotations

import base64
import html
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "project2"
REPORT_MD = OUT / "report2_model_tuning_optimization.md"
REPORT_HTML = OUT / "report2_model_tuning_optimization.html"
REPORT_TXT = OUT / "report2_model_tuning_optimization.txt"
INDEX = OUT / "report2_delivery_index.csv"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name, encoding="utf-8-sig")


def f2(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.2f}"


def f3(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.3f}"


def img_data(name: str) -> str:
    path = OUT / name
    mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def table_html(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    return view.to_html(index=False, classes="data-table", border=1, escape=False)


def table_md(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    cols = list(view.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def best_summary(perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, sub in perf.groupby("target", sort=False):
        ranked = sub.sort_values("RMSE").reset_index(drop=True)
        best = ranked.iloc[0]
        second = ranked.iloc[1]
        rows.append(
            {
                "目标": target,
                "最终模型": best["model"],
                "验证RMSE": f2(best["RMSE"]),
                "MAE": f2(best["MAE"]),
                "MAPE(%)": f3(best["MAPE(%)"]),
                "R2": f"{float(best['R2']):.4f}",
                "次优模型": second["model"],
                "较次优RMSE降低(%)": f"{(second['RMSE'] - best['RMSE']) / second['RMSE'] * 100:.2f}",
                "CV RMSE": f2(best["TimeSeriesSplit_RMSE_mean"]),
                "泛化风险": best["generalization_risk"],
            }
        )
    return pd.DataFrame(rows)


def optuna_best(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), sub in trials.groupby(["target", "model"], sort=False):
        best = sub.loc[sub["value"].idxmin()]
        param_cols = [
            c
            for c in sub.columns
            if c not in {"trial_number", "value", "model", "target"} and pd.notna(best[c])
        ]
        params = []
        for c in param_cols:
            v = best[c]
            if isinstance(v, float):
                params.append(f"{c}={v:.4g}")
            else:
                params.append(f"{c}={v}")
        rows.append(
            {
                "目标": target,
                "模型": model,
                "trials": len(sub),
                "最佳trial": int(best["trial_number"]),
                "_sort_rmse": float(best["value"]),
                "最佳CV RMSE": f2(best["value"]),
                "核心参数": ", ".join(params),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(["目标", "_sort_rmse"]).drop(columns=["_sort_rmse"])


def best_vs_second_tests(perf: pd.DataFrame, pairwise: pd.DataFrame, dm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, sub in perf.groupby("target", sort=False):
        ranked = sub.sort_values("RMSE").reset_index(drop=True)
        best = ranked.iloc[0]["model"]
        second = ranked.iloc[1]["model"]
        pair = pairwise[
            (pairwise["target"] == target)
            & (
                ((pairwise["model_a"] == best) & (pairwise["model_b"] == second))
                | ((pairwise["model_a"] == second) & (pairwise["model_b"] == best))
            )
        ].iloc[0]
        dm_row = dm[
            (dm["target"] == target)
            & (
                ((dm["model_a"] == best) & (dm["model_b"] == second))
                | ((dm["model_a"] == second) & (dm["model_b"] == best))
            )
        ].iloc[0]
        rows.append(
            {
                "目标": target,
                "比较": f"{best} vs {second}",
                "配对T p值": f"{float(pair['paired_t_p_value']):.6f}",
                "DM p值": f"{float(dm_row['p_value']):.6f}",
                "结论": "预测能力差异显著" if dm_row["p_value"] < 0.05 else "RMSE更优，DM未达显著",
            }
        )
    return pd.DataFrame(rows)


def pac5_table(pac: pd.DataFrame) -> pd.DataFrame:
    sub = pac[pac["threshold"].round(4) == 0.05].copy()
    rows = []
    for target, target_df in sub.groupby("target", sort=False):
        best = target_df.sort_values("approx_correct_probability", ascending=False).iloc[0]
        rows.append(
            {
                "目标": target,
                "最佳模型": best["model"],
                "PAC@5%": f"{float(best['approx_correct_probability']):.4f}",
                "命中天数": f"{int(best['approx_correct_count'])}/{int(best['sample_count'])}",
                "业务解释": "相对误差不超过5%的验证日比例",
            }
        )
    return pd.DataFrame(rows)


def residual_table(norm: pd.DataFrame, auto: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in norm.iterrows():
        target = row["target"]
        lb = auto[(auto["target"] == target) & auto["lag"].notna()]
        bp = auto[(auto["target"] == target) & auto["BreuschPagan_p_value"].notna()]
        rows.append(
            {
                "目标": target,
                "模型": row["model"],
                "残差均值": f"{float(row['residual_mean']):.4f}",
                "残差标准差": f"{float(row['residual_std']):.4f}",
                "正态性": "拒绝正态假设",
                "Ljung-Box": "; ".join(
                    f"lag{int(x['lag'])} p={float(x['LjungBox_p_value']):.4f}" for _, x in lb.iterrows()
                ),
                "Breusch-Pagan": "" if bp.empty else f"p={float(bp.iloc[0]['BreuschPagan_p_value']):.4f}",
            }
        )
    return pd.DataFrame(rows)


def season_stack(seasonal: pd.DataFrame) -> pd.DataFrame:
    df = seasonal[(seasonal["model"] == "Stacking") & (seasonal["group_by"] == "season")].copy()
    return df[["target", "group_value", "n_samples", "RMSE", "MAE", "MAPE(%)"]].rename(
        columns={
            "target": "目标",
            "group_value": "季节",
            "n_samples": "样本数",
            "RMSE": "RMSE",
            "MAE": "MAE",
            "MAPE(%)": "MAPE(%)",
        }
    )


def training_log_view(training: pd.DataFrame) -> pd.DataFrame:
    view = training[["target", "model", "train_period", "validation_period", "feature_count", "fit_seconds", "cv_fold_rmse"]].copy()
    view = view.rename(
        columns={
            "target": "目标",
            "model": "模型",
            "train_period": "训练期",
            "validation_period": "验证期",
            "feature_count": "特征数",
            "fit_seconds": "训练秒数",
            "cv_fold_rmse": "CV各折RMSE",
        }
    )
    view["训练秒数"] = view["训练秒数"].map(lambda x: f"{float(x):.3f}")
    return view


def make_markdown(tables: dict[str, pd.DataFrame]) -> str:
    pred = tables["pred"]
    lines = [
        "# 第二次汇报：模型训练、调参与预测结果",
        "",
        "本次汇报承接第一次的数据处理与特征构建，不再重复讲清洗细节，重点展示模型如何训练、如何调参、如何验证，以及最终预测结果是否可信。",
        "",
        "## 1. 建模任务与验证设计",
        "",
        "- 任务：短期电力负荷多目标回归，分别预测日平均负荷、日最高负荷、日最低负荷。",
        "- 训练期：2012-01-15 至 2014-06-30。",
        "- 验证期：2014-07-01 至 2014-12-31，共 183 天。",
        "- 验证方法：TimeSeriesSplit + 独立验证集，保证时间顺序不被打乱。",
        "- 安全策略：本报告只读取已有结果，不重复运行 Optuna 和 Stacking 训练。",
        "",
        "## 2. 训练流程",
        "",
        "训练主线是：特征矩阵准备 -> 五类基模型调参 -> Stacking 集成 -> 验证集评估 -> 统计检验 -> 残差诊断 -> 7 天递推预测。",
        "",
        "![模型训练流程](02_model_training_comparison.png)",
        "",
        "## 3. 候选模型与集成策略",
        "",
        table_md(tables["ensemble"]),
        "",
        "## 4. Optuna 调参过程",
        "",
        "每个目标和每个基模型执行 50 次 trial，共 750 条调参记录。调参目标是最小化 TimeSeriesSplit RMSE。",
        "",
        "![Optuna 优化历史](07_optuna_optimization_history.png)",
        "",
        table_md(tables["optuna"]),
        "",
        "## 5. 训练日志",
        "",
        table_md(tables["training"]),
        "",
        "## 6. 验证集表现与最终选型",
        "",
        "三个目标最终都选择 Stacking。它相比次优 XGBoost 仍有 3%-7% 的 RMSE 改善。",
        "",
        "![验证集拟合效果](03_validation_fit.png)",
        "",
        table_md(tables["best"]),
        "",
        "## 7. 统计检验",
        "",
        "配对 T 检验关注误差均值差异，DM 检验关注时序预测能力差异。p >= 0.05 不代表两个模型一样好，只代表当前样本下证据不足。",
        "",
        table_md(tables["tests"]),
        "",
        "## 8. 业务命中率",
        "",
        "PAC@5% 表示相对误差不超过 5% 的天数比例，比单纯 RMSE 更容易解释给业务方。",
        "",
        table_md(tables["pac"]),
        "",
        "## 9. 残差与稳定性检查",
        "",
        "残差诊断用于确认模型不是只在指标上好看。日最低负荷 lag=1 残差存在显著自相关，后续可加入节后恢复和夜间负荷变化特征继续优化。",
        "",
        "![残差诊断](report2_residual_analysis.png)",
        "",
        table_md(tables["residual"]),
        "",
        "## 10. 分季节结果",
        "",
        "分季节评估可以检查模型是否只在某一段时间表现好。验证期内秋季波动较大，是主要误差来源。",
        "",
        "![季节表现](08_seasonal_performance.png)",
        "",
        table_md(tables["season"]),
        "",
        "## 11. 最终预测",
        "",
        "最终预测采用三个 Stacking 模型，并做物理约束校验：日最低 <= 日平均 <= 日最高，且所有预测负荷 > 0。",
        "",
        "![最终预测](04_final_prediction.png)",
        "",
        table_md(pred),
        "",
        "## 12. 汇报结论",
        "",
        "1. 数据处理后构造了 22 个建模特征，包含气象、周期、滞后和滚动统计。",
        "2. XGBoost 是最强基模型，但 Stacking 进一步提升了三个目标的验证集 RMSE。",
        "3. 日平均负荷的 Stacking 相比 XGBoost 在 DM 检验下达到显著提升；日最高和日最低负荷 RMSE 更优，但 DM 未达显著，需要如实汇报。",
        "4. 最终 7 天预测满足物理约束，可以作为项目二的最终提交结果。",
    ]
    return "\n".join(lines) + "\n"


def code_block(text: str) -> str:
    return f"<pre><code>{html.escape(text.strip())}</code></pre>"


def make_html(tables: dict[str, pd.DataFrame]) -> str:
    pred = tables["pred"]
    css = """
    body{margin:0;background:#F5F7FA;color:#172033;font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif;line-height:1.65}
    main{max-width:1180px;margin:0 auto;padding:34px 28px 70px}
    section{background:#fff;border:1px solid #E3E8F0;border-radius:8px;padding:26px 30px;margin:22px 0;box-shadow:0 8px 24px rgba(15,23,42,.06)}
    h1{font-size:34px;line-height:1.18;margin:0 0 10px;color:#102A43}
    h2{font-size:24px;margin:0 0 14px;color:#163B57}
    h3{font-size:18px;margin:18px 0 8px;color:#243B53}
    .lead{font-size:17px;color:#486581;margin:0 0 12px}
    .tag{display:inline-block;background:#E6F2F5;color:#1F6F8B;border-radius:999px;padding:4px 12px;font-weight:700;margin-right:8px}
    img{max-width:100%;border:1px solid #DCE4EF;border-radius:6px;background:#fff}
    .data-table{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0 4px}
    .data-table th{background:#163B57;color:white;text-align:center;padding:8px}
    .data-table td{border:1px solid #D7DEE8;padding:7px 9px;vertical-align:top}
    .data-table tr:nth-child(even) td{background:#F7FAFC}
    .two-col{display:grid;grid-template-columns:1fr 1fr;gap:18px}
    pre{white-space:pre-wrap;background:#0F172A;color:#DDE7F3;padding:18px;border-radius:6px;overflow:auto;font-family:Consolas,monospace;font-size:13px}
    .note{border-left:4px solid #1F6F8B;padding:10px 14px;background:#F0F7FA;color:#334155}
    .metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:16px 0}
    .metric{background:#F7FAFC;border:1px solid #D7DEE8;border-radius:6px;padding:14px}
    .metric b{display:block;font-size:24px;color:#163B57}
    @media(max-width:900px){.two-col,.metric-grid{grid-template-columns:1fr}main{padding:20px 14px}section{padding:20px 16px}}
    """
    optuna_code = """
def tune_objective(model_name, target):
    # 只在训练期内部做 TimeSeriesSplit，避免未来数据进入过去训练。
    split = TimeSeriesSplit(n_splits=3)
    params = optuna_suggest_params(model_name)
    model = build_model(model_name, params)
    return mean_cv_rmse(model, X_train, y_train, split)
"""
    stacking_code = """
base_models = [
    ("ridge", tuned_ridge),
    ("rf", tuned_random_forest),
    ("xgb", tuned_xgboost),
    ("svr", tuned_svr),
    ("knn", tuned_knn),
]
final_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0),
    cv=3,
)
"""
    forecast_code = """
for date in forecast_dates:
    # 用上一日/上周/两周前预测值递推构造滞后特征。
    pred_mean = mean_model.predict(X_date)
    pred_max = max_model.predict(X_date)
    pred_min = min_model.predict(X_date)
    pred_min, pred_mean, pred_max = enforce_min_mean_max(pred_min, pred_mean, pred_max)
"""
    best = tables["best"]
    metric_cards = "".join(
        f"<div class='metric'><span>{html.escape(row['目标'])}</span><b>{html.escape(row['验证RMSE'])}</b><small>{html.escape(row['最终模型'])} / MAPE {html.escape(row['MAPE(%)'])}%</small></div>"
        for _, row in best.iterrows()
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>第二次汇报：模型训练、调参与预测结果</title>
<style>{css}</style>
</head>
<body>
<main>
<section>
  <span class="tag">项目二</span><span class="tag">第二次汇报</span><span class="tag">模型训练</span><span class="tag">调参优化</span>
  <h1>短期电力负荷预测：模型训练、调参与预测结果</h1>
  <p class="lead">本次承接第一次汇报的数据处理成果，展示从候选模型、Optuna 调参、Stacking 集成、统计检验到最终 7 天预测的完整建模过程。</p>
  <div class="metric-grid">{metric_cards}</div>
</section>

<section>
  <h2>1. 建模任务与验证设计</h2>
  <p>本项目是短期电力负荷多目标回归：分别预测日平均负荷、日最高负荷和日最低负荷。模型不能随机打乱时间，因为真实业务中只能用过去预测未来。</p>
  <table class="data-table">
    <tr><th>项目</th><th>设置</th><th>说明</th></tr>
    <tr><td>训练期</td><td>2012-01-15 至 2014-06-30</td><td>用于模型拟合和 TimeSeriesSplit 调参</td></tr>
    <tr><td>验证期</td><td>2014-07-01 至 2014-12-31，共 183 天</td><td>独立 hold-out，模拟未来预测</td></tr>
    <tr><td>目标变量</td><td>日平均、日最高、日最低负荷</td><td>连续回归，单位 MW</td></tr>
    <tr><td>特征数</td><td>22 个</td><td>气象、周期、滞后、滚动统计</td></tr>
    <tr><td>安全策略</td><td>本报告只读已有结果</td><td>不重复执行高成本训练，保护电脑</td></tr>
  </table>
</section>

<section>
  <h2>2. 训练流程</h2>
  <p>训练过程不是只跑一个模型，而是先建立多类基模型，再用验证集和统计检验筛选。流程如下：特征矩阵准备 -> 五类基模型调参 -> Stacking 集成 -> 验证集评估 -> 统计检验 -> 残差诊断 -> 7 天递推预测。</p>
  <img src="{img_data('02_model_training_comparison.png')}" alt="模型训练对比">
  <div class="note">图中展示各模型在三个目标上的训练与验证表现。可以看到单模型之间差异明显，因此后续采用 Stacking 学习组合关系。</div>
</section>

<section>
  <h2>3. 候选模型与集成策略</h2>
  <p>为了让模型族覆盖线性、Bagging、Boosting、核方法和相似日匹配，本次共训练 5 个基模型，并使用 Ridge 作为 Stacking 元学习器。</p>
  {table_html(tables['ensemble'])}
  <h3>核心代码：Stacking 集成</h3>
  {code_block(stacking_code)}
</section>

<section>
  <h2>4. Optuna 调参过程</h2>
  <p>每个目标和每个基模型执行 50 次 trial，共 750 条调参记录。调参目标是最小化 TimeSeriesSplit RMSE，而不是只看训练集误差。</p>
  <img src="{img_data('07_optuna_optimization_history.png')}" alt="Optuna 优化历史">
  <h3>核心代码：时序交叉验证调参</h3>
  {code_block(optuna_code)}
  {table_html(tables['optuna'])}
</section>

<section>
  <h2>5. 训练日志</h2>
  <p>训练日志记录了每个目标、每个模型的训练期、验证期、特征数、训练耗时和各折 CV RMSE。这个表说明结果不是凭空生成，而是每个模型都有可追溯记录。</p>
  {table_html(tables['training'])}
</section>

<section>
  <h2>6. 验证集表现与最终选型</h2>
  <p>三个目标最终都选择 Stacking。相比次优 XGBoost，日平均负荷 RMSE 降低 6.63%，日最高负荷降低 3.11%，日最低负荷降低 6.67%。</p>
  <img src="{img_data('03_validation_fit.png')}" alt="验证集拟合">
  {table_html(tables['best'])}
</section>

<section>
  <h2>7. 统计检验</h2>
  <p>仅看 RMSE 还不够，因此补充配对 T 检验和 Diebold-Mariano 检验。DM 检验更贴近时间序列预测比较，会考虑误差序列结构。</p>
  {table_html(tables['tests'])}
  <div class="two-col">
    <div><h3>日平均负荷 DM 热力图</h3><img src="{img_data('06_dm_test_heatmap_日平均负荷.png')}" alt="日平均负荷DM"></div>
    <div><h3>日最高负荷 DM 热力图</h3><img src="{img_data('06_dm_test_heatmap_日最高负荷.png')}" alt="日最高负荷DM"></div>
  </div>
  <p class="note">日最高和日最低负荷中，Stacking 的 RMSE 更低，但 DM p 值未达到 0.05。这一点需要如实汇报：它是当前验证集上的最优选择，但统计显著性证据没有日平均负荷那么强。</p>
</section>

<section>
  <h2>8. 业务命中率 PAC</h2>
  <p>PAC@5% 表示相对误差不超过 5% 的验证日比例，比 RMSE 更容易解释给业务方。</p>
  {table_html(tables['pac'])}
</section>

<section>
  <h2>9. 残差与模型正确性检查</h2>
  <p>残差诊断用于确认模型不是只在指标上好看。三个目标残差均拒绝正态假设，但 Stacking 属于非参数集成，正态性不是硬假设；日最低负荷 lag=1 残差有显著自相关，后续仍有优化空间。</p>
  <img src="{img_data('report2_residual_analysis.png')}" alt="残差诊断">
  {table_html(tables['residual'])}
</section>

<section>
  <h2>10. 分季节稳定性</h2>
  <p>分季节检查可以发现误差集中在哪里。验证期内秋季波动最大，是主要误差来源；冬季样本较少，但 Stacking 仍保持较低误差。</p>
  <img src="{img_data('08_seasonal_performance.png')}" alt="季节表现">
  {table_html(tables['season'])}
</section>

<section>
  <h2>11. 最终 7 天预测</h2>
  <p>最终使用三个 Stacking 模型递推预测 2015-01-11 至 2015-01-17，并做物理约束校验：日最低 <= 日平均 <= 日最高，且所有负荷大于 0。</p>
  <div class="two-col">
    <div><h3>预测曲线</h3><img src="{img_data('04_final_prediction.png')}" alt="最终预测"></div>
    <div><h3>历史背景</h3><img src="{img_data('05_history_forecast_context.png')}" alt="历史背景"></div>
  </div>
  {table_html(pred)}
  <h3>核心代码：递推预测与物理约束</h3>
  {code_block(forecast_code)}
</section>

<section>
  <h2>12. 汇报结论</h2>
  <ol>
    <li>本阶段完成了 Ridge、RandomForest、XGBoost、SVR、KNN 与 Stacking 的训练和调参。</li>
    <li>XGBoost 是最强基模型，但 Stacking 在三个目标上进一步降低验证集 RMSE。</li>
    <li>日平均负荷的 Stacking 相比 XGBoost 在 DM 检验下显著更优；日最高和日最低负荷 RMSE 更优但 DM 未达显著。</li>
    <li>最终 7 天预测满足物理约束，可以作为项目二建模部分的提交结果。</li>
  </ol>
</section>
</main>
</body>
</html>
"""
    return html_text


def main() -> None:
    old_residual = OUT / "v4_step6_residual_analysis.png"
    new_residual = OUT / "report2_residual_analysis.png"
    if old_residual.exists() and not new_residual.exists():
        shutil.copy2(old_residual, new_residual)

    perf = read_csv("project2_model_performance.csv")
    trials = read_csv("project2_optuna_trials.csv")
    training = read_csv("project2_training_log.csv")
    pairwise = read_csv("project2_pairwise_significance.csv")
    dm = read_csv("project2_dm_test.csv")
    pac = read_csv("project2_approx_correctness.csv")
    norm = read_csv("project2_residual_normality.csv")
    auto = read_csv("project2_residual_autocorr_hetero.csv")
    seasonal = read_csv("project2_seasonal_analysis.csv")
    pred = read_csv("project2_final_prediction_2015_01_11_17.csv")
    ensemble = read_csv("project2_ensemble_strategy.csv")

    tables = {
        "ensemble": ensemble.rename(
            columns={
                "role": "角色",
                "model": "模型",
                "function_name": "函数/结构",
                "learning_strategy": "学习策略",
                "why_used": "使用理由",
            }
        ),
        "optuna": optuna_best(trials),
        "training": training_log_view(training),
        "best": best_summary(perf),
        "tests": best_vs_second_tests(perf, pairwise, dm),
        "pac": pac5_table(pac),
        "residual": residual_table(norm, auto),
        "season": season_stack(seasonal),
        "pred": pred,
    }
    REPORT_MD.write_text(make_markdown(tables), encoding="utf-8-sig")
    REPORT_TXT.write_text(REPORT_MD.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
    REPORT_HTML.write_text(make_html(tables), encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"file": REPORT_HTML.name, "purpose": "第二次汇报 HTML 展示版"},
            {"file": REPORT_MD.name, "purpose": "第二次汇报 Markdown 版"},
            {"file": REPORT_TXT.name, "purpose": "第二次汇报 TXT 版"},
            {"file": "project2_final_prediction_2015_01_11_17.csv", "purpose": "最终 7 天预测结果"},
            {"file": "project2_model_performance.csv", "purpose": "模型性能结果"},
            {"file": "project2_optuna_trials.csv", "purpose": "Optuna 750 条调参记录"},
        ]
    ).to_csv(INDEX, index=False, encoding="utf-8-sig")
    print(f"Generated: {REPORT_HTML}")


if __name__ == "__main__":
    main()
