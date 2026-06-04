#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the V6 final second-part report and integration notes.

V6 keeps V5 Stacking as the final main model, and adds PyTorch GRU/LSTM/
Attention-GRU as a GPU-accelerated deep-learning exploration.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "project2"
REPORT_HTML = OUT / "report2_model_tuning_optimization.html"
REPORT_MD = OUT / "report2_model_tuning_optimization.md"
REPORT_TXT = OUT / "report2_model_tuning_optimization.txt"
INDEX = OUT / "report2_delivery_index.csv"
INTEGRATION = OUT / "part3_integration_notes.md"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name, encoding="utf-8-sig")


def read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def img_data(name: str) -> str:
    path = OUT / name
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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


def code_block(text: str) -> str:
    return f"<pre><code>{html.escape(text.strip())}</code></pre>"


def best_summary(perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, sub in perf.groupby("target", sort=False):
        ranked = sub.sort_values("RMSE").reset_index(drop=True)
        best = ranked.iloc[0]
        second = ranked.iloc[1]
        rows.append(
            {
                "目标": target,
                "最终主模型": best["model"],
                "验证RMSE": f"{best['RMSE']:.2f}",
                "MAE": f"{best['MAE']:.2f}",
                "MAPE(%)": f"{best['MAPE(%)']:.3f}",
                "R2": f"{best['R2']:.4f}",
                "次优模型": second["model"],
                "较次优RMSE降低(%)": f"{(second['RMSE'] - best['RMSE']) / second['RMSE'] * 100:.2f}",
                "结论": "保留为最终主模型",
            }
        )
    return pd.DataFrame(rows)


def optuna_best(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), sub in trials.groupby(["target", "model"], sort=False):
        best = sub.loc[sub["value"].idxmin()]
        rows.append(
            {
                "目标": target,
                "模型": model,
                "trials": len(sub),
                "最佳CV RMSE": f"{best['value']:.2f}",
                "最佳trial": int(best["trial_number"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["目标", "最佳CV RMSE"])


def dl_vs_stacking(v5_perf: pd.DataFrame, dl_perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = {"日平均负荷": "日平均负荷", "日最高负荷": "日最高负荷", "日最低负荷": "日最低负荷"}
    for target in labels:
        stack = v5_perf[(v5_perf["target"] == target) & (v5_perf["model"] == "Stacking")].iloc[0]
        rows.append(
            {
                "目标": target,
                "模型": "Stacking(V6最终主模型)",
                "RMSE": f"{stack['RMSE']:.2f}",
                "MAE": f"{stack['MAE']:.2f}",
                "MAPE(%)": f"{stack['MAPE(%)']:.3f}",
                "R2": f"{stack['R2']:.4f}",
                "定位": "最终主模型",
            }
        )
        for _, row in dl_perf[dl_perf["target"] == target].sort_values("RMSE").iterrows():
            rows.append(
                {
                    "目标": target,
                    "模型": row["model"],
                    "RMSE": f"{row['RMSE']:.2f}",
                    "MAE": f"{row['MAE']:.2f}",
                    "MAPE(%)": f"{row['MAPE(%)']:.3f}",
                    "R2": f"{row['R2']:.4f}",
                    "定位": "深度学习探索",
                }
            )
    return pd.DataFrame(rows)


def dl_summary(dl_perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in dl_perf.groupby("model", sort=False):
        rows.append(
            {
                "模型": model,
                "平均RMSE": f"{sub['RMSE'].mean():.2f}",
                "平均MAPE(%)": f"{sub['MAPE(%)'].mean():.3f}",
                "最佳epoch": int(sub["best_epoch"].iloc[0]),
                "训练秒数": f"{sub['fit_seconds'].iloc[0]:.2f}",
                "设备": sub["device"].iloc[0],
                "序列长度": int(sub["seq_len"].iloc[0]),
                "结论": "最佳深度模型" if model == sub.groupby("model")["RMSE"].mean().index[0] else "对比模型",
            }
        )
    result = pd.DataFrame(rows)
    best_model = dl_perf.groupby("model")["RMSE"].mean().sort_values().index[0]
    result["结论"] = result["模型"].map(lambda m: "最佳深度模型" if m == best_model else "对比模型")
    return result.sort_values("平均RMSE")


def rolling_summary(rolling: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, sub in rolling.groupby("target", sort=False):
        rows.append(
            {
                "目标": target,
                "窗口数": sub[["window_start", "window_end"]].drop_duplicates().shape[0],
                "平均4周RMSE": f"{sub['RMSE'].mean():.2f}",
                "最低窗口RMSE": f"{sub['RMSE'].min():.2f}",
                "最高窗口RMSE": f"{sub['RMSE'].max():.2f}",
                "用途": "模拟4周滚动验证稳定性",
            }
        )
    return pd.DataFrame(rows)


def final_forecast_comparison(stack_pred: pd.DataFrame, dl_pred: pd.DataFrame) -> pd.DataFrame:
    dl = dl_pred.rename(
        columns={
            "预测日平均(MW)": "GRU日平均(MW)",
            "预测日最高(MW)": "GRU日最高(MW)",
            "预测日最低(MW)": "GRU日最低(MW)",
        }
    )
    st = stack_pred.rename(
        columns={
            "预测日平均(MW)": "Stacking日平均(MW)",
            "预测日最高(MW)": "Stacking日最高(MW)",
            "预测日最低(MW)": "Stacking日最低(MW)",
        }
    )
    merged = st[["日期", "星期", "Stacking日平均(MW)", "Stacking日最高(MW)", "Stacking日最低(MW)"]].merge(
        dl[["日期", "GRU日平均(MW)", "GRU日最高(MW)", "GRU日最低(MW)"]], on="日期", how="left"
    )
    return merged


def make_integration_notes() -> str:
    return """# 第三部分对接说明：成品展示、网页设计与模型调用

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
"""


def make_markdown(tables: dict[str, pd.DataFrame], manifest: dict) -> str:
    lines = [
        "# 第二次汇报最终版：模型训练、调参与深度学习探索",
        "",
        "V6 是第二部分模型训练的最终版：保留 V5 的 Stacking 作为正式主模型，同时新增 PyTorch + GPU 的 GRU、LSTM 和轻量 Temporal Attention-GRU 探索实验。",
        "",
        "## 1. 最终结论",
        "",
        "- 最终主模型：Stacking。",
        "- 深度学习探索：GRU、LSTM、Attention-GRU，使用 PyTorch 和 GPU 训练。",
        f"- GPU 环境：{manifest.get('cuda_name')}，device={manifest.get('device')}。",
        "- 4 周滚动验证：新增滚动窗口检查，用于模拟短周期预测稳定性。",
        "",
        "## 2. Stacking 主模型表现",
        "",
        table_md(tables["best"]),
        "",
        "## 3. PyTorch 深度学习探索",
        "",
        table_md(tables["dl_summary"]),
        "",
        "## 4. Stacking 与深度学习对比",
        "",
        table_md(tables["dl_vs_stack"]),
        "",
        "## 5. 4周滚动验证",
        "",
        table_md(tables["rolling_summary"]),
        "",
        "## 6. 最终预测对比",
        "",
        table_md(tables["forecast_compare"]),
        "",
        "## 7. 对接说明",
        "",
        "详见 `part3_integration_notes.md`。",
    ]
    return "\n".join(lines) + "\n"


def make_html(tables: dict[str, pd.DataFrame], manifest: dict) -> str:
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
    .three-col{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
    .metric{background:#F7FAFC;border:1px solid #D7DEE8;border-radius:6px;padding:14px}
    .metric b{display:block;font-size:24px;color:#163B57}
    pre{white-space:pre-wrap;background:#0F172A;color:#DDE7F3;padding:18px;border-radius:6px;overflow:auto;font-family:Consolas,monospace;font-size:13px}
    .note{border-left:4px solid #1F6F8B;padding:10px 14px;background:#F0F7FA;color:#334155}
    @media(max-width:900px){.two-col,.three-col{grid-template-columns:1fr}main{padding:20px 14px}section{padding:20px 16px}}
    """
    pytorch_code = """
class TemporalAttention(nn.Module):
    def forward(self, hidden_states):
        score = self.attn(hidden_states).squeeze(-1)
        weight = torch.softmax(score, dim=1)
        context = torch.sum(hidden_states * weight.unsqueeze(-1), dim=1)
        return context, weight
"""
    handoff_json = """
{
  "main_model": "Stacking",
  "prediction_file": "project2_final_prediction_2015_01_11_17.csv",
  "deep_learning_reference": "project2_deep_learning_forecast_2015_01_11_17.csv",
  "display_report": "report2_model_tuning_optimization.html"
}
"""
    cards = "".join(
        f"<div class='metric'><span>{html.escape(row['目标'])}</span><b>{html.escape(row['验证RMSE'])}</b><small>{html.escape(row['最终主模型'])} / {html.escape(row['MAPE(%)'])}% MAPE</small></div>"
        for _, row in tables["best"].iterrows()
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>第二次汇报最终版：模型训练、调参与深度学习探索</title><style>{css}</style></head>
<body><main>
<section>
  <span class="tag">项目二</span><span class="tag">V6最终版</span><span class="tag">Stacking主模型</span><span class="tag">PyTorch GPU探索</span>
  <h1>短期电力负荷预测：模型训练、调参与深度学习探索</h1>
  <p class="lead">V6 是第二部分模型训练的最后优化版：最终主模型仍采用 Stacking，同时新增基于 PyTorch 的 GRU、LSTM 和轻量 Temporal Attention-GRU，用 GPU 训练并作为深度学习探索与答辩亮点。</p>
  <div class="three-col">{cards}</div>
</section>

<section>
  <h2>1. V6 最终建模结论</h2>
  <table class="data-table">
    <tr><th>问题</th><th>V6 决策</th><th>原因</th></tr>
    <tr><td>最终主模型</td><td>Stacking</td><td>三个目标整体最稳，验证 RMSE 和统计检验体系已经完整</td></tr>
    <tr><td>LSTM/GRU 是否必要</td><td>必要，但作为探索模型</td><td>展示 PyTorch + GPU + 时序神经网络能力，同时验证深度学习适用性</td></tr>
    <tr><td>Attention 是否必要</td><td>作为轻量添头</td><td>可解释不同历史天权重，但当前小数据下性能弱于 GRU/LSTM</td></tr>
    <tr><td>验证集是否改成4-6周</td><td>不替代原验证集，新增4周滚动验证</td><td>保留183天总体评估，同时模拟短周期业务预测稳定性</td></tr>
  </table>
  <p class="note">答辩表达建议：我们不仅使用了传统机器学习模型，还尝试了基于 PyTorch 的 GRU/LSTM 时序神经网络，并利用 GPU 加速训练，探索深度学习在电力负荷预测中的应用；最终基于数据结果理性选择 Stacking 作为主模型。</p>
</section>

<section>
  <h2>2. 传统机器学习主线：Stacking 保留为最终主模型</h2>
  <p>V5 已完成 Ridge、RandomForest、XGBoost、SVR、KNN 和 Stacking 的调参、验证、统计检验。V6 不推翻该结论，而是在其上补充深度学习探索。</p>
  <img src="{img_data('03_validation_fit.png')}" alt="Stacking验证拟合">
  {table_html(tables['best'])}
</section>

<section>
  <h2>3. PyTorch + GPU 深度学习探索</h2>
  <p>V6 新增三类多输出时序模型：GRU、LSTM、Attention-GRU。输入为过去 28 天的负荷、气象和周期特征序列，输出为日平均、日最高、日最低三个目标。训练环境为 {html.escape(str(manifest.get('cuda_name')))}，device={html.escape(str(manifest.get('device')))}。</p>
  <img src="{img_data('10_deep_learning_loss.png')}" alt="深度学习loss">
  {table_html(tables['dl_summary'])}
  <h3>轻量 Temporal Attention 结构</h3>
  {code_block(pytorch_code)}
</section>

<section>
  <h2>4. 深度学习验证拟合与 Attention 权重</h2>
  <div class="two-col">
    <div><h3>GRU/LSTM/Attention 验证拟合</h3><img src="{img_data('11_deep_learning_validation_fit.png')}" alt="深度学习验证拟合"></div>
    <div><h3>Attention-GRU 平均权重</h3><img src="{img_data('12_attention_weights.png')}" alt="Attention权重"></div>
  </div>
  <p class="note">结果显示：GRU/LSTM 在日最低负荷上有不错表现；Attention-GRU 当前表现较弱，说明小样本日粒度数据下注意力机制不一定带来稳定收益。它适合作为探索和可解释性补充，不作为主模型。</p>
</section>

<section>
  <h2>5. Stacking 与深度学习横向对比</h2>
  <p>对比结论很清楚：主模型仍选 Stacking；GRU/LSTM 是有效技术探索，其中 LSTM 在日最低负荷上表现突出，但整体不替代 Stacking。</p>
  <img src="{img_data('13_deep_learning_vs_stacking.png')}" alt="深度学习与Stacking对比">
  {table_html(tables['dl_vs_stack'])}
</section>

<section>
  <h2>6. 4周滚动验证：回答“是否扩大/调整验证集”</h2>
  <p>V6 没有把 183 天验证集缩小成 4-6 周，而是在保留总体评估的基础上新增 4 周滚动窗口评估。这样既有足够样本量，又能模拟实际业务中连续短期预测的稳定性。</p>
  {table_html(tables['rolling_summary'])}
</section>

<section>
  <h2>7. 最终预测：正式结果 vs 深度学习参考</h2>
  <p>正式提交仍使用 Stacking 预测。GRU 预测作为深度学习参考结果，可用于网页中展示“传统模型与深度学习模型的对比”。</p>
  <div class="two-col">
    <div><h3>正式 Stacking 预测</h3><img src="{img_data('04_final_prediction.png')}" alt="最终预测"></div>
    <div><h3>历史背景</h3><img src="{img_data('05_history_forecast_context.png')}" alt="历史背景"></div>
  </div>
  {table_html(tables['forecast_compare'])}
</section>

<section>
  <h2>8. 第三部分对接说明</h2>
  <p>后续网页展示、成品融合和模型调用应以 Stacking 为主线，以深度学习结果为扩展展示。推荐直接读取 CSV 做展示，若要在线预测再加载 joblib/pt 模型。</p>
  <table class="data-table">
    <tr><th>用途</th><th>文件</th><th>说明</th></tr>
    <tr><td>正式预测</td><td>project2_final_prediction_2015_01_11_17.csv</td><td>网页卡片和最终结果表优先使用</td></tr>
    <tr><td>深度学习参考</td><td>project2_deep_learning_forecast_2015_01_11_17.csv</td><td>用于展示 PyTorch GRU 对比</td></tr>
    <tr><td>主模型文件</td><td>models/load_*_Stacking.joblib</td><td>在线预测时加载</td></tr>
    <tr><td>深度学习模型</td><td>deep_learning/*.pt</td><td>展示/扩展用，不替代主模型</td></tr>
    <tr><td>详细对接说明</td><td>part3_integration_notes.md</td><td>交给第三部分同学使用</td></tr>
  </table>
  <h3>建议统一 JSON</h3>
  {code_block(handoff_json)}
</section>

<section>
  <h2>9. V6 汇报结论</h2>
  <ol>
    <li>Stacking 仍是最终主模型，作为正式预测输出。</li>
    <li>V6 新增 PyTorch + GPU 的 GRU/LSTM/Attention-GRU，提升答辩技术含量。</li>
    <li>GRU 是平均 RMSE 最好的深度学习模型；LSTM 在日最低负荷上表现最好。</li>
    <li>Attention-GRU 作为轻量注意力探索，提供可解释性，但当前不作为主模型。</li>
    <li>第三部分可直接读取 CSV/PNG/HTML 进行网页展示和项目融合。</li>
  </ol>
</section>
</main></body></html>"""


def main() -> None:
    v5_perf = read_csv("project2_model_performance.csv")
    trials = read_csv("project2_optuna_trials.csv")
    dl_perf = read_csv("project2_deep_learning_performance.csv")
    rolling = read_csv("project2_rolling_4week_validation.csv")
    stack_pred = read_csv("project2_final_prediction_2015_01_11_17.csv")
    dl_pred = read_csv("project2_deep_learning_forecast_2015_01_11_17.csv")
    manifest = read_json("project2_deep_learning_manifest.json")

    tables = {
        "best": best_summary(v5_perf),
        "optuna": optuna_best(trials),
        "dl_summary": dl_summary(dl_perf),
        "dl_vs_stack": dl_vs_stacking(v5_perf, dl_perf),
        "rolling_summary": rolling_summary(rolling),
        "forecast_compare": final_forecast_comparison(stack_pred, dl_pred),
    }
    REPORT_MD.write_text(make_markdown(tables, manifest), encoding="utf-8-sig")
    REPORT_TXT.write_text(REPORT_MD.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
    REPORT_HTML.write_text(make_html(tables, manifest), encoding="utf-8-sig")
    INTEGRATION.write_text(make_integration_notes(), encoding="utf-8-sig")

    pd.DataFrame(
        [
            {"file": REPORT_HTML.name, "purpose": "第二部分最终汇报 HTML"},
            {"file": REPORT_MD.name, "purpose": "第二部分最终汇报 Markdown"},
            {"file": "part3_integration_notes.md", "purpose": "第三部分网页/成品展示/模型调用对接说明"},
            {"file": "project2_final_prediction_2015_01_11_17.csv", "purpose": "正式 Stacking 预测"},
            {"file": "project2_deep_learning_performance.csv", "purpose": "PyTorch GRU/LSTM/Attention-GRU 结果"},
            {"file": "project2_rolling_4week_validation.csv", "purpose": "4周滚动验证"},
        ]
    ).to_csv(INDEX, index=False, encoding="utf-8-sig")
    print(f"Generated V6 final report: {REPORT_HTML}")


if __name__ == "__main__":
    main()
