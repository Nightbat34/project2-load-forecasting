"""
Generate the first practicum presentation package: data processing only.

Outputs are written to output/report1_data_processing and are intended for
teacher review before any GitHub upload.
"""

from __future__ import annotations

import base64
import html
import inspect
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import project2_forecast as p2


ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("LOAD_DATA_PATH", ROOT.parent / "Data" / "附件1-电网负荷数据.xlsx"))
OUT = ROOT / "output" / "report1_data_processing"
ASSETS = OUT / "assets"


def ensure_output_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)


def set_chinese_font() -> None:
    """Try common Chinese fonts so exported images do not show tofu squares."""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_df_image(df: pd.DataFrame, path: Path, title: str, max_rows: int = 8) -> None:
    """Render a small dataframe as a screenshot-like PNG for the report."""
    shown = df.head(max_rows).copy()
    shown = shown.fillna("")
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: "" if x == "" else f"{x:.2f}")

    fig_h = 1.25 + 0.42 * (len(shown) + 1)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=12)
    table = ax.table(
        cellText=shown.astype(str).values,
        colLabels=[str(c) for c in shown.columns],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#D7DEE8")
        if row == 0:
            cell.set_facecolor("#163B57")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F7FAFC" if row % 2 else "white")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_code_image(func, path: Path, title: str, width: int = 112) -> str:
    """Render important code as a PNG and return the raw source text."""
    source = inspect.getsource(func)
    source = textwrap.dedent(source).rstrip()
    wrapped_lines: list[str] = []
    for line in source.splitlines():
        if len(line) <= width:
            wrapped_lines.append(line)
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=width, subsequent_indent="    "))

    fig_h = max(4.4, 0.23 * (len(wrapped_lines) + 4))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    ax.set_facecolor("#0F172A")
    fig.patch.set_facecolor("#0F172A")
    ax.text(0.02, 0.97, title, va="top", ha="left", fontsize=17, color="#F8FAFC", fontweight="bold")
    ax.text(
        0.02,
        0.88,
        "\n".join(wrapped_lines),
        va="top",
        ha="left",
        fontsize=10.5,
        color="#DDE7F3",
        family="Microsoft YaHei",
        linespacing=1.25,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return source


def image_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def table_html(df: pd.DataFrame, max_rows: int = 12, float_fmt: str = "{:.2f}") -> str:
    shown = df.head(max_rows).copy()
    formatters = {
        col: (lambda x, fmt=float_fmt: "" if pd.isna(x) else fmt.format(x))
        for col in shown.columns
        if pd.api.types.is_float_dtype(shown[col])
    }
    return shown.to_html(index=False, classes="data-table", escape=False, formatters=formatters)


def markdown_table(df: pd.DataFrame) -> str:
    """Small dependency-free Markdown table renderer."""
    cols = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        rendered = []
        for value in row:
            if pd.isna(value):
                rendered.append("")
            else:
                rendered.append(str(value).replace("\n", " ").replace("|", "\\|"))
        rows.append(rendered)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def profile_data(load_raw: pd.DataFrame, weather_raw: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    time_cols = [c for c in load_raw.columns if c != "YMD"]
    target_mask = daily["date"].between("2015-01-11", "2015-01-17")
    return pd.DataFrame(
        [
            {
                "检查项": "Area_Load 原始负荷表",
                "结果": f"{load_raw.shape[0]} 行 x {load_raw.shape[1]} 列",
                "说明": "每天一行，96 个 15 分钟采样点。",
            },
            {
                "检查项": "Area_Weather 原始天气表",
                "结果": f"{weather_raw.shape[0]} 行 x {weather_raw.shape[1]} 列",
                "说明": "每天一行，包含最高/最低/平均温度、湿度、降雨量。",
            },
            {
                "检查项": "负荷采样列数",
                "结果": f"{len(time_cols)} 个",
                "说明": "96 = 24 小时 x 每小时 4 个 15 分钟点。",
            },
            {
                "检查项": "预测目标期负荷缺失",
                "结果": f"{int(load_raw.loc[load_raw['YMD'].between(20150111, 20150117), time_cols].isna().sum().sum())} 个单元格",
                "说明": "正好 7 天 x 96 点，是题目要求预测的目标，不作为异常删除。",
            },
            {
                "检查项": "目标期天气是否完整",
                "结果": f"{int(daily.loc[target_mask, ['temp_max', 'temp_min', 'temp_avg', 'humidity', 'rainfall']].isna().sum().sum())} 个缺失",
                "说明": "递推预测时可以使用目标期已给定天气。",
            },
            {
                "检查项": "建模日粒度表",
                "结果": f"{daily.shape[0]} 行 x {daily.shape[1]} 列",
                "说明": "由 15 分钟负荷聚合为日最大、日最小、日平均。",
            },
        ]
    )


def plot_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.axis("off")
    steps = [
        ("原始 Excel", "Area_Load 宽表\nArea_Weather 天气表"),
        ("清洗重构", "melt 展开 96 个时刻\nYMD 转日期"),
        ("日粒度聚合", "load_max\nload_min\nload_mean"),
        ("天气合并", "按 YMD 左连接\n温度/湿度/降雨"),
        ("特征工程", "日历周期\nHDD/CDD\n滞后与滚动特征"),
    ]
    x0, y, w, h, gap = 0.04, 0.42, 0.15, 0.28, 0.045
    colors = ["#163B57", "#1F6F8B", "#3A8B6D", "#B07D2B", "#6E5FA8"]
    for i, (title, detail) in enumerate(steps):
        x = x0 + i * (w + gap)
        box = plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=colors[i], edgecolor="none")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.67, title, color="white", fontsize=15, fontweight="bold", ha="center")
        ax.text(x + w / 2, y + h * 0.33, detail, color="white", fontsize=10.5, ha="center", va="center")
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + w + gap * 0.75, y + h / 2),
                xytext=(x + w + gap * 0.15, y + h / 2),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#334155"),
            )
    ax.text(0.04, 0.86, "数据处理主线：从 15 分钟负荷宽表到可建模的日粒度特征表", fontsize=19, fontweight="bold")
    ax.text(
        0.04,
        0.18,
        "关键原则：只用预测日前已经知道的信息构造滞后/滚动特征；2015-01-11 至 2015-01-17 的负荷缺失是预测目标。",
        fontsize=12.5,
        color="#334155",
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_missingness(load_raw: pd.DataFrame, weather_raw: pd.DataFrame, path: Path) -> None:
    time_cols = [c for c in load_raw.columns if c != "YMD"]
    load_missing = load_raw.set_index("YMD")[time_cols].isna().sum(axis=1)
    weather_named = weather_raw.copy()
    weather_named.columns = ["YMD", "temp_max", "temp_min", "temp_avg", "humidity", "rainfall"]
    weather_missing = weather_named.set_index("YMD").drop(columns=[]).isna().sum(axis=1)
    view = pd.DataFrame(
        {
            "date": pd.to_datetime(load_missing.index.astype(str), format="%Y%m%d"),
            "load_missing_cells": load_missing.values,
            "weather_missing_cells": weather_missing.reindex(load_missing.index).fillna(0).values,
        }
    )
    focus = view[view["date"].between("2014-12-25", "2015-01-17")]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(focus["date"], focus["load_missing_cells"], width=0.8, color="#BA3E3E", label="负荷缺失单元格")
    ax.plot(focus["date"], focus["weather_missing_cells"], color="#1F6F8B", marker="o", label="天气缺失单元格")
    ax.axvspan(pd.Timestamp("2015-01-11"), pd.Timestamp("2015-01-17"), color="#F8C471", alpha=0.25)
    ax.set_title("缺失值检查：目标期负荷缺失，天气完整", loc="left", fontsize=17, fontweight="bold")
    ax.set_ylabel("缺失单元格数量")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.autofmt_xdate(rotation=35)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_2014_daily_load(daily: pd.DataFrame, path: Path) -> None:
    d2014 = daily[daily["date"].dt.year == 2014].copy()
    fig, ax = plt.subplots(figsize=(14, 5.6))
    ax.plot(d2014["date"], d2014["load_max"], color="#BA3E3E", lw=1.8, label="日最高负荷")
    ax.plot(d2014["date"], d2014["load_mean"], color="#1F6F8B", lw=1.8, label="日平均负荷")
    ax.plot(d2014["date"], d2014["load_min"], color="#3A8B6D", lw=1.8, label="日最低负荷")
    ax.set_title("2014 年日负荷曲线：保留季节性、周周期和极值变化", loc="left", fontsize=17, fontweight="bold")
    ax.set_ylabel("负荷")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, loc="upper left")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_load_duration(daily: pd.DataFrame, path: Path) -> None:
    d2014 = daily[daily["date"].dt.year == 2014].copy()
    values = {
        "日最高负荷": np.sort(d2014["load_max"].dropna().values)[::-1],
        "日平均负荷": np.sort(d2014["load_mean"].dropna().values)[::-1],
        "日最低负荷": np.sort(d2014["load_min"].dropna().values)[::-1],
    }
    fig, ax = plt.subplots(figsize=(14, 5.6))
    for label, arr in values.items():
        ax.plot(np.arange(1, len(arr) + 1), arr, lw=2, label=label)
    ax.set_title("2014 年负荷持续曲线：观察全年高低负荷分布", loc="left", fontsize=17, fontweight="bold")
    ax.set_xlabel("按负荷从高到低排序后的天数")
    ax.set_ylabel("负荷")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_blocks(path: Path) -> None:
    blocks = [
        ("基础目标", ["load_max", "load_min", "load_mean"], "#163B57"),
        ("天气特征", ["temp_max", "temp_min", "temp_avg", "humidity", "rainfall"], "#1F6F8B"),
        ("衍生天气", ["temp_range", "hdd", "cdd"], "#3A8B6D"),
        ("日历周期", ["dayofweek", "month_sin/cos", "dow_sin/cos", "doy_sin/cos"], "#B07D2B"),
        ("时序记忆", ["lag_1", "lag_7", "lag_14", "roll_mean/std"], "#6E5FA8"),
    ]
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.axis("off")
    ax.text(0.04, 0.88, "特征工程结构：让模型同时看到天气、日历和历史负荷惯性", fontsize=18, fontweight="bold")
    for i, (title, items, color) in enumerate(blocks):
        x = 0.05 + i * 0.19
        ax.add_patch(plt.Rectangle((x, 0.34), 0.16, 0.34, transform=ax.transAxes, facecolor=color, alpha=0.95))
        ax.text(x + 0.08, 0.61, title, color="white", fontsize=14, fontweight="bold", ha="center")
        ax.text(x + 0.08, 0.46, "\n".join(items), color="white", fontsize=10.2, ha="center", va="center")
    ax.text(
        0.05,
        0.16,
        "说明：滚动统计使用 shift(1) 后再 rolling，表示今天预测时只使用昨天及以前的真实历史，避免数据泄漏。",
        fontsize=12.5,
        color="#334155",
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_html(profile: pd.DataFrame, assets: dict[str, Path], code_sources: dict[str, str]) -> None:
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
    @media(max-width:900px){.two-col{grid-template-columns:1fr}main{padding:20px 14px}section{padding:20px 16px}}
    """
    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>第一次汇报：数据处理与特征构建</title>
<style>{css}</style>
</head>
<body>
<main>
<section>
  <span class="tag">项目二</span><span class="tag">第一次汇报</span><span class="tag">数据处理</span>
  <h1>短期电力负荷预测：数据处理与特征构建</h1>
  <p class="lead">本次只汇报数据处理部分：原始数据理解、清洗重构、缺失值判断、2014 年负荷分析、天气合并、建模特征表构建。算法训练和模型选择放到第二次汇报。</p>
</section>

<section>
  <h2>1. 原始数据理解</h2>
  <p>附件 Excel 包含两个核心工作表：<b>Area_Load</b> 是电力负荷，<b>Area_Weather</b> 是天气。负荷表中每天有 96 个 15 分钟采样点，因此先要把宽表转换成可分析的日粒度数据。</p>
  {table_html(profile, max_rows=10)}
  <div class="two-col">
    <div><h3>Area_Load 原始表截图</h3><img src="{image_uri(assets['raw_load'])}"></div>
    <div><h3>Area_Weather 原始表截图</h3><img src="{image_uri(assets['raw_weather'])}"></div>
  </div>
</section>

<section>
  <h2>2. 数据处理主线</h2>
  <p>处理流程从原始 Excel 出发，先重构负荷表，再聚合为题目要求预测的日最高、日最低、日平均负荷，并与天气表合并。</p>
  <img src="{image_uri(assets['pipeline'])}">
  <div class="note">讲解重点：为什么不是直接把 96 个时刻都拿去预测？因为项目目标是预测 2015-01-11 至 2015-01-17 的日最高、日最低、日平均负荷，日粒度聚合能和目标完全对应。</div>
</section>

<section>
  <h2>3. 代码截图：读取、重构、聚合、合并</h2>
  <img src="{image_uri(assets['code_load_daily'])}">
  <h3>推导解释</h3>
  <p>原始负荷宽表的每一行是一日，列 <code>T0000</code> 到 <code>T2345</code> 表示一天内 96 个采样点。使用 <code>melt</code> 将宽表变成长表后，每条记录对应“某一天某一时刻的负荷”；再按 <code>YMD</code> 分组，分别取最大值、最小值和平均值，得到三个预测目标。</p>
  <pre>{html.escape(code_sources['load_daily_data'])}</pre>
</section>

<section>
  <h2>4. 缺失值检查与业务判断</h2>
  <p>2015-01-11 至 2015-01-17 的负荷缺失不是脏数据，而是题目要求预测的未来目标；天气数据在这一段完整，可以作为预测输入。</p>
  <img src="{image_uri(assets['missingness'])}">
  <div class="note">结论：不删除目标期 7 天，也不使用插值填补目标负荷；后续预测要用递推方式生成这 7 天的负荷。</div>
</section>

<section>
  <h2>5. 2014 年负荷分析</h2>
  <p>按照实训要求，对 2014 年的日负荷进行分析。日最高、日平均、日最低三条曲线可以展示季节性和短期波动；负荷持续曲线可以展示全年高负荷和低负荷分布。</p>
  <img src="{image_uri(assets['load_2014'])}">
  <br><br>
  <img src="{image_uri(assets['duration_2014'])}">
</section>

<section>
  <h2>6. 天气合并与衍生变量</h2>
  <p>天气表按 <code>YMD</code> 与日负荷表合并。进一步构造 <code>temp_range</code>、<code>hdd</code>、<code>cdd</code>，让模型能表达“低温供热”和“高温制冷”对负荷的影响。</p>
  <img src="{image_uri(assets['merged_sample'])}">
</section>

<section>
  <h2>7. 代码截图：日历、周期和天气特征</h2>
  <img src="{image_uri(assets['code_add_features'])}">
  <h3>推导解释</h3>
  <p>月份、星期和年内日序号都是周期变量。如果直接把 12 月和 1 月当普通数字，模型会误以为二者距离很远；所以使用 <code>sin/cos</code> 编码，把周期映射到圆上，保留首尾相接的季节规律。</p>
  <pre>{html.escape(code_sources['add_features'])}</pre>
</section>

<section>
  <h2>8. 代码截图：滞后与滚动特征</h2>
  <img src="{image_uri(assets['feature_blocks'])}">
  <br><br>
  <img src="{image_uri(assets['code_target_features'])}">
  <h3>推导解释</h3>
  <p>电力负荷有明显惯性：昨天、上周同日、两周前同日通常是强信号。滚动均值和标准差用于描述近期水平和波动。所有滚动统计都先 <code>shift(1)</code>，保证今天的预测不会偷看今天真实负荷，避免数据泄漏。</p>
  <pre>{html.escape(code_sources['make_target_features'])}</pre>
</section>

<section>
  <h2>9. 第一次汇报结论</h2>
  <p>经过数据处理，原始 Excel 已转换为可建模的日粒度特征表：目标变量是日最高、日最低、日平均负荷；输入变量包含天气、日历周期、天气衍生变量、历史滞后和滚动统计。下一次汇报可以在这个数据集基础上进入算法模型比较、训练过程和最终模型选择。</p>
</section>
</main>
</body>
</html>
"""
    (OUT / "report1_data_processing.html").write_text(html_doc, encoding="utf-8")


def make_markdown(profile: pd.DataFrame, code_sources: dict[str, str]) -> None:
    md = f"""# 第一次汇报：数据处理与特征构建

本次汇报只覆盖项目二的数据处理部分，不展开模型训练。

## 1. 原始数据

- `Area_Load`：每天一行，96 个 15 分钟负荷采样点。
- `Area_Weather`：每天一行，包含最高温、最低温、平均温、湿度、降雨量。
- 预测目标期：`2015-01-11` 至 `2015-01-17`。

{markdown_table(profile)}

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
{code_sources['load_daily_data']}
```

![缺失值检查](assets/04_missingness_check.png)

## 4. 核心代码：基础特征工程

![日历周期与天气衍生特征代码截图](assets/10_code_add_features.png)

```python
{code_sources['add_features']}
```

![2014 年日负荷曲线](assets/05_2014_daily_load.png)

## 5. 核心代码：滞后与滚动特征

![特征工程结构](assets/08_feature_engineering_blocks.png)

![滞后与滚动特征代码截图](assets/11_code_make_target_features.png)

```python
{code_sources['make_target_features']}
```

## 6. 汇报结论

处理后的数据表已经把原始 15 分钟负荷转换成项目要求的日最高、日最低、日平均三个预测目标，并补充了天气、日历周期和历史负荷特征。第二次汇报即可进入算法模型训练与对比。
"""
    (OUT / "report1_data_processing.md").write_text(md, encoding="utf-8")


def make_presentation_guide() -> None:
    guide = """# 第一次汇报演示方法

## 推荐打开方式

1. 直接双击打开 `report1_data_processing.html`，或在浏览器里打开这个文件。
2. 汇报时按网页从上到下讲，老师问代码时切到对应“代码截图”小节。
3. 如果需要看原始代码，打开项目根目录的 `project2_forecast.py`，重点讲三个函数：`load_daily_data`、`add_features`、`make_target_features`。

## 5 分钟讲法

1. 先说明三次汇报安排：第一次只讲数据处理，算法留到第二次。
2. 说明原始 Excel 的两个表：负荷表每天 96 个点，天气表每天一条记录。
3. 讲数据处理主线：宽表转长表，按天聚合，合并天气，构造特征。
4. 重点解释缺失值：2015-01-11 至 2015-01-17 的 672 个负荷缺失是预测目标，不是坏数据。
5. 展示 2014 年负荷曲线，说明数据存在季节性和波动。
6. 讲特征工程：天气、日历周期、历史滞后、滚动统计。
7. 最后强调 `shift(1)` 防止数据泄漏，第二次汇报会进入模型训练。

## 老师可能会问的问题

- 为什么要把 96 个采样点聚合成日最大/最小/平均？
  因为项目要求预测的目标就是未来 7 天的日最高、日最低、日平均负荷。

- 目标期负荷缺失为什么不插值？
  因为这些缺失就是要预测的未来结果，插值会把未知答案伪造成已知数据。

- 为什么要做 sin/cos 周期编码？
  因为月份、星期是周期变量，12 月和 1 月、周日和周一在时间上相邻，普通数字编码会破坏这种关系。

- 为什么滞后和滚动特征要用 `shift(1)`？
  为了保证预测当天时只能使用当天以前的信息，避免数据泄漏。
"""
    (OUT / "report1_presentation_guide.md").write_text(guide, encoding="utf-8")


def make_speaker_notes(profile: pd.DataFrame) -> None:
    notes = f"""# 第一次汇报讲稿：数据处理部分

## 开场

老师好，我们组负责项目二：短期电力负荷预测。按照三次汇报安排，第一次我只讲数据处理部分，后面的算法模型和整体网页报告会放到第二、第三次汇报。

## 数据来源

原始 Excel 有两个工作表。`Area_Load` 是负荷数据，每天一行，每天有 96 个 15 分钟采样点；`Area_Weather` 是天气数据，每天一行，包含温度、湿度和降雨量。

关键数据检查结果：

{markdown_table(profile)}

## 为什么要重构数据

项目最终要求预测 2015 年 1 月 11 日到 1 月 17 日每天的最大负荷、最小负荷和平均负荷。因此不能只停留在原始 15 分钟宽表，而要把每天 96 个点聚合成三个日粒度目标。

代码里用 `melt` 把宽表转成长表，之后按 `YMD` 分组计算最大值、最小值和均值。这一步完成后，预测目标和题目要求完全对齐。

## 缺失值判断

目标期 7 天一共有 672 个负荷缺失单元格，也就是 7 天乘以每天 96 个采样点。这不是普通缺失值，而是题目让我们预测的未来负荷。天气数据在目标期完整，所以后续可以作为预测输入。

## 2014 年负荷分析

我对 2014 年的日最高、日平均和日最低负荷画了曲线，可以看到负荷存在季节变化和短期波动。负荷持续曲线则从高到低排序，帮助观察全年高负荷和低负荷的分布。

## 特征工程

特征工程分成三类：

1. 天气特征：最高温、最低温、平均温、湿度、降雨量。
2. 日历周期特征：星期、月份、是否周末，以及 sin/cos 周期编码。
3. 历史负荷特征：昨天、上周同日、两周前同日，以及 7 天和 14 天滚动均值、滚动标准差。

这里特别注意防止数据泄漏。滚动特征先使用 `shift(1)`，表示预测今天时只使用昨天及以前的信息，不会偷看当天真实负荷。

## 结尾

第一次汇报的结论是：我们已经把原始 Excel 处理成了可用于建模的日粒度特征表。下一次汇报会基于这张表展示算法模型选择、训练过程、调参和模型对比。
"""
    (OUT / "report1_speaker_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    set_chinese_font()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find data file: {DATA_PATH}")

    load_raw = pd.read_excel(DATA_PATH, sheet_name="Area_Load")
    weather_raw = pd.read_excel(DATA_PATH, sheet_name="Area_Weather")
    weather_named = weather_raw.copy()
    weather_named.columns = ["YMD", "temp_max", "temp_min", "temp_avg", "humidity", "rainfall"]
    daily = p2.load_daily_data()

    profile = profile_data(load_raw, weather_raw, daily)
    profile.to_csv(OUT / "report1_data_profile.csv", index=False, encoding="utf-8-sig")
    daily.head(20).to_csv(OUT / "report1_daily_processed_sample.csv", index=False, encoding="utf-8-sig")

    time_cols = [c for c in load_raw.columns if c != "YMD"]
    raw_load_view = load_raw[["YMD", *time_cols[:10], *time_cols[-2:]]]
    daily_view = daily[
        [
            "YMD",
            "load_max",
            "load_min",
            "load_mean",
            "temp_max",
            "temp_min",
            "temp_avg",
            "humidity",
            "rainfall",
            "temp_range",
            "hdd",
            "cdd",
        ]
    ].head(10)

    assets = {
        "raw_load": ASSETS / "01_raw_load_sample.png",
        "raw_weather": ASSETS / "02_raw_weather_sample.png",
        "pipeline": ASSETS / "03_data_pipeline.png",
        "missingness": ASSETS / "04_missingness_check.png",
        "load_2014": ASSETS / "05_2014_daily_load.png",
        "duration_2014": ASSETS / "06_2014_load_duration.png",
        "merged_sample": ASSETS / "07_merged_daily_feature_sample.png",
        "feature_blocks": ASSETS / "08_feature_engineering_blocks.png",
        "code_load_daily": ASSETS / "09_code_load_daily_data.png",
        "code_add_features": ASSETS / "10_code_add_features.png",
        "code_target_features": ASSETS / "11_code_make_target_features.png",
    }

    save_df_image(raw_load_view, assets["raw_load"], "Area_Load 原始负荷表节选")
    save_df_image(weather_named, assets["raw_weather"], "Area_Weather 原始天气表节选")
    plot_pipeline(assets["pipeline"])
    plot_missingness(load_raw, weather_raw, assets["missingness"])
    plot_2014_daily_load(daily, assets["load_2014"])
    plot_load_duration(daily, assets["duration_2014"])
    save_df_image(daily_view, assets["merged_sample"], "日粒度负荷 + 天气 + 衍生变量节选", max_rows=10)
    plot_feature_blocks(assets["feature_blocks"])

    code_sources = {
        "load_daily_data": save_code_image(p2.load_daily_data, assets["code_load_daily"], "核心代码 1：读取、重构、聚合、合并"),
        "add_features": save_code_image(p2.add_features, assets["code_add_features"], "核心代码 2：日历周期与天气衍生特征"),
        "make_target_features": save_code_image(
            p2.make_target_features,
            assets["code_target_features"],
            "核心代码 3：滞后与滚动特征，避免数据泄漏",
        ),
    }

    make_html(profile, assets, code_sources)
    make_markdown(profile, code_sources)
    make_presentation_guide()
    make_speaker_notes(profile)

    print(f"Generated first report package: {OUT}")
    print(f"HTML: {OUT / 'report1_data_processing.html'}")
    print(f"Speaker notes: {OUT / 'report1_speaker_notes.md'}")


if __name__ == "__main__":
    main()
