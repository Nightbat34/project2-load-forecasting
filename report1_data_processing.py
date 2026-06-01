"""
第一次汇报材料生成脚本：只负责“数据处理”部分。

汇报时可以这样对照：
1. HTML 页面负责展示：图表、表格、代码截图、中文解释。
2. 本脚本负责生成这些材料：读取 Excel、检查缺失、检测异常值、画图、生成 HTML/PDF。
3. 真正的建模训练代码在 project2_forecast.py；本脚本只调用其中的数据处理函数，不重新训练模型。
"""

from __future__ import annotations

import base64
import html
import inspect
import os
import textwrap
from pathlib import Path

import matplotlib

# 使用 Agg 后端：在不打开图形窗口的情况下保存 PNG，适合自动生成汇报材料。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import project2_forecast as p2


ROOT = Path(__file__).resolve().parent
# 默认数据位置：process 的上一级 Data 文件夹。也可以用 LOAD_DATA_PATH 环境变量临时指定数据文件。
DATA_PATH = Path(os.environ.get("LOAD_DATA_PATH", ROOT.parent / "Data" / "附件1-电网负荷数据.xlsx"))
# 第一次汇报所有输出统一放在这个目录，方便审核和删除旧版。
OUT = ROOT / "output" / "report1_data_processing"
ASSETS = OUT / "assets"


def ensure_output_dirs() -> None:
    """确保输出目录存在；后续所有图片、HTML、CSV 都写入这里。"""
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)


def set_chinese_font() -> None:
    """设置中文字体，避免图表导出后中文显示成方块。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_df_image(df: pd.DataFrame, path: Path, title: str, max_rows: int = 8) -> None:
    """把 DataFrame 渲染成图片，HTML 里用它模拟“数据截图”。"""
    # 只展示前几行，避免表格太长影响汇报阅读。
    shown = df.head(max_rows).copy()
    shown = shown.fillna("")
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: "" if x == "" else f"{x:.2f}")

    # 根据行数动态调整图片高度，行多时自动变高，防止文字挤在一起。
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
    # 表头深色、正文隔行浅色，让截图更接近正式报告表格。
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
    """把核心函数源码渲染成代码截图，同时返回源码供 HTML 的 <pre> 展示。"""
    source = inspect.getsource(func)
    source = textwrap.dedent(source).rstrip()
    wrapped_lines: list[str] = []
    # 代码截图宽度有限，长行先软换行，避免图片右侧被截断。
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
    """把本地 PNG 转成 base64 内嵌到 HTML，单独发 HTML 时图片也不会丢。"""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def table_html(df: pd.DataFrame, max_rows: int = 12, float_fmt: str = "{:.2f}") -> str:
    """把 DataFrame 转为带 CSS class 的 HTML 表格。"""
    shown = df.head(max_rows).copy()
    formatters = {
        col: (lambda x, fmt=float_fmt: "" if pd.isna(x) else fmt.format(x))
        for col in shown.columns
        if pd.api.types.is_float_dtype(shown[col])
    }
    return shown.to_html(index=False, classes="data-table", escape=False, formatters=formatters)


def markdown_table(df: pd.DataFrame) -> str:
    """不依赖 tabulate 的 Markdown 表格生成器，用于讲稿和 md 备份。"""
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


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    """写 CSV；如果旧文件被浏览器/Excel 占用，就写到 *_new.csv，保证 HTML 生成不中断。"""
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_new{path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"CSV file is locked, wrote fallback: {fallback}")


def profile_data(load_raw: pd.DataFrame, weather_raw: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """生成“原始数据理解”检查表，对应 HTML 第 1 节。"""
    time_cols = [c for c in load_raw.columns if c != "YMD"]
    # 目标期是题目要求预测的 2015-01-11 到 2015-01-17。
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
    """画数据处理流程图，对应 HTML 第 2 节。"""
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
        # 每个矩形代表一个处理阶段，箭头表示从原始数据到建模特征的流向。
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
    """画缺失值检查图，对应 HTML 第 4 节。"""
    time_cols = [c for c in load_raw.columns if c != "YMD"]
    # 每天有 96 个负荷点，按行统计每天缺失了多少个负荷采样值。
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
    # 只截取目标期前后，突出 2015-01-11 到 2015-01-17 的缺失是预测任务本身。
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
    """画 2014 年日最高/日平均/日最低负荷曲线，对应 HTML 第 7 节。"""
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
    """画负荷持续曲线：把 2014 年负荷从高到低排序，看全年负荷分布。"""
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
    """画特征工程模块图，对应 HTML 第 10 节。"""
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


FEATURE_CN = {
    # 统一维护特征名的中文解释，后面异常值表、特征权重表都会用到。
    "load_max": "日最高负荷",
    "load_min": "日最低负荷",
    "load_mean": "日平均负荷",
    "temp_max": "最高温度",
    "temp_min": "最低温度",
    "temp_avg": "平均温度",
    "humidity": "相对湿度",
    "rainfall": "降雨量",
    "temp_range": "日温差",
    "hdd": "供热度日",
    "cdd": "制冷度日",
    "month_sin": "月份周期正弦",
    "month_cos": "月份周期余弦",
    "dow_sin": "星期周期正弦",
    "dow_cos": "星期周期余弦",
    "doy_sin": "年内日周期正弦",
    "doy_cos": "年内日周期余弦",
    "is_weekend": "是否周末",
}


def feature_label(feature: str) -> str:
    """把英文特征名翻译成汇报用中文含义。"""
    if feature in FEATURE_CN:
        return FEATURE_CN[feature]
    for target, target_cn in p2.TARGETS.items():
        # 滞后项：例如 load_mean_lag_7 表示“日平均负荷前 7 天的值”。
        if feature.startswith(f"{target}_lag_"):
            lag = feature.rsplit("_", 1)[-1]
            return f"{target_cn}前 {lag} 天滞后值"
        # 滚动均值：例如过去 7 天平均负荷，表达近期负荷水平。
        if feature.startswith(f"{target}_roll_mean_"):
            win = feature.rsplit("_", 1)[-1]
            return f"{target_cn}过去 {win} 天滚动均值"
        # 滚动标准差：表达近期波动程度。
        if feature.startswith(f"{target}_roll_std_"):
            win = feature.rsplit("_", 1)[-1]
            return f"{target_cn}过去 {win} 天滚动标准差"
    return feature


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """使用 IQR 四分位距规则识别“极端候选值”，对应 HTML 第 6 节。

    注意：IQR 标出来的是统计意义上的极端值，不等于错误数据。
    对电力负荷预测来说，极端天气和极端负荷反而很有研究价值，
    所以这里的策略是“识别并标记”，不是直接删除或截尾。
    """
    # 只用已知历史期计算异常边界，不能把 2015-01-11 之后的预测目标期混进去。
    known = df[df["date"] <= pd.Timestamp("2015-01-10")].copy()
    rows: list[dict[str, object]] = []
    for col in columns:
        values = known[col].dropna()
        # Q1/Q3 是 25% 和 75% 分位数；IQR = Q3 - Q1。
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        # 经典箱线图异常值边界：低于 Q1-1.5IQR 或高于 Q3+1.5IQR。
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flags = (values < lower) | (values > upper)
        rows.append(
            {
                "字段": col,
                "中文含义": feature_label(col),
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "下界": lower,
                "上界": upper,
                "极端候选数量": int(flags.sum()),
                "极端占比(%)": round(float(flags.mean() * 100), 2),
                "业务解释": "可能是极端天气、节假日或生产活动变化导致",
                "处理策略": "保留原始值，增加极端事件标记，不作为错误删除",
            }
        )
    return pd.DataFrame(rows)


def add_extreme_event_flags(df: pd.DataFrame, outlier_summary: pd.DataFrame) -> pd.DataFrame:
    """根据 IQR 边界增加极端事件标记，但不改变原始数值。"""
    cleaned = df.copy()
    for _, row in outlier_summary.iterrows():
        col = str(row["字段"])
        lower = float(row["下界"])
        upper = float(row["上界"])
        # 标记字段用于解释和后续建模扩展；原始负荷/天气值全部保留。
        cleaned[f"{col}_extreme_flag"] = ((cleaned[col] < lower) | (cleaned[col] > upper)).astype(int)
    return cleaned


def physical_quality_checks(load_raw: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """检查真正需要修正的物理不可能值，例如负负荷、湿度越界。"""
    time_cols = [c for c in load_raw.columns if c != "YMD"]
    known_load = load_raw[load_raw["YMD"] <= 20150110][time_cols]
    rows = [
        {
            "检查对象": "15 分钟负荷",
            "规则": "负荷应大于 0",
            "问题数量": int((known_load <= 0).sum().sum()),
            "处理结论": "若出现才视为错误；当前不需要修正",
        },
        {
            "检查对象": "湿度",
            "规则": "0 <= humidity <= 100",
            "问题数量": int(((daily["humidity"] < 0) | (daily["humidity"] > 100)).sum()),
            "处理结论": "越界才视为错误；当前不需要修正",
        },
        {
            "检查对象": "降雨量",
            "规则": "rainfall >= 0",
            "问题数量": int((daily["rainfall"] < 0).sum()),
            "处理结论": "负降雨才视为错误；当前不需要修正",
        },
        {
            "检查对象": "温度关系",
            "规则": "temp_min <= temp_avg <= temp_max",
            "问题数量": int(((daily["temp_min"] > daily["temp_avg"]) | (daily["temp_avg"] > daily["temp_max"])).sum()),
            "处理结论": "关系不成立才视为错误；当前不需要修正",
        },
    ]
    return pd.DataFrame(rows)


def compute_feature_weights(daily: pd.DataFrame) -> pd.DataFrame:
    """计算数据处理阶段的特征参考权重，对应 HTML 第 11 节。"""
    rows: list[dict[str, object]] = []
    for target, target_cn in p2.TARGETS.items():
        # 复用主项目里的 make_target_features，保证汇报展示的特征和真实建模特征一致。
        feature_df, features = p2.make_target_features(daily, target)
        feature_df = feature_df[feature_df["date"] <= pd.Timestamp("2015-01-10")]
        # 滞后/滚动特征前几天会产生 NaN，计算相关性前要去掉。
        feature_df = feature_df.dropna(subset=[target, *features])
        scores: list[tuple[str, float, float]] = []
        for feature in features:
            # 这里用“单变量 Pearson 相关系数”做解释性排序，不代表最终模型参数。
            corr = feature_df[[target, feature]].corr().iloc[0, 1]
            if pd.isna(corr) or np.isinf(corr):
                corr = 0.0
            # 权重看强弱，所以取绝对值；正负方向单独保存在“相关方向”里。
            scores.append((feature, float(corr), abs(float(corr))))
        # 把每个目标变量下所有特征的相关性绝对值归一化成百分比。
        total = sum(score for _, _, score in scores) or 1.0
        for feature, corr, score in scores:
            rows.append(
                {
                    "目标变量": target_cn,
                    "特征名": feature,
                    "中文含义": feature_label(feature),
                    "相关方向": "正相关" if corr >= 0 else "负相关",
                    "相关系数": round(corr, 4),
                    "参考权重(%)": round(score / total * 100, 2),
                }
            )
    return pd.DataFrame(rows).sort_values(["目标变量", "参考权重(%)"], ascending=[True, False])


def build_feature_dictionary(daily: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """生成“每个特征的示例值”表，方便汇报时解释特征到底是什么。"""
    rows: list[dict[str, object]] = []
    for target, target_cn in p2.TARGETS.items():
        feature_df, features = p2.make_target_features(daily, target)
        feature_df = feature_df[feature_df["date"] <= pd.Timestamp("2015-01-10")]
        # 取历史期最后一条完整样本作为演示样本，日期接近预测起点，最容易讲清楚。
        sample = feature_df.dropna(subset=features).tail(1).iloc[0]
        weight_part = weights[weights["目标变量"] == target_cn].set_index("特征名")
        for feature in features:
            rows.append(
                {
                    "目标变量": target_cn,
                    "特征名": feature,
                    "中文含义": feature_label(feature),
                    "示例日期": sample["date"].strftime("%Y-%m-%d"),
                    "示例值": round(float(sample[feature]), 4),
                    "参考权重(%)": float(weight_part.loc[feature, "参考权重(%)"]),
                    "相关方向": str(weight_part.loc[feature, "相关方向"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["目标变量", "参考权重(%)"], ascending=[True, False])


def plot_raw_15min_distribution(load_raw: pd.DataFrame, path: Path) -> None:
    """画原始 15 分钟负荷分布，对应 HTML 第 5 节第一张图。"""
    time_cols = [c for c in load_raw.columns if c != "YMD"]
    # 把所有历史日期、所有 96 个时刻的负荷拉平成一个数组，看总体分布。
    known = load_raw[load_raw["YMD"] <= 20150110][time_cols].to_numpy().ravel()
    known = known[~pd.isna(known)]
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.hist(known, bins=60, color="#1F6F8B", alpha=0.84, edgecolor="white")
    ax.set_title("原始 15 分钟负荷分布：先观察总体形态，再做日粒度聚合", loc="left", fontsize=17, fontweight="bold")
    ax.set_xlabel("15 分钟负荷值")
    ax.set_ylabel("频数")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_daily_distribution(daily: pd.DataFrame, path: Path) -> None:
    """画日最高/日平均/日最低负荷分布，对应 HTML 第 5 节第二张图。"""
    known = daily[daily["date"] <= pd.Timestamp("2015-01-10")]
    cols = ["load_max", "load_mean", "load_min"]
    titles = ["日最高负荷", "日平均负荷", "日最低负荷"]
    colors = ["#BA3E3E", "#1F6F8B", "#3A8B6D"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, col, title, color in zip(axes, cols, titles, colors):
        ax.hist(known[col].dropna(), bins=36, color=color, alpha=0.82, edgecolor="white")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("日粒度负荷分布：三个预测目标的取值范围和偏态", x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_weather_distribution(daily: pd.DataFrame, path: Path) -> None:
    """画主要天气变量分布，对应 HTML 第 5 节第三张图。"""
    known = daily[daily["date"] <= pd.Timestamp("2015-01-10")]
    cols = ["temp_avg", "humidity", "rainfall", "temp_range"]
    titles = ["平均温度", "湿度", "降雨量", "日温差"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, col, title in zip(axes.ravel(), cols, titles):
        ax.hist(known[col].dropna(), bins=36, color="#6E5FA8", alpha=0.82, edgecolor="white")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("天气变量分布：检查气象输入的范围和偏态", x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_weather_load_reasoning(daily: pd.DataFrame, path: Path) -> None:
    """画天气与负荷关系图，说明为什么想到合并天气并构造衍生变量。"""
    known = daily[(daily["date"] >= "2012-01-01") & (daily["date"] <= "2014-12-31")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].scatter(known["temp_avg"], known["load_mean"], s=12, alpha=0.45, color="#1F6F8B")
    axes[0].set_title("平均温度 vs 日平均负荷", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("平均温度")
    axes[0].set_ylabel("日平均负荷")

    axes[1].scatter(known["cdd"], known["load_max"], s=12, alpha=0.45, color="#BA3E3E")
    axes[1].set_title("CDD 制冷需求 vs 日最高负荷", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("CDD = max(temp_avg - 26, 0)")
    axes[1].set_ylabel("日最高负荷")

    axes[2].scatter(known["hdd"], known["load_min"], s=12, alpha=0.45, color="#3A8B6D")
    axes[2].set_title("HDD 供热需求 vs 日最低负荷", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("HDD = max(18 - temp_avg, 0)")
    axes[2].set_ylabel("日最低负荷")

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("从数据中验证天气联想：温度变化会带来制冷/供热负荷变化", x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_outlier_box_compare(raw: pd.DataFrame, flagged: pd.DataFrame, path: Path) -> None:
    """画箱线图：说明 IQR 找到的是极端事件候选，而不是错误数据。"""
    raw_known = raw[raw["date"] <= pd.Timestamp("2015-01-10")]
    # 负荷和天气量纲差异很大，分成两行展示，避免天气变量被负荷数值压扁。
    groups = [
        (["load_max", "load_mean", "load_min"], ["日最高", "日平均", "日最低"], "负荷变量"),
        (["temp_avg", "humidity", "rainfall"], ["平均温度", "湿度", "降雨"], "天气变量"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row, (cols, labels, title) in enumerate(groups):
        axes[row, 0].boxplot([raw_known[c].dropna() for c in cols], tick_labels=labels, patch_artist=True)
        axes[row, 0].set_title(f"{title}：IQR 识别出的极端候选", loc="left", fontsize=14, fontweight="bold")
        axes[row, 1].boxplot([raw_known[c].dropna() for c in cols], tick_labels=labels, patch_artist=True)
        # 第二张图不做数值改动，而是强调“极端值保留、仅打标记”。
        axes[row, 1].set_title(f"{title}：保留原始值并增加极端标记", loc="left", fontsize=14, fontweight="bold")
    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("数值")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_before_after_distribution(raw: pd.DataFrame, cleaned: pd.DataFrame, path: Path) -> None:
    """画负荷分布与 IQR 边界图，说明极端值保留而非硬删除。"""
    raw_known = raw[raw["date"] <= pd.Timestamp("2015-01-10")]
    cols = ["load_max", "load_mean", "load_min"]
    titles = ["日最高负荷", "日平均负荷", "日最低负荷"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, col, title in zip(axes, cols, titles):
        values = raw_known[col].dropna()
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        ax.hist(values, bins=34, alpha=0.62, label="原始分布", color="#1F6F8B")
        ax.axvline(lower, color="#BA3E3E", linestyle="--", lw=1.8, label="IQR 边界" if col == cols[0] else None)
        ax.axvline(upper, color="#BA3E3E", linestyle="--", lw=1.8)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("负荷分布与 IQR 边界：极端值保留，边界仅用于标记", x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_outlier_timeline(raw: pd.DataFrame, outlier_summary: pd.DataFrame, path: Path) -> None:
    """把 2014 年的极端事件标在时间线上，方便解释极端天气价值。"""
    known = raw[raw["date"].dt.year == 2014].copy()
    fig, ax = plt.subplots(figsize=(14, 5.6))
    colors = {"load_max": "#BA3E3E", "load_mean": "#1F6F8B", "load_min": "#3A8B6D"}
    for col, color in colors.items():
        ax.plot(known["date"], known[col], lw=1.5, color=color, label=feature_label(col) if col in FEATURE_CN else p2.TARGETS[col])
        bound = outlier_summary[outlier_summary["字段"] == col].iloc[0]
        # 根据 IQR 上下界找出 2014 年落在边界外的日期，用黑点标出。
        flags = (known[col] < float(bound["下界"])) | (known[col] > float(bound["上界"]))
        ax.scatter(known.loc[flags, "date"], known.loc[flags, col], color="#111827", s=26, zorder=5)
    ax.set_title("2014 年极端事件位置：黑点为 IQR 规则识别出的候选日期", loc="left", fontsize=17, fontweight="bold")
    ax.set_ylabel("负荷")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, loc="upper left")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_weights(weights: pd.DataFrame, path: Path) -> None:
    """画特征参考权重 Top 图，对应 HTML 第 11 节。"""
    # 只取前 22 个权重最高的特征，避免图太长影响阅读。
    top = weights.sort_values("参考权重(%)", ascending=False).head(22).copy()
    top["标签"] = top["目标变量"] + " | " + top["中文含义"]
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(top["标签"][::-1], top["参考权重(%)"][::-1], color="#1F6F8B")
    ax.set_title("特征参考权重 Top 22：基于单变量相关性的解释性排序", loc="left", fontsize=17, fontweight="bold")
    ax.set_xlabel("参考权重(%)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_source_map(path: Path) -> None:
    """画特征来源图，解释特征名从哪里来、和现实含义如何对应。"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    groups = [
        ("天气原始特征", "Excel Area_Weather\n最高/最低/平均温度\n湿度、降雨", "#1F6F8B"),
        ("天气衍生特征", "temp_range\nhdd\ncdd", "#3A8B6D"),
        ("日历周期特征", "month_sin/cos\ndow_sin/cos\ndoy_sin/cos\nis_weekend", "#B07D2B"),
        ("历史负荷特征", "lag_1/7/14\nroll_mean_7/14\nroll_std_7/14", "#6E5FA8"),
    ]
    x0, y, w, h, gap = 0.05, 0.40, 0.19, 0.34, 0.045
    for i, (title, detail, color) in enumerate(groups):
        x = x0 + i * (w + gap)
        ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.95))
        ax.text(x + w / 2, y + h * 0.68, title, color="white", fontsize=14, fontweight="bold", ha="center")
        ax.text(x + w / 2, y + h * 0.35, detail, color="white", fontsize=10.5, ha="center", va="center")
        if i < len(groups) - 1:
            ax.annotate(
                "",
                xy=(x + w + gap * 0.75, y + h / 2),
                xytext=(x + w + gap * 0.18, y + h / 2),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=2.3, color="#334155"),
            )
    ax.text(0.05, 0.88, "特征名来源：从原始数据、业务衍生、日历周期和历史负荷自动组合", fontsize=18, fontweight="bold")
    ax.text(
        0.05,
        0.18,
        "代码中最终特征列表 = MODEL_WEATHER + CALENDAR_FEATURES + lag_features；特征演示表从这份列表自动生成，不是手工编出来的。",
        fontsize=12.5,
        color="#334155",
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_html(
    profile: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    quality_checks: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    assets: dict[str, Path],
    code_sources: dict[str, str],
) -> None:
    """生成最终 HTML 汇报文件。

    这个函数把前面生成的表格、图片和代码截图串成网页。
    汇报时看到的 HTML 章节编号，基本都能在这里找到对应位置。
    """
    # CSS 只负责页面样式，不参与任何数据计算。
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
    # HTML 里直接内嵌 base64 图片，因此打开单个 html 文件也能看到所有图。
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
  <p class="lead">本次只汇报数据处理部分：原始数据理解、清洗重构、缺失值判断、分布分析、异常值检测与处理、2014 年负荷分析、天气合并、特征值演示和参考权重。算法训练和模型选择放到第二次汇报。</p>
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
  <p>处理流程从原始 Excel 出发，先重构负荷表，再聚合为题目要求预测的日最高、日最低、日平均负荷，并与天气表合并。随后不是简单删掉极端值，而是先判断它们是不是“统计极端候选”，再区分物理错误和极端天气样本。</p>
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
  <h2>5. 数据分布分析</h2>
  <p>先看原始 15 分钟负荷的总体分布，再看聚合后的日最高、日平均、日最低负荷分布，最后检查天气变量分布。分布图的作用是判断变量是否偏态、是否存在长尾，以及后续异常值处理是否必要。</p>
  <img src="{image_uri(assets['raw_15min_dist'])}">
  <br><br>
  <img src="{image_uri(assets['daily_distribution'])}">
  <br><br>
  <img src="{image_uri(assets['weather_distribution'])}">
</section>

<section>
  <h2>6. 极端值识别与质量检查</h2>
  <p>IQR 四分位距规则适合发现“统计极端候选值”：下界 = Q1 - 1.5 × IQR，上界 = Q3 + 1.5 × IQR。但在电力负荷预测中，极端天气和极端负荷不一定是错误，反而往往最有研究价值。因此本项目不把 IQR 候选值直接删除或截尾，而是保留原始值，并增加极端事件标记。</p>
  <h3>物理不可能值检查</h3>
  <p>真正需要修正的是物理不可能或逻辑不成立的数据，例如负负荷、湿度超过 100、负降雨、最低温高于平均温等。</p>
  {table_html(quality_checks, max_rows=10)}
  <h3>IQR 极端候选值统计</h3>
  {table_html(outlier_summary, max_rows=20)}
  <img src="{image_uri(assets['box_before_after'])}">
  <br><br>
  <img src="{image_uri(assets['distribution_before_after'])}">
  <br><br>
  <img src="{image_uri(assets['outlier_timeline'])}">
  <h3>代码截图：极端值识别与质量检查</h3>
  <img src="{image_uri(assets['code_outliers'])}">
  <pre>{html.escape(code_sources['outliers'])}</pre>
  <div class="note">汇报口径：IQR 只是提醒我们“这里值得解释”，不是判定错误的最终标准。极端天气对应的负荷变化是预测模型必须学习的场景，所以应保留并标记。</div>
</section>

<section>
  <h2>7. 2014 年负荷分析</h2>
  <p>按照实训要求，对 2014 年的日负荷进行分析。日最高、日平均、日最低三条曲线可以展示季节性和短期波动；负荷持续曲线可以展示全年高负荷和低负荷分布。</p>
  <img src="{image_uri(assets['load_2014'])}">
  <br><br>
  <img src="{image_uri(assets['duration_2014'])}">
</section>

<section>
  <h2>8. 天气合并与衍生变量</h2>
  <p>这一部分不是凭空加特征，而是从业务机理和数据观察共同推出来的。电力负荷会受到温度影响：高温时空调制冷增加，低温时供热或取暖设备增加，湿度和降雨也会改变居民、商业和工业用电行为。因此原始天气表不能孤立放着，需要按日期 <code>YMD</code> 合并到日负荷表中。</p>
  <h3>怎么想到的：业务联想</h3>
  <p>负荷预测的目标是“某一天”的最高、最低、平均负荷，而天气表也是“某一天”的天气记录，两者天然可以用日期对齐。进一步看，单独的平均温度还不够表达用电机理，所以构造三个衍生变量：</p>
  <div class="note"><code>temp_range = temp_max - temp_min</code>：表示一天内温差，温差大时负荷波动可能更明显。<br><code>hdd = max(18 - temp_avg, 0)</code>：低于舒适温度时的供热需求强度。<br><code>cdd = max(temp_avg - 26, 0)</code>：高于高温阈值时的制冷需求强度。</div>
  <h3>怎么发现的：数据验证</h3>
  <p>先用散点图验证天气和负荷是否存在关系。图中可以看到，温度、CDD、HDD 与不同负荷目标存在可观察的关联，因此把天气变量加入特征表是有依据的。</p>
  <img src="{image_uri(assets['weather_load_reasoning'])}">
  <h3>怎么实现的：日期合并 + 衍生变量</h3>
  <p>实现上分两步：第一步在 <code>load_daily_data()</code> 中把负荷宽表聚合成日粒度，再按 <code>YMD</code> 和天气表 <code>merge</code>；第二步在 <code>add_features()</code> 中计算 <code>temp_range</code>、<code>hdd</code>、<code>cdd</code>。</p>
  <img src="{image_uri(assets['merged_sample'])}">
  <h3>代码体现 1：按 YMD 合并天气</h3>
  <img src="{image_uri(assets['code_load_daily'])}">
  <h3>代码体现 2：构造天气衍生变量</h3>
  <img src="{image_uri(assets['code_add_features'])}">
</section>

<section>
  <h2>9. 代码截图：日历、周期和天气特征</h2>
  <img src="{image_uri(assets['code_add_features'])}">
  <h3>推导解释</h3>
  <p>月份、星期和年内日序号都是周期变量。如果直接把 12 月和 1 月当普通数字，模型会误以为二者距离很远；所以使用 <code>sin/cos</code> 编码，把周期映射到圆上，保留首尾相接的季节规律。</p>
  <pre>{html.escape(code_sources['add_features'])}</pre>
</section>

<section>
  <h2>10. 代码截图：滞后与滚动特征</h2>
  <img src="{image_uri(assets['feature_blocks'])}">
  <br><br>
  <img src="{image_uri(assets['code_target_features'])}">
  <h3>推导解释</h3>
  <p>电力负荷有明显惯性：昨天、上周同日、两周前同日通常是强信号。滚动均值和标准差用于描述近期水平和波动。所有滚动统计都先 <code>shift(1)</code>，保证今天的预测不会偷看今天真实负荷，避免数据泄漏。</p>
  <pre>{html.escape(code_sources['make_target_features'])}</pre>
</section>

<section>
  <h2>11. 特征值演示与参考权重</h2>
  <p>下表展示每个建模特征的中文含义、示例日期、示例值和数据处理阶段的参考权重。这里的特征不是手工随便写出来的，而是由代码中的特征列表自动生成：<code>MODEL_WEATHER + CALENDAR_FEATURES + lag_features</code>。</p>
  <h3>特征名是哪来的</h3>
  <p>特征来源分为四类：第一类是天气表原始字段，如温度、湿度、降雨；第二类是基于业务机理构造的天气衍生变量，如 <code>temp_range</code>、<code>hdd</code>、<code>cdd</code>；第三类是日历周期变量，用来表达月份、星期和季节周期；第四类是历史负荷变量，用来表达昨天、上周同日、两周前同日以及近期滚动水平。</p>
  <img src="{image_uri(assets['feature_source_map'])}">
  <h3>数学含义怎么和现实结合</h3>
  <div class="note">数学上，<code>lag_7</code> 是前 7 天的目标值；现实上，它代表“上周同一天的用电习惯”。<br><code>roll_mean_7</code> 是过去 7 天均值；现实上，它代表近期用电水平。<br><code>roll_std_7</code> 是过去 7 天标准差；现实上，它代表近期负荷波动程度。<br><code>month_sin/cos</code>、<code>dow_sin/cos</code> 是周期编码；现实上，它们代表季节和星期规律。</div>
  <h3>和天气衍生变量的关系</h3>
  <p>第 8 节讲的是天气变量为什么要合并、为什么要衍生；第 11 节是在此基础上把所有可建模特征统一列出来。也就是说，<code>temp_range</code>、<code>hdd</code>、<code>cdd</code> 是特征体系中的“天气衍生特征”部分，后面再和日历特征、历史负荷特征一起进入建模数据表。</p>
  <h3>示例值怎么获取</h3>
  <p>示例值来自历史期最后一条完整样本，即 2015-01-10。选择这一天是因为它紧挨着预测窗口 2015-01-11，最适合说明“预测第一天时，模型能看到哪些已知信息”。</p>
  <h3>参考权重怎么计算</h3>
  <p>权重不是最终模型参数，而是用每个特征和预测目标之间的单变量 Pearson 相关系数计算：先取相关系数绝对值表示关系强弱，再在同一目标变量下归一化成百分比。因此它只用于数据处理阶段的解释性排序。</p>
  <img src="{image_uri(assets['feature_demo'])}">
  <br><br>
  <img src="{image_uri(assets['feature_weights'])}">
  {table_html(feature_dictionary, max_rows=80)}
  <h3>代码截图：特征参考权重计算</h3>
  <img src="{image_uri(assets['code_feature_weights'])}">
  <pre>{html.escape(code_sources['feature_weights'])}</pre>
</section>

<section>
  <h2>12. 第一次汇报结论</h2>
  <p>经过数据处理，原始 Excel 已转换为可建模的日粒度特征表：目标变量是日最高、日最低、日平均负荷；输入变量包含天气、日历周期、天气衍生变量、历史滞后和滚动统计。新版还补充了缺失值判断、分布检查、极端值识别、物理质量校验、极端事件保留策略，以及每个特征的示例值和参考权重。下一次汇报可以在这个数据集基础上进入算法模型比较、训练过程和最终模型选择。</p>
</section>
</main>
</body>
</html>
"""
    (OUT / "report1_data_processing.html").write_text(html_doc, encoding="utf-8")


def make_markdown(
    profile: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    quality_checks: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    code_sources: dict[str, str],
) -> None:
    """生成 Markdown 备份版，便于复制到报告书或二次编辑。"""
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

## 3. 天气合并与衍生变量的来源

天气特征来自业务联想和数据验证：高温会带来制冷负荷，低温会带来供热/取暖负荷，温差会影响一天内负荷波动。因此先按 `YMD` 将天气表合并到日负荷表，再构造 `temp_range`、`hdd`、`cdd`。

![天气与负荷关系验证](assets/12_weather_load_reasoning.png)

关键公式：

- `temp_range = temp_max - temp_min`
- `hdd = max(18 - temp_avg, 0)`
- `cdd = max(temp_avg - 26, 0)`

## 4. 核心代码：读取与日粒度聚合

![读取、重构、聚合、合并代码截图](assets/09_code_load_daily_data.png)

```python
{code_sources['load_daily_data']}
```

![缺失值检查](assets/04_missingness_check.png)

## 5. 数据分布、极端值识别与质量检查

![原始 15 分钟负荷分布](assets/04_raw_15min_distribution.png)

![日粒度负荷分布](assets/05_daily_load_distribution.png)

![天气变量分布](assets/06_weather_distribution.png)

IQR 规则用于识别统计极端候选值，但这些值不一定是错误。极端天气和极端负荷具有研究价值，因此本项目保留原始值并增加极端事件标记。

真正需要修正的是物理不可能值：

{markdown_table(quality_checks)}

{markdown_table(outlier_summary)}

![箱线图前后对比](assets/07_box_before_after.png)

![分布前后对比](assets/08_distribution_before_after.png)

```python
{code_sources['outliers']}
```

## 6. 核心代码：基础特征工程

![日历周期与天气衍生特征代码截图](assets/10_code_add_features.png)

```python
{code_sources['add_features']}
```

![2014 年日负荷曲线](assets/05_2014_daily_load.png)

## 7. 核心代码：滞后与滚动特征

![特征工程结构](assets/08_feature_engineering_blocks.png)

![滞后与滚动特征代码截图](assets/11_code_make_target_features.png)

```python
{code_sources['make_target_features']}
```

## 8. 特征值演示与参考权重

特征名由代码自动生成，不是手工编表：

- `MODEL_WEATHER`：天气原始特征 + 天气衍生变量
- `CALENDAR_FEATURES`：日历周期变量
- `lag_features`：滞后与滚动窗口变量

![特征来源图](assets/21_feature_source_map.png)

示例值来自 2015-01-10，这是预测窗口前最后一条完整历史样本。参考权重使用单变量 Pearson 相关系数的绝对值归一化得到，只表示数据处理阶段的解释性排序。

{markdown_table(feature_dictionary)}

![特征权重](assets/15_feature_weights.png)

```python
{code_sources['feature_weights']}
```

## 9. 汇报结论

处理后的数据表已经把原始 15 分钟负荷转换成项目要求的日最高、日最低、日平均三个预测目标，并补充了天气、日历周期、极端事件标记、历史负荷特征和特征参考权重。第二次汇报即可进入算法模型训练与对比。
"""
    (OUT / "report1_data_processing.md").write_text(md, encoding="utf-8")


def make_presentation_guide() -> None:
    """生成演示顺序和答辩问题提示。"""
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
5. 展示分布图和箱线图：先判断长尾和极端候选值，再解释 IQR 只是识别工具，不直接判定错误。
6. 展示 2014 年负荷曲线，说明数据存在季节性和波动。
7. 讲特征工程：天气、日历周期、历史滞后、滚动统计。
8. 讲天气合并为什么合理：业务上高温制冷、低温供热；数据上散点图能看到温度和负荷有关；代码上用 `YMD` 合并并构造 `temp_range/hdd/cdd`。
9. 展示特征值演示表和参考权重，强调这是数据处理阶段的解释性排序，不是最终模型参数。
10. 最后强调 `shift(1)` 防止数据泄漏，第二次汇报会进入模型训练。

## 老师可能会问的问题

- 为什么要把 96 个采样点聚合成日最大/最小/平均？
  因为项目要求预测的目标就是未来 7 天的日最高、日最低、日平均负荷。

- 目标期负荷缺失为什么不插值？
  因为这些缺失就是要预测的未来结果，插值会把未知答案伪造成已知数据。

- 为什么要做 sin/cos 周期编码？
  因为月份、星期是周期变量，12 月和 1 月、周日和周一在时间上相邻，普通数字编码会破坏这种关系。

- 为什么滞后和滚动特征要用 `shift(1)`？
  为了保证预测当天时只能使用当天以前的信息，避免数据泄漏。

- 为什么 IQR 标出的值不直接删除或截尾？
  因为 IQR 标出的是统计极端值，不等于错误。极端天气和极端负荷很可能是项目最需要研究的样本，所以保留原始值并增加极端事件标记。

- 特征权重是不是模型训练结果？
  不是。这里是第一部分数据处理汇报中的解释性权重，使用单变量相关性计算，用来说明特征和目标之间的关系强弱。

- 特征名是从哪里来的？
  不是手工写的。代码中 `make_target_features()` 返回 `MODEL_WEATHER + CALENDAR_FEATURES + lag_features`，所以表里的特征名就是实际进入建模数据表的字段。

- 特征的数学含义怎么联系现实？
  例如 `lag_7` 数学上是前 7 天目标值，现实上是上周同一天用电习惯；`roll_mean_7` 数学上是过去 7 天均值，现实上是近期用电水平；`hdd/cdd` 数学上是温度阈值函数，现实上是供热/制冷需求。

- 天气衍生变量是怎么想到的？
  先从业务常识联想：温度影响空调和取暖用电；再从数据验证：画温度、CDD、HDD 与负荷的散点图；最后在代码里按 `YMD` 合并天气，并用公式构造 `temp_range`、`hdd`、`cdd`。
"""
    (OUT / "report1_presentation_guide.md").write_text(guide, encoding="utf-8")


def make_speaker_notes(profile: pd.DataFrame) -> None:
    """生成可照读的中文讲稿。"""
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

## 分布与异常值

我先看了原始 15 分钟负荷、日粒度负荷和天气变量的分布，再用箱线图识别极端候选值。这里采用 IQR 规则：

- 下界 = Q1 - 1.5 × IQR
- 上界 = Q3 + 1.5 × IQR

但这里要强调：IQR 标出来的是统计极端值，不等于错误数据。电力负荷受极端天气、节假日和生产活动影响很明显，这些极端样本本身很有研究价值，所以我不删除、不截尾，而是保留原始值并增加极端事件标记。真正要修正的是物理不可能值，比如负负荷、湿度超过 100、负降雨等。

## 特征工程

特征工程分成三类：

1. 天气特征：最高温、最低温、平均温、湿度、降雨量。
2. 日历周期特征：星期、月份、是否周末，以及 sin/cos 周期编码。
3. 历史负荷特征：昨天、上周同日、两周前同日，以及 7 天和 14 天滚动均值、滚动标准差。

天气特征的来源不是随便加的。首先从业务上看，高温会增加空调制冷负荷，低温会增加供热或取暖负荷；然后从数据上画散点图，观察平均温度、CDD、HDD 与负荷确实有关；最后在代码上用 `YMD` 把天气表和日负荷表合并，再构造 `temp_range`、`hdd`、`cdd` 三个衍生变量。

这里特别注意防止数据泄漏。滚动特征先使用 `shift(1)`，表示预测今天时只使用昨天及以前的信息，不会偷看当天真实负荷。

## 特征值演示和权重

我还给每个建模特征做了示例值和参考权重。特征名不是手工编出来的，而是代码自动返回的 `MODEL_WEATHER + CALENDAR_FEATURES + lag_features`。

这些特征可以这样解释：天气衍生变量来自第 8 节，例如 `hdd` 和 `cdd` 表示供热、制冷需求；日历周期变量表示季节和星期规律；滞后变量表示昨天、上周同日、两周前同日的用电惯性；滚动均值和滚动标准差表示近期负荷水平和波动程度。

示例值取 2015 年 1 月 10 日，因为它是预测窗口前最后一条完整历史样本。权重不是最终模型参数，而是数据处理阶段用单变量相关性归一化得到的解释性排序，用来说明哪些特征和负荷目标的关系更强。

## 结尾

第一次汇报的结论是：我们已经把原始 Excel 处理成了可用于建模的日粒度特征表。下一次汇报会基于这张表展示算法模型选择、训练过程、调参和模型对比。
"""
    (OUT / "report1_speaker_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    """脚本入口：按顺序完成数据读取、分析、画图和材料生成。"""
    ensure_output_dirs()
    set_chinese_font()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find data file: {DATA_PATH}")

    # 1. 读取原始 Excel 两个工作表。
    load_raw = pd.read_excel(DATA_PATH, sheet_name="Area_Load")
    weather_raw = pd.read_excel(DATA_PATH, sheet_name="Area_Weather")
    weather_named = weather_raw.copy()
    weather_named.columns = ["YMD", "temp_max", "temp_min", "temp_avg", "humidity", "rainfall"]
    # 2. 调用项目主脚本里的日粒度处理函数，保证和正式项目逻辑一致。
    daily = p2.load_daily_data()

    # 3. 极端候选值识别：检测用原始 daily，后续保留原值并增加标记。
    outlier_summary = detect_outliers_iqr(daily, ["load_max", "load_min", "load_mean", "temp_avg", "humidity", "rainfall"])
    quality_checks = physical_quality_checks(load_raw, daily)
    flagged_daily = add_extreme_event_flags(daily, outlier_summary)
    # 4. 计算特征参考权重，并生成“特征名-中文含义-示例值”的讲解表。
    weights = compute_feature_weights(flagged_daily)
    feature_dictionary = build_feature_dictionary(flagged_daily, weights)
    # 5. 汇总本次汇报开头的数据检查结果。
    profile = pd.concat(
        [
            profile_data(load_raw, weather_raw, daily),
            pd.DataFrame(
                [
                    {
                        "检查项": "IQR 检测字段数",
                        "结果": f"{len(outlier_summary)} 个",
                        "说明": "对负荷目标和主要天气变量做异常值检测。",
                    },
                    {
                        "检查项": "IQR 极端候选处理方式",
                        "结果": "保留原值并增加标记",
                        "说明": "极端天气/负荷有研究价值，不直接删除或截尾。",
                    },
                    {
                        "检查项": "特征参考权重字段数",
                        "结果": f"{len(feature_dictionary)} 个",
                        "说明": "按目标变量分别计算解释性权重。",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    # 6. 导出 CSV：如果老师问具体数值，可以直接打开这些表查。
    safe_to_csv(profile, OUT / "report1_data_profile.csv")
    safe_to_csv(flagged_daily.head(20), OUT / "report1_daily_processed_sample.csv")
    safe_to_csv(outlier_summary, OUT / "report1_outlier_summary.csv")
    safe_to_csv(quality_checks, OUT / "report1_physical_quality_checks.csv")
    safe_to_csv(weights, OUT / "report1_feature_weights.csv")
    safe_to_csv(feature_dictionary, OUT / "report1_feature_dictionary.csv")

    time_cols = [c for c in load_raw.columns if c != "YMD"]
    # 原始负荷表列太多，这里只截取前 10 个时刻和最后 2 个时刻做截图。
    raw_load_view = load_raw[["YMD", *time_cols[:10], *time_cols[-2:]]]
    # 展示处理后的日粒度表：目标负荷 + 天气 + 衍生天气变量。
    daily_view = flagged_daily[
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

    # 统一登记所有图片输出路径，后面 HTML 通过这些 key 找到图片。
    assets = {
        "raw_load": ASSETS / "01_raw_load_sample.png",
        "raw_weather": ASSETS / "02_raw_weather_sample.png",
        "pipeline": ASSETS / "03_data_pipeline.png",
        "missingness": ASSETS / "04_missingness_check.png",
        "raw_15min_dist": ASSETS / "04_raw_15min_distribution.png",
        "daily_distribution": ASSETS / "05_daily_load_distribution.png",
        "weather_distribution": ASSETS / "06_weather_distribution.png",
        "box_before_after": ASSETS / "07_box_before_after.png",
        "distribution_before_after": ASSETS / "08_distribution_before_after.png",
        "outlier_timeline": ASSETS / "09_outlier_timeline.png",
        "load_2014": ASSETS / "10_2014_daily_load.png",
        "duration_2014": ASSETS / "11_2014_load_duration.png",
        "weather_load_reasoning": ASSETS / "12_weather_load_reasoning.png",
        "merged_sample": ASSETS / "12_merged_daily_feature_sample.png",
        "feature_blocks": ASSETS / "13_feature_engineering_blocks.png",
        "code_load_daily": ASSETS / "14_code_load_daily_data.png",
        "code_outliers": ASSETS / "15_code_outliers.png",
        "code_add_features": ASSETS / "16_code_add_features.png",
        "code_target_features": ASSETS / "17_code_make_target_features.png",
        "feature_demo": ASSETS / "18_feature_demo.png",
        "feature_weights": ASSETS / "19_feature_weights.png",
        "code_feature_weights": ASSETS / "20_code_feature_weights.png",
        "feature_source_map": ASSETS / "21_feature_source_map.png",
    }

    # 7. 生成表格截图和统计图。
    save_df_image(raw_load_view, assets["raw_load"], "Area_Load 原始负荷表节选")
    save_df_image(weather_named, assets["raw_weather"], "Area_Weather 原始天气表节选")
    plot_pipeline(assets["pipeline"])
    plot_missingness(load_raw, weather_raw, assets["missingness"])
    plot_raw_15min_distribution(load_raw, assets["raw_15min_dist"])
    plot_daily_distribution(flagged_daily, assets["daily_distribution"])
    plot_weather_distribution(flagged_daily, assets["weather_distribution"])
    plot_outlier_box_compare(daily, flagged_daily, assets["box_before_after"])
    plot_before_after_distribution(daily, flagged_daily, assets["distribution_before_after"])
    plot_outlier_timeline(daily, outlier_summary, assets["outlier_timeline"])
    plot_2014_daily_load(daily, assets["load_2014"])
    plot_load_duration(daily, assets["duration_2014"])
    plot_weather_load_reasoning(flagged_daily, assets["weather_load_reasoning"])
    save_df_image(daily_view, assets["merged_sample"], "日粒度负荷 + 天气 + 衍生变量节选", max_rows=10)
    plot_feature_blocks(assets["feature_blocks"])
    save_df_image(feature_dictionary.head(18), assets["feature_demo"], "特征值演示节选", max_rows=18)
    plot_feature_weights(weights, assets["feature_weights"])
    plot_feature_source_map(assets["feature_source_map"])

    # 8. 生成核心代码截图。code_sources 同时会被 HTML 的代码块引用。
    code_sources = {
        "load_daily_data": save_code_image(p2.load_daily_data, assets["code_load_daily"], "核心代码 1：读取、重构、聚合、合并"),
        "outliers": save_code_image(
            lambda: None,
            assets["code_outliers"],
            "核心代码 2：IQR 异常值检测与截尾处理",
        ),
        "add_features": save_code_image(p2.add_features, assets["code_add_features"], "核心代码 3：日历周期与天气衍生特征"),
        "make_target_features": save_code_image(
            p2.make_target_features,
            assets["code_target_features"],
            "核心代码 4：滞后与滚动特征，避免数据泄漏",
        ),
        "feature_weights": save_code_image(
            lambda: None,
            assets["code_feature_weights"],
            "核心代码 5：特征参考权重计算",
        ),
    }

    # 下面两个 code_sources 手动拼接多个函数，方便 HTML 一次展示完整逻辑。
    code_sources["outliers"] = textwrap.dedent(
        f"""
        {inspect.getsource(detect_outliers_iqr)}

        {inspect.getsource(add_extreme_event_flags)}

        {inspect.getsource(physical_quality_checks)}
        """
    ).strip()
    code_sources["feature_weights"] = textwrap.dedent(
        f"""
        {inspect.getsource(compute_feature_weights)}

        {inspect.getsource(build_feature_dictionary)}
        """
    ).strip()

    # 重新渲染更有意义的函数截图，覆盖前面 lambda 占位生成的图片。
    save_code_image(detect_outliers_iqr, assets["code_outliers"], "核心代码 2：IQR 异常值检测")
    save_code_image(compute_feature_weights, assets["code_feature_weights"], "核心代码 5：特征参考权重计算")

    # 9. 生成最终交付文件：HTML、Markdown、讲稿、演示指南。
    make_html(profile, outlier_summary, quality_checks, feature_dictionary, assets, code_sources)
    make_markdown(profile, outlier_summary, quality_checks, feature_dictionary, code_sources)
    make_presentation_guide()
    make_speaker_notes(profile)

    print(f"Generated first report package: {OUT}")
    print(f"HTML: {OUT / 'report1_data_processing.html'}")
    print(f"Speaker notes: {OUT / 'report1_speaker_notes.md'}")


if __name__ == "__main__":
    main()
