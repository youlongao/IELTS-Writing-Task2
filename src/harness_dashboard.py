"""Streamlit dashboard for Harness observability."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness.observability import SCORE_METRICS, load_runs, summarize_rows


def zh(text: str) -> str:
    """Decode unicode escapes while keeping source mostly ASCII."""
    return text.encode("ascii").decode("unicode_escape")


METRIC_LABELS = {
    "task_response": zh("\\u4efb\\u52a1\\u56de\\u5e94"),
    "coherence_cohesion": zh("\\u8fde\\u8d2f\\u4e0e\\u8854\\u63a5"),
    "lexical_resource": zh("\\u8bcd\\u6c47\\u8d44\\u6e90"),
    "grammatical_range": zh("\\u8bed\\u6cd5\\u8303\\u56f4\\u4e0e\\u51c6\\u786e\\u5ea6"),
    "specificity_score": zh("\\u5177\\u4f53\\u5ea6"),
    "band_alignment": zh("Band \\u5bf9\\u9f50\\u5ea6"),
    "overall": zh("\\u7efc\\u5408\\u5206"),
}

ERROR_LABELS = {
    "generic_or_underdeveloped": zh("\\u5185\\u5bb9\\u7a7a\\u6cdb/\\u5c55\\u5f00\\u4e0d\\u8db3"),
    "no_major_error": zh("\\u65e0\\u660e\\u663e\\u95ee\\u9898"),
    "weak_task_response": zh("\\u4efb\\u52a1\\u56de\\u5e94\\u4e0d\\u8db3"),
    "weak_coherence": zh("\\u8fde\\u8d2f\\u8854\\u63a5\\u4e0d\\u8db3"),
    "weak_lexical_resource": zh("\\u8bcd\\u6c47\\u8d44\\u6e90\\u4e0d\\u8db3"),
    "weak_grammar_range": zh("\\u8bed\\u6cd5\\u8303\\u56f4\\u4e0d\\u8db3"),
    "no_rag_chunks": zh("RAG \\u672a\\u53ec\\u56de\\u7247\\u6bb5"),
    "runtime_error": zh("\\u8fd0\\u884c\\u9519\\u8bef"),
}


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def error_label(label: str) -> str:
    return ERROR_LABELS.get(label, label.replace("_", " "))


def error_labels_zh(labels) -> str:
    if not isinstance(labels, list):
        return ""
    return " | ".join(error_label(label) for label in labels)


st.set_page_config(
    page_title="Harness Observability",
    page_icon="HO",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Harness 数据观测面板 (Harness Observability)")
st.caption("按时间线观察范文生成质量、错误标签、token 成本和调用耗时。")


@st.cache_data(show_spinner=False)
def cached_load_runs(runs_dir: str):
    return load_runs(runs_dir)


with st.sidebar:
    st.header("数据源")
    runs_dir = st.text_input("Harness JSON 目录", value="runs")
    if st.button("刷新数据", use_container_width=True):
        cached_load_runs.clear()
        st.rerun()

rows = cached_load_runs(runs_dir)

if not rows:
    st.info("暂未发现 Harness JSON。请先运行 compare/evaluator 并导出到 runs/*.json。")
    st.code(
        ".\\.venv\\Scripts\\python.exe -m src.harness.compare "
        "--provider-a deepseek --provider-b openai --judge-provider deepseek "
        "--test-count 5 --seed 42 --json-output runs\\compare_sample_essay.json",
        language="powershell",
    )
    st.stop()

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - Streamlit normally includes pandas.
    st.error(f"无法加载 pandas，面板无法渲染：{exc}")
    st.stop()

df = pd.DataFrame(rows)
df["run_time_dt"] = pd.to_datetime(df["run_time"], errors="coerce")
df = df.sort_values("run_time_dt")

with st.sidebar:
    st.header("筛选")
    min_time = df["run_time_dt"].min()
    max_time = df["run_time_dt"].max()
    if pd.notna(min_time) and pd.notna(max_time):
        date_range = st.date_input(
            "时间范围",
            value=(min_time.date(), max_time.date()),
        )
    else:
        date_range = ()

    providers = sorted(value for value in df["provider"].dropna().unique() if value)
    selected_providers = st.multiselect("生成 Provider", providers, default=providers)

    judges = sorted(value for value in df["fixed_judge"].dropna().unique() if value)
    selected_judges = st.multiselect("Judge", judges, default=judges)

    targets = sorted(value for value in df["harness_target"].dropna().unique() if value)
    selected_targets = st.multiselect("评测对象", targets, default=targets)

    all_labels = sorted({
        label
        for labels in df["error_labels"].dropna()
        for label in (labels if isinstance(labels, list) else [])
    })
    selected_labels = st.multiselect("错误标签", all_labels, format_func=error_label)

filtered = df.copy()
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        (filtered["run_time_dt"].dt.date >= start_date)
        & (filtered["run_time_dt"].dt.date <= end_date)
    ]
if selected_providers:
    filtered = filtered[filtered["provider"].isin(selected_providers)]
if selected_judges:
    filtered = filtered[filtered["fixed_judge"].isin(selected_judges)]
if selected_targets:
    filtered = filtered[filtered["harness_target"].isin(selected_targets)]
if selected_labels:
    filtered = filtered[
        filtered["error_labels"].apply(
            lambda labels: any(label in labels for label in selected_labels)
            if isinstance(labels, list)
            else False
        )
    ]

summary = summarize_rows(filtered.to_dict("records"))

card1, card2, card3, card4, card5 = st.columns(5)
card1.metric("运行次数", summary["runs"])
card2.metric("Case 数", summary["cases"])
card3.metric("平均 Overall", f"{summary['avg_overall']:.2f}")
card4.metric("平均耗时", f"{summary['avg_latency']:.1f}s")
card5.metric("Token 总量", f"{summary['generation_tokens'] + summary['judge_tokens']:,}")

st.divider()

if filtered.empty:
    st.warning("当前筛选条件下没有数据。")
    st.stop()

trend_df = filtered.set_index("run_time_dt")
provider_options = sorted(filtered["provider"].dropna().unique())

left, right = st.columns([2, 1])
with left:
    st.subheader("时间线：核心分数趋势")
    metric = st.selectbox(
        "趋势指标",
        SCORE_METRICS,
        index=SCORE_METRICS.index("overall"),
        format_func=metric_label,
    )
    chart_data = (
        filtered
        .groupby([pd.Grouper(key="run_time_dt", freq="1min"), "provider"])[metric]
        .mean()
        .reset_index()
    )
    chart = (
        alt.Chart(chart_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("run_time_dt:T", title="时间"),
            y=alt.Y(f"{metric}:Q", title=metric_label(metric), scale=alt.Scale(domain=[0, 9])),
            color=alt.Color("provider:N", title="Provider"),
            tooltip=[
                alt.Tooltip("run_time_dt:T", title="时间"),
                alt.Tooltip("provider:N", title="Provider"),
                alt.Tooltip(f"{metric}:Q", title=metric_label(metric), format=".2f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

with right:
    st.subheader("当前问题分布")
    label_counts = summary["top_error_labels"]
    if label_counts:
        label_df = pd.DataFrame(
            [
                {"问题标签": error_label(label), "数量": count}
                for label, count in label_counts.items()
            ]
        )
        label_chart = (
            alt.Chart(label_df)
            .mark_bar()
            .encode(
                x=alt.X("问题标签:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("数量:Q", title="数量"),
                tooltip=[
                    alt.Tooltip("问题标签:N", title="问题标签"),
                    alt.Tooltip("数量:Q", title="数量"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(label_chart, use_container_width=True)
    else:
        st.info("没有错误标签。")

st.subheader("指标均值对比")
metric_table = (
    filtered
    .groupby(["provider", "fixed_judge"])[SCORE_METRICS + ["latency_seconds", "generation_total_tokens", "judge_total_tokens"]]
    .mean()
    .round(2)
    .reset_index()
)
metric_table_display = metric_table.rename(columns={
    "provider": "生成 Provider",
    "fixed_judge": "Judge",
    "latency_seconds": "平均耗时(秒)",
    "generation_total_tokens": "生成 Token",
    "judge_total_tokens": "Judge Token",
    **METRIC_LABELS,
})
st.dataframe(metric_table_display, use_container_width=True, hide_index=True)

cost_col, latency_col = st.columns(2)
with cost_col:
    st.subheader("Token 成本趋势")
    token_chart = (
        filtered
        .assign(total_tokens=filtered["generation_total_tokens"] + filtered["judge_total_tokens"])
        .groupby([pd.Grouper(key="run_time_dt", freq="1min"), "provider"])["total_tokens"]
        .sum()
        .reset_index()
        .pivot(index="run_time_dt", columns="provider", values="total_tokens")
    )
    if len(token_chart.index) <= 1:
        st.caption("当前只有一个时间点，先用柱状图展示；积累多次测试后会自动切换为折线趋势。")
        st.bar_chart(token_chart)
    else:
        st.line_chart(token_chart)

with latency_col:
    st.subheader("耗时趋势")
    latency_chart = (
        filtered
        .groupby([pd.Grouper(key="run_time_dt", freq="1min"), "provider"])["latency_seconds"]
        .mean()
        .reset_index()
        .pivot(index="run_time_dt", columns="provider", values="latency_seconds")
    )
    if len(latency_chart.index) <= 1:
        st.caption("当前只有一个时间点，先用柱状图展示；积累多次测试后会自动切换为折线趋势。")
        st.bar_chart(latency_chart)
    else:
        st.line_chart(latency_chart)

st.subheader("Case 级详情")
detail_columns = [
    "run_time",
    "run_id",
    "test_case_id",
    "provider",
    "fixed_judge",
    "overall",
    "task_response",
    "coherence_cohesion",
    "lexical_resource",
    "grammatical_range",
    "specificity_score",
    "latency_seconds",
    "generation_total_tokens",
    "judge_total_tokens",
    "error_labels",
    "question",
    "file_name",
]
detail_display = filtered[detail_columns].copy()
detail_display["error_labels"] = detail_display["error_labels"].apply(error_labels_zh)
detail_display = detail_display.rename(columns={
    "run_time": "运行时间",
    "run_id": "运行 ID",
    "test_case_id": "题目 ID",
    "provider": "生成 Provider",
    "fixed_judge": "Judge",
    "latency_seconds": "耗时(秒)",
    "generation_total_tokens": "生成 Token",
    "judge_total_tokens": "Judge Token",
    "error_labels": "错误标签",
    "question": "题目",
    "file_name": "来源文件",
    **METRIC_LABELS,
})
st.dataframe(
    detail_display.sort_values("运行时间", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "下载当前筛选数据 CSV",
    data=detail_display.to_csv(index=False).encode("utf-8-sig"),
    file_name="harness_observability_filtered.csv",
    mime="text/csv",
)
