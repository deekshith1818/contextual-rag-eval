"""
dashboard/app.py
----------------
Phase 7 — k8s-RAG Observability Dashboard

Visualises query logs produced by src/observability/logger.py (Phase 6).
Loads all logs/queries_*.jsonl files and renders:

  • Headline metric cards (total queries, avg latency, total cost, avg cost/query)
  • Latency breakdown chart (stacked bar: retrieval / reranking / generation per query)
  • Cost timeline (per-query + cumulative line chart)
  • Pipeline comparison (baseline vs contextual — latency & cost side-by-side)
  • Raw query log table (searchable, sortable)

NOTE: Dashboard is built to scale — 16 data points shown here is demo-scale data
      produced by smoke_test_observability.py. Charts will fill in naturally as more
      queries are logged through the live pipeline.

Run locally:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="k8s-RAG Observability",
    page_icon="☸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR      = _PROJECT_ROOT / "logs"

# ── Colour palette ────────────────────────────────────────────────────────────
CLR_BASELINE    = "#6366f1"   # indigo
CLR_CONTEXTUAL  = "#10b981"   # emerald
CLR_RETRIEVAL   = "#f59e0b"   # amber
CLR_RERANKING   = "#ef4444"   # red  (the bottleneck)
CLR_GENERATION  = "#3b82f6"   # blue
CLR_BG          = "#0f172a"   # slate-900
CLR_SURFACE     = "#1e293b"   # slate-800
CLR_BORDER      = "#334155"   # slate-700
CLR_TEXT        = "#f1f5f9"   # slate-100
CLR_MUTED       = "#94a3b8"   # slate-400

PLOTLY_TEMPLATE = "plotly_dark"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Global ── */
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {CLR_BG};
      color: {CLR_TEXT};
      font-family: 'Inter', 'Segoe UI', sans-serif;
  }}
  [data-testid="stSidebar"] {{
      background-color: {CLR_SURFACE};
      border-right: 1px solid {CLR_BORDER};
  }}

  /* ── Metric cards ── */
  [data-testid="metric-container"] {{
      background: {CLR_SURFACE};
      border: 1px solid {CLR_BORDER};
      border-radius: 12px;
      padding: 1rem 1.25rem;
      transition: box-shadow .2s;
  }}
  [data-testid="metric-container"]:hover {{
      box-shadow: 0 0 0 2px {CLR_BASELINE}55;
  }}
  [data-testid="stMetricLabel"]  {{ color: {CLR_MUTED}; font-size: 0.8rem; }}
  [data-testid="stMetricValue"]  {{ color: {CLR_TEXT};  font-size: 1.7rem; font-weight: 700; }}
  [data-testid="stMetricDelta"]  {{ font-size: 0.78rem; }}

  /* ── Section headers ── */
  .section-header {{
      font-size: 1.05rem; font-weight: 600;
      color: {CLR_TEXT};
      border-left: 3px solid {CLR_BASELINE};
      padding-left: .65rem; margin: 1.5rem 0 .75rem 0;
  }}

  /* ── Divider ── */
  hr {{ border-color: {CLR_BORDER}; opacity: .4; }}

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}

  /* ── Badges ── */
  .badge-baseline   {{ background:{CLR_BASELINE};   color:#fff; border-radius:4px; padding:2px 8px; font-size:.75rem; }}
  .badge-contextual {{ background:{CLR_CONTEXTUAL}; color:#fff; border-radius:4px; padding:2px 8px; font-size:.75rem; }}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)   # refresh every 30s so live runs show up quickly
def load_all_logs() -> pd.DataFrame:
    """Read every logs/queries_*.jsonl file and return a combined DataFrame."""
    records = []
    for path in sorted(LOGS_DIR.glob("queries_*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Parse timestamp → UTC-aware datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"]      = df["timestamp"].dt.date

    # Friendly label for the query (truncated)
    df["query_short"] = df["query"].str[:55] + "…"

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["query_num"] = range(1, len(df) + 1)

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _plotly_cfg() -> dict:
    return {"displayModeBar": False}


def section(label: str) -> None:
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex; align-items:center; gap:.75rem; margin-bottom:.25rem'>
        <span style='font-size:2rem'>☸️</span>
        <div>
            <h1 style='margin:0; font-size:1.6rem; font-weight:800; color:#f1f5f9'>
                k8s-RAG Observability
            </h1>
            <p style='margin:0; font-size:.85rem; color:#94a3b8'>
                Real-time query latency · cost tracking · pipeline comparison
            </p>
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df_all = load_all_logs()

    if df_all.empty:
        st.warning(
            "No log files found in `logs/`. "
            "Run `python smoke_test_observability.py` first to generate data.",
            icon="⚠️",
        )
        return

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Filters")

        min_date = df_all["date"].min()
        max_date = df_all["date"].max()

        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        # Gracefully handle single-date selection
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range[0] if date_range else min_date

        pipeline_opts = ["All"] + sorted(df_all["pipeline"].unique().tolist())
        selected_pipe = st.selectbox("Pipeline", pipeline_opts)

        model_opts = ["All"] + sorted(df_all["generation_model"].unique().tolist())
        selected_model = st.selectbox("Generation model", model_opts)

        st.markdown("---")
        st.caption(f"📁 Log dir: `{LOGS_DIR.relative_to(_PROJECT_ROOT)}`")
        st.caption(f"📄 Files: {len(list(LOGS_DIR.glob('queries_*.jsonl')))}")
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Apply filters ─────────────────────────────────────────────────────────
    mask = (df_all["date"] >= start_date) & (df_all["date"] <= end_date)
    if selected_pipe != "All":
        mask &= df_all["pipeline"] == selected_pipe
    if selected_model != "All":
        mask &= df_all["generation_model"] == selected_model
    df = df_all[mask].copy()

    if df.empty:
        st.info("No events match the current filters.", icon="ℹ️")
        return

    # ── 1. Headline metric cards ───────────────────────────────────────────────
    section("📊 Headline Metrics")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Queries",       f"{len(df):,}")
    c2.metric("Avg End-to-End",      f"{df['total_latency_ms'].mean()/1000:.2f}s")
    c3.metric("Reranker Avg",        f"{df['rerank_latency_ms'].mean()/1000:.2f}s",
              delta="bottleneck ⚠️", delta_color="inverse")
    c4.metric("Total Cost",          f"${df['cost_usd'].sum():.4f}")
    c5.metric("Avg Cost / Query",    f"${df['cost_usd'].mean():.4f}")

    st.markdown("")

    # ── 2. Latency breakdown — stacked bar per query ───────────────────────────
    section("⏱️ Latency Breakdown — Per Query (retrieval · reranking · generation)")
    st.caption(
        "Stack shows the three pipeline stages. The **red reranking layer** dominates "
        "on CPU — this is the `BAAI/bge-reranker-base` cross-encoder."
    )

    fig_lat = go.Figure()
    x_labels = df["query_short"].tolist()
    pip_colors = df["pipeline"].map({"baseline": CLR_BASELINE, "contextual": CLR_CONTEXTUAL})

    fig_lat.add_trace(go.Bar(
        name="Retrieval (hybrid+RRF)",
        x=x_labels,
        y=df["retrieval_latency_ms"] / 1000,
        marker_color=CLR_RETRIEVAL,
        hovertemplate="<b>%{x}</b><br>Retrieval: %{y:.2f}s<extra></extra>",
    ))
    fig_lat.add_trace(go.Bar(
        name="Reranking (cross-encoder)",
        x=x_labels,
        y=df["rerank_latency_ms"] / 1000,
        marker_color=CLR_RERANKING,
        hovertemplate="<b>%{x}</b><br>Reranking: %{y:.2f}s<extra></extra>",
    ))
    fig_lat.add_trace(go.Bar(
        name="Generation (LLM)",
        x=x_labels,
        y=df["generation_latency_ms"] / 1000,
        marker_color=CLR_GENERATION,
        hovertemplate="<b>%{x}</b><br>Generation: %{y:.2f}s<extra></extra>",
    ))
    fig_lat.update_layout(
        barmode="stack",
        template=PLOTLY_TEMPLATE,
        height=380,
        margin=dict(l=0, r=0, t=10, b=140),
        legend=dict(orientation="h", yanchor="top", y=-0.45, xanchor="center", x=0.5),
        xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(title="Latency (s)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_lat, use_container_width=True, config=_plotly_cfg())

    # ── 3. Cost timeline ──────────────────────────────────────────────────────
    section("💰 Cost Over Time")

    col_a, col_b = st.columns(2)

    with col_a:
        st.caption("Per-query cost (coloured by pipeline)")
        pipe_color_map = {"baseline": CLR_BASELINE, "contextual": CLR_CONTEXTUAL}
        fig_cost = px.scatter(
            df,
            x="timestamp", y="cost_usd",
            color="pipeline",
            color_discrete_map=pipe_color_map,
            hover_data={"query_short": True, "total_latency_ms": ":.0f"},
            labels={"cost_usd": "Cost (USD)", "timestamp": "Time"},
            template=PLOTLY_TEMPLATE,
        )
        fig_cost.update_traces(marker=dict(size=10, line=dict(width=1, color="#1e293b")))
        fig_cost.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(title=""),
        )
        st.plotly_chart(fig_cost, use_container_width=True, config=_plotly_cfg())

    with col_b:
        st.caption("Cumulative cost over queries")
        df_sorted     = df.sort_values("timestamp").copy()
        df_sorted["cumulative_cost"] = df_sorted["cost_usd"].cumsum()
        fig_cum = px.line(
            df_sorted,
            x="query_num", y="cumulative_cost",
            color="pipeline",
            color_discrete_map=pipe_color_map,
            markers=True,
            labels={"cumulative_cost": "Cumulative Cost (USD)", "query_num": "Query #"},
            template=PLOTLY_TEMPLATE,
        )
        fig_cum.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(title=""),
        )
        st.plotly_chart(fig_cum, use_container_width=True, config=_plotly_cfg())

    # ── 4. Pipeline comparison ─────────────────────────────────────────────────
    section("🔀 Pipeline Comparison — Baseline vs Contextual")

    pipes = df["pipeline"].unique().tolist()
    if len(pipes) < 2:
        st.info(
            "Filter includes only one pipeline. Select 'All' to see the comparison.",
            icon="ℹ️",
        )
    else:
        col_l, col_r = st.columns(2)

        # Latency comparison — grouped bar
        with col_l:
            st.caption("Avg stage latency by pipeline (s)")
            lat_agg = (
                df.groupby("pipeline")[
                    ["retrieval_latency_ms", "rerank_latency_ms", "generation_latency_ms"]
                ].mean() / 1000
            ).rename(columns={
                "retrieval_latency_ms":  "Retrieval",
                "rerank_latency_ms":     "Reranking",
                "generation_latency_ms": "Generation",
            }).reset_index()

            fig_cmp = go.Figure()
            stage_colors = {
                "Retrieval":  CLR_RETRIEVAL,
                "Reranking":  CLR_RERANKING,
                "Generation": CLR_GENERATION,
            }
            for stage, color in stage_colors.items():
                fig_cmp.add_trace(go.Bar(
                    name=stage,
                    x=lat_agg["pipeline"],
                    y=lat_agg[stage],
                    marker_color=color,
                    hovertemplate=f"{stage}: %{{y:.2f}}s<extra></extra>",
                ))
            fig_cmp.update_layout(
                barmode="group",
                template=PLOTLY_TEMPLATE,
                height=300,
                margin=dict(l=0, r=0, t=10, b=10),
                yaxis=dict(title="Latency (s)"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_cmp, use_container_width=True, config=_plotly_cfg())

        # Cost comparison — violin / box
        with col_r:
            st.caption("Cost per query distribution by pipeline (USD)")
            fig_vio = px.box(
                df,
                x="pipeline", y="cost_usd",
                color="pipeline",
                color_discrete_map=pipe_color_map,
                points="all",
                labels={"cost_usd": "Cost (USD)", "pipeline": "Pipeline"},
                template=PLOTLY_TEMPLATE,
            )
            fig_vio.update_traces(marker=dict(size=7))
            fig_vio.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_vio, use_container_width=True, config=_plotly_cfg())

        # Summary stats table
        st.markdown("")
        summary = df.groupby("pipeline").agg(
            Queries        = ("query", "count"),
            Avg_E2E_s      = ("total_latency_ms", lambda x: f"{x.mean()/1000:.2f}s"),
            P95_E2E_s      = ("total_latency_ms", lambda x: f"{x.quantile(.95)/1000:.2f}s"),
            Avg_Cost_USD   = ("cost_usd",  lambda x: f"${x.mean():.4f}"),
            Total_Cost_USD = ("cost_usd",  lambda x: f"${x.sum():.4f}"),
            Avg_In_Tok     = ("input_tokens",  "mean"),
            Avg_Out_Tok    = ("output_tokens", "mean"),
        ).reset_index()
        summary.columns = [c.replace("_", " ") for c in summary.columns]
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── 5. Token usage ────────────────────────────────────────────────────────
    section("🔤 Token Usage per Query")

    fig_tok = go.Figure()
    fig_tok.add_trace(go.Bar(
        name="Input tokens",
        x=x_labels,
        y=df["input_tokens"],
        marker_color=CLR_BASELINE,
        hovertemplate="%{x}<br>Input: %{y:,}<extra></extra>",
    ))
    fig_tok.add_trace(go.Bar(
        name="Output tokens",
        x=df["query_short"].tolist(),
        y=df["output_tokens"],
        marker_color=CLR_CONTEXTUAL,
        hovertemplate="%{x}<br>Output: %{y:,}<extra></extra>",
    ))
    fig_tok.update_layout(
        barmode="group",
        template=PLOTLY_TEMPLATE,
        height=300,
        margin=dict(l=0, r=0, t=10, b=140),
        xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(title="Tokens"),
        legend=dict(orientation="h", yanchor="top", y=-0.45, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_tok, use_container_width=True, config=_plotly_cfg())

    # ── 6. Raw query log table ─────────────────────────────────────────────────
    section("🗂️ Raw Query Events")

    search = st.text_input("🔍 Search queries", placeholder="e.g. ConfigMap, DaemonSet…")
    df_table = df.copy()
    if search:
        df_table = df_table[df_table["query"].str.contains(search, case=False, na=False)]

    display_cols = {
        "timestamp":            "Timestamp (UTC)",
        "pipeline":             "Pipeline",
        "query":                "Query",
        "total_latency_ms":     "E2E (ms)",
        "retrieval_latency_ms": "Retrieval (ms)",
        "rerank_latency_ms":    "Rerank (ms)",
        "generation_latency_ms":"Generation (ms)",
        "input_tokens":         "Input Tok",
        "output_tokens":        "Output Tok",
        "cost_usd":             "Cost (USD)",
    }
    df_display = df_table[list(display_cols.keys())].rename(columns=display_cols)
    df_display["Timestamp (UTC)"] = df_display["Timestamp (UTC)"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for ms_col in ["E2E (ms)", "Retrieval (ms)", "Rerank (ms)", "Generation (ms)"]:
        df_display[ms_col] = df_display[ms_col].round(0).astype(int)
    df_display["Cost (USD)"] = df_display["Cost (USD)"].apply(lambda x: f"${x:.5f}")

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=min(400, 50 + len(df_display) * 35),
    )
    st.caption(
        f"Showing {len(df_display)} of {len(df)} events. "
        "Demo-scale data (16 queries from smoke_test_observability.py). "
        "Charts scale automatically as more queries are logged."
    )

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "k8s-RAG Eval · Phase 7 · "
        "[GitHub](https://github.com/deekshith1818/contextual-rag-eval) · "
        f"Data from `{LOGS_DIR.relative_to(_PROJECT_ROOT)}/`"
    )


if __name__ == "__main__":
    main()
