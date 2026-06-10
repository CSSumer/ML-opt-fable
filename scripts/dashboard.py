#!/usr/bin/env python3
"""Non-quant-friendly results dashboard.

    streamlit run scripts/dashboard.py

Dual-layer design:
* Executive Summary — Growth of $10,000 vs. baseline, plain-English scorecard.
* Risk vs. Return Landscape — interactive scatter of every strategy.
* Advanced Quantitative Analytics — raw metrics, corrected p-values and
  resource logs hidden in a collapsible accordion so the main view stays
  uncluttered.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

PLAIN_ENGLISH = {
    "annualized_return": ("Average Yearly Growth",
                          "How much the portfolio grew per year, on average."),
    "annualized_volatility": ("Bumpiness of the Ride",
                              "How much the value swings around — lower is calmer."),
    "sharpe_ratio": ("Reward per Unit of Risk",
                     "Return earned for each unit of risk taken — higher is better."),
    "sortino_ratio": ("Reward per Unit of Downside",
                      "Like the above, but only penalises losses, not gains."),
    "max_drawdown": ("Worst-Case Historical Drop",
                     "The deepest peak-to-trough loss an investor would have felt."),
    "calmar_ratio": ("Growth vs. Worst Drop",
                     "Yearly growth divided by the worst drop — higher is better."),
    "avg_monthly_turnover": ("Monthly Trading Activity",
                             "How much of the portfolio is traded each month."),
}


@st.cache_data
def load_runs() -> dict[str, Path]:
    if not RESULTS_DIR.exists():
        return {}
    runs = sorted(
        (p for p in RESULTS_DIR.iterdir() if (p / "results.pkl").exists()),
        reverse=True,
    )
    return {p.name: p for p in runs}


def load_run(run_dir: Path):
    with open(run_dir / "results.pkl", "rb") as f:
        summary = pickle.load(f)
    eval_path = run_dir / "evaluation.pkl"
    tables = None
    if eval_path.exists():
        with open(eval_path, "rb") as f:
            tables = pickle.load(f)
    return summary, tables


def growth_of_10k(returns: pd.Series) -> pd.Series:
    return 10_000.0 * (1.0 + returns).cumprod()


def main() -> None:
    st.set_page_config(page_title="Portfolio Research Dashboard", layout="wide")
    st.title("ML Portfolio Optimisation — Results")

    runs = load_runs()
    if not runs:
        st.warning(
            "No saved runs found. Run a backtest first:\n\n"
            "`python scripts/run_backtest.py --experiment rq1 --synthetic`"
        )
        return
    run_name = st.sidebar.selectbox("Backtest run", list(runs))
    summary, tables = load_run(runs[run_name])
    results = summary["results"]
    if not results:
        st.error("This run produced no successful strategies — check run.log.")
        return

    strategies = list(results)
    # Baseline: plain equal-weight if present, else the first strategy.
    default_baseline = next(
        (s for s in strategies if s.endswith("_equal") or "histmean" in s),
        strategies[0],
    )
    chosen = st.sidebar.selectbox("Strategy", strategies)
    baseline = st.sidebar.selectbox(
        "Baseline to compare against", strategies,
        index=strategies.index(default_baseline),
    )

    # ----------------------------------------------------- Executive Summary
    st.header("Executive Summary")
    st.caption(
        "If you had invested **$10,000** at the start of the test period, "
        "here is what it would have grown to (after estimated trading costs)."
    )
    chart = pd.DataFrame(
        {
            f"Your strategy ({chosen})": growth_of_10k(results[chosen].net_returns),
            f"Baseline ({baseline})": growth_of_10k(results[baseline].net_returns),
        }
    )
    st.line_chart(chart)

    if tables is not None and chosen in tables["metrics"].index:
        row = tables["metrics"].loc[chosen]
        cols = st.columns(3)
        cards = [
            ("annualized_return", "{:.1%}"),
            ("max_drawdown", "{:.1%}"),
            ("sharpe_ratio", "{:.2f}"),
            ("annualized_volatility", "{:.1%}"),
            ("calmar_ratio", "{:.2f}"),
            ("avg_monthly_turnover", "{:.1%}"),
        ]
        for i, (key, fmt) in enumerate(cards):
            if key not in row or pd.isna(row[key]):
                continue
            label, help_text = PLAIN_ENGLISH[key]
            cols[i % 3].metric(label, fmt.format(row[key]), help=help_text)

    # --------------------------------------------- Risk vs Return Landscape
    st.header("Risk vs. Return Landscape")
    st.caption(
        "Every dot is one strategy. **Up** means more return; **left** means "
        "less risk. The best strategies sit toward the **top-left**."
    )
    if tables is not None:
        scatter = tables["metrics"].reset_index()[
            ["strategy", "annualized_volatility", "annualized_return",
             "sharpe_ratio"]
        ].rename(columns={
            "annualized_volatility": "Risk (yearly volatility)",
            "annualized_return": "Return (yearly)",
            "sharpe_ratio": "Reward per unit of risk",
        })
        try:
            import plotly.express as px

            fig = px.scatter(
                scatter, x="Risk (yearly volatility)", y="Return (yearly)",
                color="Reward per unit of risk", text="strategy",
                color_continuous_scale="Viridis",
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.scatter_chart(
                scatter, x="Risk (yearly volatility)", y="Return (yearly)"
            )

    # --------------------------------------- Technical Accordion (collapsed)
    with st.expander("Advanced Quantitative Analytics", expanded=False):
        if tables is not None:
            st.subheader("Full metric table (net of transaction costs)")
            st.dataframe(tables["metrics"].round(4))
            st.subheader(
                f"Pairwise Sharpe-difference tests vs. '{tables['benchmark']}' "
                "(Jobson-Korkie-Memmel, multiple-testing corrected)"
            )
            st.dataframe(tables["tests"].round(4))
            st.subheader("Resource usage per strategy")
            res_cols = [c for c in (
                "training_time_seconds", "inference_latency_ms",
                "peak_memory_mb") if c in tables["metrics"].columns]
            st.dataframe(tables["metrics"][res_cols].round(2))
        st.subheader("Skipped combinations")
        if summary["skipped"]:
            for name, tb in summary["skipped"].items():
                st.text(f"SKIPPED: {name}")
                st.code(tb[-2000:])
        else:
            st.text("None — every combination completed.")
        st.subheader("Monthly returns (raw)")
        st.dataframe(
            pd.DataFrame({n: r.net_returns for n, r in results.items()}).round(4)
        )


if __name__ == "__main__":
    main()
