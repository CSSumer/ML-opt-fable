# Beginner's Guide — No Maths Required

This project answers a simple question: **can modern machine-learning models
build a better long-term stock portfolio than the classic textbook methods?**

## The idea in one paragraph

Every month, each "strategy" looks at the recent behaviour of S&P 500 stocks
and the economy, guesses which stocks will do well next month, and divides a
pot of money between them. We then *only* look at what actually happened the
following month to score it. We repeat this month after month for years —
always making decisions with information that was truly available at the
time, never with hindsight — and compare how each strategy would have grown
an investment of $10,000.

## What's being compared?

- **Naive baselines** — "guess zero" and "assume the past average continues".
  If a fancy model can't beat these, it isn't earning its complexity.
- **Classic finance models** — linear regression and a Fama-French-style
  factor model, the workhorses of academic finance.
- **Machine learning** — XGBoost (decision trees) and an LSTM neural network
  (reads sequences of months like a sentence).
- **End-to-end** — a neural network that skips the "predict first" step and
  directly decides portfolio weights.

## How to run it (three commands)

```bash
pip install -r requirements.txt
python scripts/run_backtest.py --experiment rq1 --synthetic   # quick demo, no internet needed
streamlit run scripts/dashboard.py                            # opens the dashboard in your browser
```

(For real market data, see the Quick Start in README.md — you'll need a free
data key.)

## Reading the dashboard

**Executive Summary** — the big chart shows what $10,000 invested in the
strategy would have become, next to a simple baseline. Higher is better;
smoother is calmer. The scorecard translates the jargon:

| You'll see | It means |
|---|---|
| Average Yearly Growth | How fast the money grew per year |
| Worst-Case Historical Drop | The most you'd have been down at the worst moment |
| Bumpiness of the Ride | How wildly the value swings month to month |
| Reward per Unit of Risk | Growth earned for each unit of risk taken — the headline score |

**Risk vs. Return Landscape** — every dot is a strategy. The best ones sit
**top-left**: more growth, less turbulence.

**Advanced Quantitative Analytics** — a collapsed section with the raw
statistics for technical readers. You can ignore it entirely.

## Honest caveats

- The stock list is *today's* S&P 500, so companies that failed along the way
  are missing. That flatters everyone equally, so comparisons between
  strategies are still fair — but don't read the dollar figures as a promise.
- Past performance, simulated or real, never guarantees future results.
- This is research code for a thesis, not investment advice.
