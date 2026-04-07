---
title: Portfolio Management RL Environment
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Portfolio Management RL Environment

An OpenEnv-compatible reinforcement learning environment where AI agents manage a stock portfolio by trading **Reliance Industries (RELIANCE.NS)** using 10 years of historical OHLCV data and professional technical indicators.

## Motivation

Portfolio management is a genuine real-world task that fund managers, traders, and robo-advisors perform daily. This environment models the core decision loop: observe market conditions, interpret technical signals, and decide whether to buy, hold, or sell — all while managing risk and transaction costs. It provides a challenging, realistic testbed for evaluating LLM and RL agents on sequential financial decision-making.

## Action Space

| Action | Description |
|--------|-------------|
| `0` | **HOLD** — Do nothing |
| `1` | **BUY** — Buy maximum shares with all available cash (0.1% transaction cost) |
| `2` | **SELL** — Sell all held shares for cash (0.1% transaction cost) |

Type: Discrete (3 actions). Defined as `PortfolioAction` Pydantic model with `action: int` field constrained to `[0, 1, 2]`.

## Observation Space

Each observation is a `PortfolioObservation` Pydantic model containing:

| Field | Type | Description |
|-------|------|-------------|
| `close_prices` | `List[float]` | Last 10 closing prices |
| `cash` | `float` | Cash available in portfolio |
| `shares` | `int` | Number of shares held |
| `portfolio_value` | `float` | Total portfolio value (cash + shares) |
| `day_in_episode` | `int` | Current day in episode |
| `total_days` | `int` | Total days in this episode |
| `rolling_benchmark` | `float` | 50-day rolling benchmark return |
| `episode_return_pct` | `float` | Agent return % since episode start |
| `benchmark_return_pct` | `float` | Buy-and-hold return % |
| `rsi` | `float` | RSI(14) indicator |
| `macd` | `float` | MACD line value |
| `macd_signal` | `float` | MACD signal line |
| `macd_hist` | `float` | MACD histogram |
| `macd_cross` | `int` | MACD crossover: -1=bearish, 0=none, 1=bullish |
| `sma50` | `float` | 50-day simple moving average |
| `sma200` | `float` | 200-day simple moving average |
| `golden_cross` | `bool` | Whether golden cross is active |
| `cross_signal` | `int` | Cross signal: -1=death, 0=none, 1=golden |
| `reward` | `float` | Reward from last action |
| `done` | `bool` | Episode termination flag |
| `text_observation` | `str` | Human-readable observation for LLM agents |

## Tasks

### Task 1: Calm Market (Easy)

- **Episode length:** 63 days
- **Window selection:** Low-volatility periods (std < 75th percentile)
- **Metrics:** Beat buy-hold (40%), max drawdown (30%), beat fixed deposit (30%)
- **Pass threshold:** 0.3

### Task 2: Full Market (Medium)

- **Episode length:** 126 days
- **Window selection:** Any valid market period
- **Metrics:** Beat buy-hold (25%), beat momentum (20%), Sharpe ratio (25%), max drawdown (15%), beat fixed deposit (15%)
- **Pass threshold:** 0.5

### Task 3: Volatile Market with Shocks (Hard)

- **Episode length:** 252 days
- **Window selection:** Must contain at least one 3%+ price move
- **Metrics:** Same as Task 2, plus hard cap at score 0.4 if drawdown exceeds 30%
- **Pass threshold:** 0.6

## Reward Function

The reward function provides **per-step signal** (not just end-of-episode):

- **Base:** Daily portfolio return minus risk-free rate (opportunity cost)
- **Bonus:** +5.0 for beating the 50-day rolling benchmark
- **Penalty:** -5.0 if cash goes negative; -10.0 if portfolio drops below 50% of initial value

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- Docker (for containerized deployment)

### Local Development

```bash
# Install dependencies
uv sync

# Run the server
uv run uvicorn portfolio.server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t portfolio-env .
docker run -p 8000:8000 portfolio-env
```

### Running Inference

```bash
# Set required environment variables
export HF_TOKEN="your-hf-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export ENV_URL="http://localhost:8000"

# Run the inference script
python inference.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment, start new episode |
| `/step` | POST | Execute action, get observation |
| `/state` | GET | Get current internal state |
| `/grader` | GET | Get grader score for current episode |
| `/tasks` | GET | Get task definitions and action schema |
| `/baseline` | POST | Run RSI rule-based baseline on all tasks |
| `/health` | GET | Health check |
| `/schema` | GET | OpenEnv schema |
| `/metadata` | GET | Environment metadata |

## Baseline Scores

RSI rule-based agent (buy when RSI < 30, sell when RSI > 70, hold otherwise):

| Task | Score | Status |
|------|-------|--------|
| Task 1 (Calm) | ~0.35 | PASS |
| Task 2 (Full) | ~0.30 | FAIL |
| Task 3 (Volatile) | ~0.25 | FAIL |

The baseline intentionally passes only Task 1, demonstrating meaningful difficulty progression.

## Project Structure

```
.
├── inference.py              # Mandatory inference script (root)
├── openenv.yaml              # OpenEnv metadata
├── Dockerfile                # Container build
├── pyproject.toml            # Dependencies
├── models.py                 # Pydantic Action/Observation models
├── client.py                 # OpenEnv client wrapper
├── data/
│   └── reliance.csv          # 10 years of Reliance Industries OHLCV data
└── server/
    ├── app.py                # FastAPI server with OpenEnv endpoints
    ├── env.py                # Core RL environment implementation
    ├── portfolio_environment.py  # OpenEnv wrapper
    └── graders.py            # Task graders + correctness tests
```
