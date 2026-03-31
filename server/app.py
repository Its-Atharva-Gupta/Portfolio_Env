"""FastAPI Server for Portfolio Management Environment"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import PortfolioAction, PortfolioObservation
    from graders import grade_episode
    from .portfolio_environment import PortfolioEnvironment
except ImportError:
    from models import PortfolioAction, PortfolioObservation
    from .graders import grade_episode
    from .portfolio_environment import PortfolioEnvironment


# ── OpenEnv standard app ──────────────────────────────────────
# Automatically registers: POST /reset, POST /step, GET /state,
# GET /schema, GET /health, GET /metadata, WS /ws, /mcp
app = create_app(
    PortfolioEnvironment,
    PortfolioAction,
    PortfolioObservation,
    env_name="portfolio",
    max_concurrent_envs=1,
)

# ── Separate env instances for custom endpoints ───────────────
# create_app manages its own instances internally.
# We maintain our own for /grader and /baseline.
_envs = {
    1: PortfolioEnvironment(task_level=1),
    2: PortfolioEnvironment(task_level=2),
    3: PortfolioEnvironment(task_level=3),
}
_current_task = 1


# ── Custom endpoints ──────────────────────────────────────────
# Only define what OpenEnv does NOT already provide.
# DO NOT redefine: /reset /step /state /schema /health /metadata /ws /mcp

@app.get("/grader")
async def get_grader():
    """Get grader score for the current episode"""
    score = grade_episode(_envs[_current_task].game)
    return score


@app.get("/tasks")
async def get_tasks():
    """Get task definitions and action schema"""
    return {
        "tasks": [
            {
                "id": 1,
                "name": "Calm Market",
                "description": "63-day episode in low-volatility window. Graded on buy-hold, drawdown, fixed deposit.",
                "difficulty": "easy",
                "episode_length": 63,
                "success_threshold": 0.3,
                "metrics": ["beat_buy_hold", "max_drawdown", "beat_fixed_deposit"]
            },
            {
                "id": 2,
                "name": "Full Market",
                "description": "126-day episode from any market period. All five metrics.",
                "difficulty": "medium",
                "episode_length": 126,
                "success_threshold": 0.5,
                "metrics": ["beat_buy_hold", "beat_momentum", "sharpe_ratio", "max_drawdown", "beat_fixed_deposit"]
            },
            {
                "id": 3,
                "name": "Volatile Market with Shocks",
                "description": "252-day episode with at least one 3%+ price move. All five metrics. Drawdown > 30% caps score at 0.4.",
                "difficulty": "hard",
                "episode_length": 252,
                "success_threshold": 0.6,
                "metrics": ["beat_buy_hold", "beat_momentum", "sharpe_ratio", "max_drawdown", "beat_fixed_deposit"]
            }
        ],
        "action_schema": {
            "type": "integer",
            "enum": [0, 1, 2],
            "descriptions": {
                "0": "HOLD - do nothing",
                "1": "BUY - buy maximum shares with available cash",
                "2": "SELL - sell all held shares"
            }
        }
    }


@app.post("/baseline")
async def baseline():
    """
    Baseline evaluation: RSI rule-based agent across all 3 tasks.
    Deterministic and reproducible — no API key required.
    Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought), else hold.
    """
    global _current_task
    results = {}

    for task_level in [1, 2, 3]:
        _current_task = task_level
        env = _envs[task_level]
        obs, _ = env.game.reset()
        done = False

        while not done:
            rsi = obs.get('rsi', 50.0)
            if rsi < 30:
                action = 1   # oversold → buy
            elif rsi > 70:
                action = 2   # overbought → sell
            else:
                action = 0   # neutral → hold

            obs, _, done, _ = env.game.step(action)

        score = grade_episode(env.game)
        results[f"task_{task_level}"] = {
            "final_score":  score["final_score"],
            "agent_return": score["agent_return"],
            "pass":         score["pass"],
        }

    return {
        "baseline_agent": "rsi_rule_based",
        "description":    "RSI rule-based agent: buy when RSI<30, sell when RSI>70, else hold.",
        "scores":         results,
    }


# ── Entry point ───────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for uv run server"""
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()