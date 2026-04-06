"""
inference.py — Portfolio Management RL Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- This script must be named inference.py and placed in the root directory
- Uses OpenAI Client for all LLM calls
"""

import asyncio
import os
import sys
from typing import List, Dict, Any
from openai import OpenAI

# ── Environment variables ─────────────────────────────────────
API_BASE_URL    = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY         = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME      = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
FALLBACK_ACTION = 0        # hold — safe default if LLM fails
TEMPERATURE     = 0.0

MAX_TOKENS      = 5

SYSTEM_PROMPT = """You are an expert stock trader managing a Reliance Industries (RELIANCE.NS) portfolio.
You receive daily market observations including price history, RSI, MACD, golden/death cross signals, and portfolio status.
Goal: maximize risk-adjusted returns, beat buy-and-hold, keep drawdown below 30%.
Respond with ONLY a single integer: 0 (hold), 1 (buy all), or 2 (sell all). Nothing else."""


def parse_action(response_text: str) -> int:
    """Parse LLM response into valid action integer. Returns fallback on failure."""
    if not response_text:
        return FALLBACK_ACTION
    try:
        action = int(response_text.strip())
        if action not in [0, 1, 2]:
            return FALLBACK_ACTION
        return action
    except (ValueError, TypeError):
        return FALLBACK_ACTION


def get_rsi_fallback_action(text_observation: str) -> int:
    """Rule-based fallback using RSI signals in text observation."""
    if "OVERSOLD" in text_observation:
        return 1   # buy
    elif "OVERBOUGHT" in text_observation:
        return 2   # sell
    return 0       # hold


async def run_task(task_level: int, client: OpenAI) -> Dict[str, Any]:
    """Run one task and return grader score."""
    from client import PortfolioEnvClient
    from models import PortfolioAction
    from server.graders import grade_episode

    print(f"\n{'='*60}")
    print(f"TASK {task_level} — "
          f"{'Calm' if task_level==1 else 'Full' if task_level==2 else 'Volatile'} Market")
    print(f"{'='*60}")

    # Start container and connect — same pattern as BrowserGymEnv.from_docker_image
    env = await PortfolioEnvClient.from_docker_image(
        image="portfolio-env:latest",
    )
    

    history:      List[str]            = []
    actions_taken: Dict[int, int]      = {0: 0, 1: 0, 2: 0}

    try:
        result      = await env.reset()
        observation = result.observation
        done        = result.done

        # conversation history — LLM adapts within episode based on past actions
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": observation.text_observation},
        ]

        step = 0
        while not done:
            step += 1

            # ── Call LLM ─────────────────────────────────────
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
                action = parse_action(response_text)

            except Exception as exc:
                print(f"  Model request failed ({exc}). Using RSI fallback.")
                action = get_rsi_fallback_action(observation.text_observation)

            actions_taken[action] += 1

            # ── Add to conversation history ───────────────────
            messages.append({"role": "assistant", "content": str(action)})

            # ── Step environment ──────────────────────────────
            result      = await env.step(PortfolioAction(action=action))
            observation = result.observation
            done        = result.done

            # ── Add result to history ─────────────────────────
            if not done:
                history_line = (
                    f"Step {step}: action={action} "
                    f"reward={result.reward:+.2f} "
                    f"pv=₹{observation.portfolio_value:,.0f}"
                )
                history.append(history_line)
                messages.append({
                    "role": "user",
                    "content": f"Result: reward={result.reward:.2f}\n\n{observation.text_observation}"
                })

            if step % 10 == 0 or done:
                action_str = ['HOLD', 'BUY ', 'SELL'][action]
                print(f"  Day {observation.day_in_episode:3d} | {action_str} | "
                      f"PV: ₹{observation.portfolio_value:>12,.2f} | "
                      f"Return: {observation.episode_return_pct:>+7.2f}% | "
                      f"RSI: {observation.rsi:.1f}")

            if done:
                print("Episode complete.")
                break

        # ── Grade episode ─────────────────────────────────────
        # fetch grader from server
        async def fetch_grader():
            # TODO: The /grader endpoint is currently broken. Return dummy score for now.
            # Return dummy score (grader endpoint needs debugging)
            return {
                "final_score": 0.0,
                "agent_return": observation.episode_return_pct / 100.0 if observation else 0.0,
                "bh_return": 0.0,
                "max_drawdown": 0.0,
                "pass": False,
            }

        grader    = await fetch_grader()
        threshold = [0.3, 0.5, 0.6][task_level - 1]
        status    = "✓ PASS" if grader['final_score'] >= threshold else "✗ FAIL"

        print(f"\n── TASK {task_level} RESULTS ─────────────────────────────────")
        print(f"  Agent Return:   {grader['agent_return']:>+7.2f}%")
        print(f"  Buy-Hold:       {grader['bh_return']:>+7.2f}%")
        print(f"  Max Drawdown:   {grader['max_drawdown']:>7.2f}%")
        if 'agent_sharpe' in grader:
            print(f"  Sharpe:         {grader['agent_sharpe']:>7.4f}")
        print(f"── FINAL SCORE: {grader['final_score']:.4f}  {status} ──")
        print(f"  Actions: Hold={actions_taken[0]}, Buy={actions_taken[1]}, Sell={actions_taken[2]}")

        return grader

    finally:
        try:
            await env.close()
        except Exception as e:
            # Docker container stop might timeout, but episode already completed
            print(f"  Warning: failed to close environment: {e}")


async def main() -> None:
    print("Portfolio Management RL Environment — Inference")
    print(f"Model:  {MODEL_NAME}")
    print(f"API:    {API_BASE_URL}")
    print(f"Token:  {'set' if API_KEY else 'NOT SET — RSI fallback will be used'}")

    # Initialize OpenAI client pointed at HuggingFace router
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "no-key")

    scores = {}
    for task in [1, 2, 3]:
        scores[f"task_{task}"] = await run_task(task, client)

    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    thresholds = [0.3, 0.5, 0.6]
    for i, (task, result) in enumerate(scores.items()):
        status = "✓ PASS" if result['final_score'] >= thresholds[i] else "✗ FAIL"
        print(f"  {task}: {result['final_score']:.4f}  {status}")


if __name__ == "__main__":
    asyncio.run(main())