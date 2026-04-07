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

STDOUT FORMAT
- The script emits exactly three line types to stdout:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path

import requests
from openai import OpenAI
from dotenv import load_dotenv
from client import PortfolioEnvClient
from models import PortfolioAction

# ── Load environment variables from .env ──────────────────────
load_dotenv()

# ── Environment variables ─────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default — must be provided

# Environment server URL — the already-running HF Space or local server
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

# If you are using docker image
# Support both naming conventions: IMAGE_NAME (sample.py) and LOCAL_IMAGE_NAME (final.md)
IMAGE_NAME = os.getenv("IMAGE_NAME")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

FALLBACK_ACTION = 0  # hold — safe default if LLM fails
TEMPERATURE = 0.0
MAX_TOKENS = 5
BENCHMARK_NAME = "portfolio"

SYSTEM_PROMPT = """You are an expert stock trader managing a Reliance Industries (RELIANCE.NS) portfolio.
You receive daily market observations including price history, RSI, MACD, golden/death cross signals, and portfolio status.
Goal: maximize risk-adjusted returns, beat buy-and-hold, keep drawdown below 30%.
Respond with ONLY a single integer: 0 (hold), 1 (buy all), or 2 (sell all). Nothing else."""

TASK_NAMES = {1: "calm-market", 2: "full-market", 3: "volatile-market"}


# ── Structured logging ────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Helpers ───────────────────────────────────────────────────

def parse_action(response_text: str) -> int:
    """Parse LLM response into valid action integer."""
    if not response_text:
        return FALLBACK_ACTION
    try:
        action = int(response_text.strip())
        if action not in (0, 1, 2):
            return FALLBACK_ACTION
        return action
    except (ValueError, TypeError):
        return FALLBACK_ACTION


def get_rsi_fallback_action(text_observation: str) -> int:
    """Rule-based fallback using RSI signals in text observation."""
    if "OVERSOLD" in text_observation:
        return 1
    elif "OVERBOUGHT" in text_observation:
        return 2
    return 0


ACTION_LABELS = {0: "hold", 1: "buy", 2: "sell"}


def get_action_label(action: int) -> str:
    """Safely get action label, default to 'unknown' if not found."""
    return ACTION_LABELS.get(action, f"unknown({action})")


def get_env_client():
    """Connect to the environment server. Returns a sync client.
    Tries ENV_URL first, then Docker image if available."""
    # Try connecting to already-running server
    try:
        resp = requests.get(f"{ENV_URL}/health", timeout=5)
        if resp.status_code == 200:
            env = PortfolioEnvClient(base_url=ENV_URL)
            return env.sync()
    except Exception:
        pass

    # Fallback: try to launch from Docker image (if IMAGE_NAME or LOCAL_IMAGE_NAME env var is set)
    docker_image = IMAGE_NAME or LOCAL_IMAGE_NAME
    if docker_image:
        try:
            env = PortfolioEnvClient.from_docker_image(docker_image)
            return env.sync()
        except Exception as exc:
            error_msg = f"Failed to launch Docker image '{docker_image}': {exc}"
            print(f"[ERROR] {error_msg}", flush=True)
            raise RuntimeError(error_msg) from exc

    # Both methods failed
    error_msg = f"Failed to connect to environment server at {ENV_URL} and Docker image not available"
    print(f"[ERROR] {error_msg}", flush=True)
    raise RuntimeError(error_msg)


def fetch_grader_score(env_url: str) -> Dict[str, Any]:
    """Fetch grader score via HTTP (grader is a custom endpoint, not on WebSocket)."""
    try:
        resp = requests.get(f"{env_url}/grader", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"final_score": 0.0, "pass": False}


# ── Run one task ──────────────────────────────────────────────

def run_task(task_level: int, llm_client: OpenAI, env: PortfolioEnvClient) -> Dict[str, Any]:
    """Run one task, emit structured logs, return grader score."""
    # Safe lookup of task name
    task_name = TASK_NAMES.get(task_level, f"unknown-task-{task_level}")
    thresholds = {1: 0.3, 2: 0.5, 3: 0.6}
    task_threshold = thresholds.get(task_level, 0.5)  # Safe threshold lookup

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        # Reset environment for this task via WebSocket
        try:
            result = env.reset(task_level=task_level)
        except Exception as exc:
            error_msg = f"Failed to reset environment: {exc}"
            print(f"[ERROR] {error_msg}", flush=True)
            log_step(step=1, action="hold", reward=0.0, done=True, error=error_msg)
            steps_taken = 1
            rewards.append(0.0)
            raise

        # Extract observation safely
        try:
            observation = result.observation
            text_obs = observation.text_observation
            done = result.done
        except (AttributeError, KeyError, TypeError) as exc:
            error_msg = f"Malformed observation from environment: {exc}"
            print(f"[ERROR] {error_msg}", flush=True)
            log_step(step=1, action="hold", reward=0.0, done=True, error=error_msg)
            steps_taken = 1
            rewards.append(0.0)
            raise RuntimeError(error_msg) from exc

        # Build conversation history for LLM
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text_obs},
            ]
        except Exception as exc:
            error_msg = f"Failed to build initial messages: {exc}"
            print(f"[ERROR] {error_msg}", flush=True)
            raise RuntimeError(error_msg) from exc

        while not done:
            steps_taken += 1
            error_msg = None

            # Call LLM for action
            try:
                completion = llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = (completion.choices[0].message.content or "").strip()
                action = parse_action(response_text)
            except Exception as exc:
                error_msg = str(exc)
                action = get_rsi_fallback_action(text_obs)

            # Step environment via WebSocket
            try:
                result = env.step(PortfolioAction(action=action))
            except Exception as exc:
                error_msg = str(exc)
                log_step(
                    step=steps_taken,
                    action=get_action_label(action),
                    reward=0.0,
                    done=True,
                    error=error_msg,
                )
                rewards.append(0.0)
                break

            # Extract result fields safely
            try:
                observation = result.observation
                reward = float(result.reward or 0.0)
                done = result.done
                text_obs = observation.text_observation
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                error_msg = f"Malformed step result: {exc}"
                log_step(
                    step=steps_taken,
                    action=get_action_label(action),
                    reward=0.0,
                    done=True,
                    error=error_msg,
                )
                rewards.append(0.0)
                break

            rewards.append(reward)

            log_step(
                step=steps_taken,
                action=get_action_label(action),
                reward=reward,
                done=done,
                error=error_msg,
            )

            # Update conversation for next LLM call
            if not done:
                try:
                    messages.append({"role": "assistant", "content": str(action)})
                    messages.append({
                        "role": "user",
                        "content": f"Result: reward={reward:.2f}\n\n{text_obs}",
                    })
                except Exception as exc:
                    error_msg = f"Failed to build next message: {exc}"
                    print(f"[ERROR] {error_msg}", flush=True)
                    log_step(
                        step=steps_taken + 1,
                        action="hold",
                        reward=0.0,
                        done=True,
                        error=error_msg,
                    )
                    rewards.append(0.0)
                    break

        # Fetch grader score via HTTP endpoint
        try:
            grader = fetch_grader_score(ENV_URL)
            score = float(grader.get("final_score", 0.0))
            success = score >= task_threshold
        except (KeyError, TypeError, ValueError) as exc:
            error_msg = f"Failed to parse grader score: {exc}"
            print(f"[ERROR] {error_msg}", flush=True)
            score = 0.0
            success = False

    except Exception as exc:
        # Catch-all: ensure [END] is always emitted
        error_msg = str(exc)
        if steps_taken == 0:
            log_step(step=1, action="hold", reward=0.0, done=True, error=error_msg)
            steps_taken = 1
            rewards.append(0.0)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_level,
        "task_name": task_name,
        "final_score": score,
        "success": success,
        "steps": steps_taken,
    }


# ── Main ──────────────────────────────────────────────────────

async def main() -> None:
    """Run all tasks, emit structured logs, and handle errors."""
    try:
        # Initialize OpenAI client
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")

        # Connect to the environment server (WebSocket for stateful interaction)
        env = get_env_client()

        results = []
        try:
            for task_level in [1, 2, 3]:
                result = run_task(task_level, llm_client, env)
                results.append(result)
        finally:
            try:
                env.close()
            except Exception:
                pass

    except Exception as exc:
        print(f"[FATAL] {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
