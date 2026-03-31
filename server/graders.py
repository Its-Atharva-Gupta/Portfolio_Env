"""Portfolio Management Graders and Tests"""

from typing import Dict, Any
import numpy as np
from .env import PortfolioEnv


def _score_component(agent_val: float, benchmark_val: float) -> float:
    """Score agent performance vs a benchmark. Returns 0.0–1.0"""
    if benchmark_val <= 0:
        return 1.0 if agent_val > benchmark_val else max(0.0, 1.0 + agent_val)
    return max(0.0, min(1.0, agent_val / benchmark_val))


def grade_task_1(env: PortfolioEnv) -> Dict[str, Any]:
    """
    Task 1 grader: Calm market, 63-day episode.
    Metrics: beat buy-hold (40%), drawdown (30%), beat fixed deposit (30%).
    Score: 0.0–1.0
    """
    m = env._compute_episode_score()

    bh_score = _score_component(m['agent_return'], m['bh_return'])
    dd_score = max(0.0, min(1.0, 1.0 - (m['max_drawdown'] / 0.30)))
    fd_score = _score_component(m['agent_return'], m['fd_return'])

    final_score = (
        0.40 * bh_score +
        0.30 * dd_score +
        0.30 * fd_score
    )

    return {
        'task': 1,
        'bh_score': round(bh_score, 4),
        'drawdown_score': round(dd_score, 4),
        'fd_score': round(fd_score, 4),
        'final_score': round(final_score, 4),
        'agent_return': round(m['agent_return'] * 100, 2),
        'bh_return': round(m['bh_return'] * 100, 2),
        'fd_return': round(m['fd_return'] * 100, 2),
        'max_drawdown': round(m['max_drawdown'] * 100, 2),
        'pass': final_score >= 0.3,
    }


def grade_task_2(env: PortfolioEnv) -> Dict[str, Any]:
    """
    Task 2 grader: Full market, 126-day episode.
    All five metrics: bh (25%), mom (20%), sharpe (25%), dd (15%), fd (15%).
    Score: 0.0–1.0
    """
    m = env._compute_episode_score()

    bh_score = _score_component(m['agent_return'], m['bh_return'])
    mom_score = _score_component(m['agent_return'], m['momentum_return'])
    
    if m['precomputed_sharpe'] > 0:
        sharpe_score = max(0.0, min(1.0, m['agent_sharpe'] / m['precomputed_sharpe']))
    else:
        sharpe_score = 1.0 if m['agent_sharpe'] > 0 else 0.0
    
    dd_score = max(0.0, min(1.0, 1.0 - (m['max_drawdown'] / 0.30)))
    fd_score = _score_component(m['agent_return'], m['fd_return'])

    final_score = (
        0.25 * bh_score +
        0.20 * mom_score +
        0.25 * sharpe_score +
        0.15 * dd_score +
        0.15 * fd_score
    )

    return {
        'task': 2,
        'bh_score': round(bh_score, 4),
        'momentum_score': round(mom_score, 4),
        'sharpe_score': round(sharpe_score, 4),
        'drawdown_score': round(dd_score, 4),
        'fd_score': round(fd_score, 4),
        'final_score': round(final_score, 4),
        'agent_return': round(m['agent_return'] * 100, 2),
        'bh_return': round(m['bh_return'] * 100, 2),
        'momentum_return': round(m['momentum_return'] * 100, 2),
        'fd_return': round(m['fd_return'] * 100, 2),
        'agent_sharpe': round(m['agent_sharpe'], 4),
        'max_drawdown': round(m['max_drawdown'] * 100, 2),
        'pass': final_score >= 0.5,
    }


def grade_task_3(env: PortfolioEnv) -> Dict[str, Any]:
    """
    Task 3 grader: Volatile market with shocks, 252-day episode.
    All five metrics with same weights.
    HARD CAP: if max_drawdown > 30%, final_score capped at 0.4.
    Score: 0.0–1.0
    """
    m = env._compute_episode_score()

    bh_score = _score_component(m['agent_return'], m['bh_return'])
    mom_score = _score_component(m['agent_return'], m['momentum_return'])
    
    if m['precomputed_sharpe'] > 0:
        sharpe_score = max(0.0, min(1.0, m['agent_sharpe'] / m['precomputed_sharpe']))
    else:
        sharpe_score = 1.0 if m['agent_sharpe'] > 0 else 0.0
    
    dd_score = max(0.0, min(1.0, 1.0 - (m['max_drawdown'] / 0.30)))
    fd_score = _score_component(m['agent_return'], m['fd_return'])

    final_score = (
        0.25 * bh_score +
        0.20 * mom_score +
        0.25 * sharpe_score +
        0.15 * dd_score +
        0.15 * fd_score
    )

    # Hard cap: catastrophic drawdown disqualifies regardless of returns
    if m['max_drawdown'] > 0.30:
        final_score = min(final_score, 0.4)

    return {
        'task': 3,
        'bh_score': round(bh_score, 4),
        'momentum_score': round(mom_score, 4),
        'sharpe_score': round(sharpe_score, 4),
        'drawdown_score': round(dd_score, 4),
        'fd_score': round(fd_score, 4),
        'final_score': round(final_score, 4),
        'drawdown_capped': m['max_drawdown'] > 0.30,
        'agent_return': round(m['agent_return'] * 100, 2),
        'bh_return': round(m['bh_return'] * 100, 2),
        'momentum_return': round(m['momentum_return'] * 100, 2),
        'fd_return': round(m['fd_return'] * 100, 2),
        'agent_sharpe': round(m['agent_sharpe'], 4),
        'historical_sharpe': round(m['precomputed_sharpe'], 4),
        'max_drawdown': round(m['max_drawdown'] * 100, 2),
        'pass': final_score >= 0.6,
    }


def grade_episode(env: PortfolioEnv) -> Dict[str, Any]:
    """Route to correct grader based on task_level"""
    graders = {1: grade_task_1, 2: grade_task_2, 3: grade_task_3}
    return graders[env.task_level](env)


# ============================================================================
# Layer 1: Correctness Tests
# ============================================================================

def test_buy_action():
    """Test buy action"""
    env = PortfolioEnv(task_level=1, seed=42)
    env.reset()
    initial_cash = env.cash
    price = env.df.loc[env.current_idx, 'Close']
    env.step(1)
    # Expected shares accounting for transaction cost
    expected_shares = int(initial_cash / (price * (1 + env.transaction_cost)))
    expected_cash = initial_cash - expected_shares * price * (1 + env.transaction_cost)
    assert env.shares == expected_shares, f"Expected {expected_shares} shares, got {env.shares}"
    assert abs(env.cash - expected_cash) < 1.0, f"Expected cash {expected_cash}, got {env.cash}"
    print("✓ test_buy_action passed")


def test_sell_action():
    """Test sell action"""
    env = PortfolioEnv(task_level=1, seed=42)
    env.reset()
    env.step(1)
    shares_before = env.shares
    assert shares_before > 0, "No shares bought"
    env.step(2)
    assert env.shares == 0, f"Expected 0 shares after sell, got {env.shares}"
    assert env.cash > 0, f"Expected cash > 0 after sell, got {env.cash}"
    print("✓ test_sell_action passed")


def test_hold_action():
    """Test hold action"""
    env = PortfolioEnv(task_level=1, seed=42)
    env.reset()
    env.step(1)
    cash_before = env.cash
    shares_before = env.shares
    env.step(0)
    assert env.cash == cash_before, f"Cash changed on hold: {cash_before} -> {env.cash}"
    assert env.shares == shares_before, f"Shares changed on hold: {shares_before} -> {env.shares}"
    print("✓ test_hold_action passed")


def test_episode_terminates():
    """Test episode terminates correctly"""
    env = PortfolioEnv(task_level=1, seed=42)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(0)
        steps += 1
        if steps > env.window_size + 10:  # Safety check
            raise RuntimeError("Episode did not terminate")
    assert steps == env.window_size, f"Expected {env.window_size} steps, got {steps}"
    print("✓ test_episode_terminates passed")


def test_no_lookahead():
    """Test no lookahead bias"""
    env = PortfolioEnv(task_level=1, seed=42)
    obs, _ = env.reset()
    assert obs['close_prices'][-1] == env.df.loc[env.current_idx, 'Close'], \
        "Current price does not match observation"
    print("✓ test_no_lookahead passed")


def test_rsi_range():
    """Test RSI is in valid range"""
    env = PortfolioEnv(task_level=1, seed=42)
    env.reset()
    rsi = env._compute_rsi()
    assert 0.0 <= rsi <= 100.0, f"RSI out of range: {rsi}"
    print(f"✓ test_rsi_range passed (RSI={rsi:.1f})")


def test_rolling_benchmark_no_lookahead():
    """Test rolling benchmark has no lookahead"""
    env = PortfolioEnv(task_level=1, seed=42)
    env.reset()
    bm = env._get_rolling_benchmark()
    start = max(0, env.current_idx - 50)
    end = env.current_idx  # exclusive
    manual = float(env.precomputed_daily_returns[start:end].mean())
    assert abs(bm - manual) < 1e-10, f"Benchmark mismatch: {bm} vs {manual}"
    print("✓ test_rolling_benchmark_no_lookahead passed")


def test_grader_scores_in_range():
    """Test grader scores are in [0, 1]"""
    for task in [1, 2, 3]:
        env = PortfolioEnv(task_level=task, seed=42)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(0)
        result = grade_episode(env)
        assert 0.0 <= result['final_score'] <= 1.0, \
            f"Task {task}: score {result['final_score']} out of range"
        print(f"✓ test_grader_scores_in_range passed for task {task} (score={result['final_score']:.4f})")


def run_all_correctness_tests():
    """Run all correctness tests"""
    test_buy_action()
    test_sell_action()
    test_hold_action()
    test_episode_terminates()
    test_no_lookahead()
    test_rsi_range()
    test_rolling_benchmark_no_lookahead()
    test_grader_scores_in_range()
    print("\n✓✓✓ All Layer 1 correctness tests passed ✓✓✓")


if __name__ == "__main__":
    run_all_correctness_tests()
