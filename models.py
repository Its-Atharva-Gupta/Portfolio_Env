"""Models for Portfolio Management Environment"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    # Fallback if openenv not installed
    Action = BaseModel
    Observation = BaseModel


class PortfolioAction(Action):
    """Action taken by the agent"""
    action: int = Field(
        ...,
        ge=0,
        le=2,
        description="0=hold, 1=buy all with available cash, 2=sell all held shares"
    )


class PortfolioObservation(Observation):
    """Observation returned to agent"""
    close_prices: List[float] = Field(..., description="Last 10 closing prices")
    cash: float = Field(..., description="Cash available in portfolio")
    shares: int = Field(..., description="Number of shares currently held")
    portfolio_value: float = Field(..., description="Total portfolio value")
    day_in_episode: int = Field(..., description="Current day number in episode")
    total_days: int = Field(..., description="Total days in this episode")
    rolling_benchmark: float = Field(..., description="50-day rolling benchmark return")
    episode_return_pct: float = Field(..., description="Agent return percentage")
    benchmark_return_pct: float = Field(..., description="Buy-and-hold return percentage")
    rsi: float = Field(..., description="RSI(14) indicator")
    macd: float = Field(..., description="MACD line value")
    macd_signal: float = Field(..., description="MACD signal line")
    macd_hist: float = Field(..., description="MACD histogram")
    macd_cross: int = Field(..., description="MACD crossover: -1=bearish, 0=none, 1=bullish")
    sma50: float = Field(..., description="50-day simple moving average")
    sma200: float = Field(..., description="200-day simple moving average")
    golden_cross: bool = Field(..., description="Golden cross active")
    cross_signal: int = Field(..., description="Cross signal: -1=death, 0=none, 1=golden")
    reward: float = Field(default=0.0, description="Reward from last action")
    done: bool = Field(default=False, description="Episode termination flag")
    text_observation: str = Field(..., description="Text observation for LLM agents")


class PortfolioState(BaseModel):
    """Complete internal state of the environment"""
    task_level: int = Field(..., description="Task level 1-3")
    current_idx: int = Field(..., description="Current index in data")
    start_idx: int = Field(..., description="Episode start index")
    cash: float = Field(..., description="Cash available")
    shares: int = Field(..., description="Shares held")
    portfolio_value: float = Field(..., description="Total portfolio value")
    portfolio_values: List[float] = Field(..., description="History of portfolio values")
    daily_returns: List[float] = Field(..., description="History of daily returns")
    episode_prices: List[float] = Field(..., description="Prices in this episode")
    day_in_episode: int = Field(..., description="Current day in episode")
    window_size: int = Field(..., description="Episode window size")
    indicators: Dict[str, Any] = Field(..., description="Current indicator values")
    score: Optional[Dict[str, Any]] = Field(default=None, description="Grader score dict")
