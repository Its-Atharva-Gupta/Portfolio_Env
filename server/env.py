"""Portfolio Management RL Environment Core Implementation"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any, Optional

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # goes to Trading_Env



class PortfolioEnv:
    """
    Core portfolio management environment.
    
    Agent trades Reliance Industries stock using 10 years of historical data
    with technical indicators (RSI, MACD, Golden/Death Cross).
    Three task levels with separate graders.
    """

    def __init__(self, task_level: int = 1, seed: int = 42):
        """
        Initialize the environment.
        
        Args:
            task_level: 1 (easy), 2 (medium), or 3 (hard)
            seed: Random seed for reproducibility
        """
        assert task_level in [1, 2, 3], "task_level must be 1, 2, or 3"
        
        self.task_level = task_level
        self.rng = np.random.default_rng(seed)
        
        # Constants
        self.initial_cash = 100000.0
        self.risk_free_annual = 0.065
        self.risk_free_daily = self.risk_free_annual / 252.0
        self.transaction_cost = 0.001  # 0.1% transaction cost
        
        # Load data FIRST
        self.df = self._load_data()
        
        # Window sizes and constraints (after loading data)
        self.window_size = {1: 63, 2: 126, 3: 252}[task_level]
        self.min_start_idx = min(200, max(50, len(self.df) // 10))  # Need 200 days for SMA200, but adapt to dataset size
        
        # Precompute metrics on full dataset
        self.precomputed_daily_returns = self.df['Close'].pct_change().dropna().values
        self.precomputed_sharpe = self._compute_sharpe(self.precomputed_daily_returns)
        
        # Pre-filter valid starts based on task
        self.valid_starts = self._compute_valid_starts()
        
        # Episode state
        self.start_idx: int = 0
        self.current_idx: int = 0
        self.cash: float = 0.0
        self.shares: int = 0
        self.portfolio_values: List[float] = []
        self.daily_returns: List[float] = []
        self.episode_prices: List[float] = []

    def _load_data(self) -> pd.DataFrame:
        """Load and clean CSV data"""
        # Try different possible paths
        possible_paths = [
                Path("/app/env/data/reliance.csv"),
            ]
        
        df_path = None
        for path in possible_paths:
            if path.exists():
                df_path = path
                break
        
        if df_path is None:
            raise FileNotFoundError(f"Could not find reliance.csv in any of {possible_paths}")
        
        # Load with skiprows if needed
        df = pd.read_csv(df_path)
        
        # Handle yfinance two-row header
        if "Adj Close" in df.columns or df.shape[1] > 6:
            df = pd.read_csv(df_path, skiprows=2)
        
        # Ensure correct columns
        if len(df.columns) > 6:
            df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close']
            df = df.drop('Adj Close', axis=1)
        else:
            df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        
        # Clean
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df = df.dropna()
        df = df.reset_index(drop=True)
        
        return df

    def _compute_valid_starts(self) -> np.ndarray:
        """Compute valid window start indices based on task level"""
        max_start = len(self.df) - self.window_size
        
        # Safety: ensure we have enough data
        if max_start < self.min_start_idx:
            # Not enough data - return single start point
            return np.array([max(0, max_start - 10)])
        
        if self.task_level == 1:
            # Task 1: low-volatility windows
            # Filter: 63-day std < 75th percentile of all 63-day stds
            stds = []
            for i in range(self.min_start_idx, max_start + 1):
                window_returns = self.precomputed_daily_returns[i:i + 63]
                if len(window_returns) >= 63:
                    stds.append(np.std(window_returns, ddof=0))
            
            if len(stds) > 0:
                threshold = np.percentile(stds, 75)
                valid = []
                for i in range(self.min_start_idx, max_start + 1):
                    window_returns = self.precomputed_daily_returns[i:i + 63]
                    if len(window_returns) >= 63:
                        if np.std(window_returns, ddof=0) < threshold:
                            valid.append(i)
                if valid:
                    return np.array(valid)
            # Fallback
            return np.arange(max(0, self.min_start_idx - 10), max_start + 1) if max_start > 0 else np.array([0])
        
        elif self.task_level == 2:
            # Task 2: any valid start
            valid_range = np.arange(self.min_start_idx, max_start + 1)
            return valid_range if len(valid_range) > 0 else np.array([0])
        
        else:  # task_level == 3
            # Task 3: must have at least one day with |return| > 3%
            valid = []
            for i in range(self.min_start_idx, max_start + 1):
                window_returns = self.precomputed_daily_returns[i:i + self.window_size]
                if len(window_returns) >= self.window_size:
                    if np.any(np.abs(window_returns) > 0.03):
                        valid.append(i)
            if valid:
                return np.array(valid)
            # Fallback - use any available window
            return np.arange(max(0, self.min_start_idx - 10), max_start + 1) if max_start > 0 else np.array([0])

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Reset environment and start new episode"""
        self.start_idx = int(self.rng.choice(self.valid_starts))
        self.current_idx = self.start_idx
        self.cash = float(self.initial_cash)
        self.shares = 0
        self.portfolio_values = [self.initial_cash]
        self.daily_returns = []
        self.episode_prices = [self.df.loc[self.start_idx, 'Close']]
        
        obs = self._get_observation()
        info = {
            'start_date': str(self.df.loc[self.start_idx, 'Date'].date()),
            'task_level': str(self.task_level)
        }
        return obs, info

    def _portfolio_value(self) -> float:
        """Compute current portfolio value"""
        price = self.df.loc[self.current_idx, 'Close']
        return float(self.cash + self.shares * price)

    def _get_rolling_benchmark(self) -> float:
        """Get 50-day rolling benchmark return (no lookahead)"""
        start = max(0, self.current_idx - 50)
        end = self.current_idx  # exclusive - does NOT include current day
        window = self.precomputed_daily_returns[start:end]
        return float(np.mean(window)) if len(window) > 0 else 0.0

    def _compute_rsi(self, period: int = 14) -> float:
        """Compute RSI(14) using correct Wilder's smoothing — no lookahead"""
        prices = self.df['Close'].iloc[:self.current_idx + 1].values
        if len(prices) < period + 1:
            return 50.0  # neutral default if insufficient history

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        if len(gains) < period:
            return 50.0

        # Step 1: seed with simple average of first `period` gains/losses
        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()

        # Step 2: apply Wilder's smoothing forward through remaining data
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi)

    def _compute_macd(self) -> Dict[str, float]:
        """Compute MACD(12, 26, 9) - no lookahead"""
        prices = self.df['Close'].iloc[:self.current_idx + 1].values
        if len(prices) < 26:
            return {
                'macd': 0.0,
                'signal': 0.0,
                'histogram': 0.0,
                'crossover': 0,
            }

        def ema(data, period):
            k = 2.0 / (period + 1)
            result = [data[0]]
            for p in data[1:]:
                result.append(p * k + result[-1] * (1 - k))
            return np.array(result)

        ema12 = ema(prices, 12)
        ema26 = ema(prices, 26)
        macd_line = ema12 - ema26
        
        if len(macd_line) < 26 + 9 - 1:
            return {
                'macd': 0.0,
                'signal': 0.0,
                'histogram': 0.0,
                'crossover': 0,
            }
        
        signal = ema(macd_line[25:], 9)
        
        if len(signal) < 2:
            return {
                'macd': float(macd_line[-1]),
                'signal': 0.0,
                'histogram': 0.0,
                'crossover': 0,
            }
        
        macd_today = float(macd_line[-1])
        signal_today = float(signal[-1])
        macd_yesterday = float(macd_line[-2])
        signal_yesterday = float(signal[-2])
        histogram = macd_today - signal_today
        
        crossover = 0
        if macd_yesterday < signal_yesterday and macd_today > signal_today:
            crossover = 1
        elif macd_yesterday > signal_yesterday and macd_today < signal_today:
            crossover = -1
        
        return {
            'macd': macd_today,
            'signal': signal_today,
            'histogram': histogram,
            'crossover': crossover,
        }

    def _compute_crossover(self) -> Dict[str, Any]:
        """Compute Golden Cross / Death Cross - no lookahead"""
        prices = self.df['Close'].iloc[:self.current_idx + 1].values
        if len(prices) < 200:
            return {
                'sma50': 0.0,
                'sma200': 0.0,
                'golden_cross': False,
                'cross_signal': 0,
            }
        
        sma50 = float(prices[-50:].mean())
        sma200 = float(prices[-200:].mean())
        golden = sma50 > sma200
        
        cross_signal = 0
        if len(prices) >= 201:
            sma50_y = float(prices[-51:-1].mean())
            sma200_y = float(prices[-201:-1].mean())
            if sma50_y <= sma200_y and sma50 > sma200:
                cross_signal = 1
            elif sma50_y >= sma200_y and sma50 < sma200:
                cross_signal = -1
        
        return {
            'sma50': sma50,
            'sma200': sma200,
            'golden_cross': golden,
            'cross_signal': cross_signal,
        }

    def _compute_indicators(self) -> Dict[str, Any]:
        """Compute all technical indicators at current_idx (no lookahead)"""
        rsi = self._compute_rsi(period=14)
        macd = self._compute_macd()
        cross = self._compute_crossover()
        
        return {
            'rsi': rsi,
            'macd': macd['macd'],
            'macd_signal': macd['signal'],
            'macd_hist': macd['histogram'],
            'macd_cross': macd['crossover'],
            'sma50': cross['sma50'],
            'sma200': cross['sma200'],
            'golden_cross': cross['golden_cross'],
            'cross_signal': cross['cross_signal'],
        }

    def _get_observation(self) -> Dict[str, Any]:
        """Build observation dict"""
        # Last 10 prices
        start_price_idx = max(0, self.current_idx - 9)
        close_prices = self.df['Close'].iloc[start_price_idx:self.current_idx + 1].tolist()
        while len(close_prices) < 10:
            close_prices.insert(0, close_prices[0])
        
        pv = self._portfolio_value()
        bh_start = self.episode_prices[0]
        bh_current = self.df.loc[self.current_idx, 'Close']
        bh_return_pct = (bh_current - bh_start) / bh_start * 100
        ep_return_pct = (pv - self.initial_cash) / self.initial_cash * 100
        indicators = self._compute_indicators()
        
        return {
            'close_prices': close_prices,
            'cash': float(self.cash),
            'shares': int(self.shares),
            'portfolio_value': float(pv),
            'day_in_episode': int(self.current_idx - self.start_idx),
            'total_days': int(self.window_size),
            'rolling_benchmark': self._get_rolling_benchmark(),
            'episode_return_pct': float(ep_return_pct),
            'benchmark_return_pct': float(bh_return_pct),
            'rsi': float(indicators['rsi']),
            'macd': float(indicators['macd']),
            'macd_signal': float(indicators['macd_signal']),
            'macd_hist': float(indicators['macd_hist']),
            'macd_cross': int(indicators['macd_cross']),
            'sma50': float(indicators['sma50']),
            'sma200': float(indicators['sma200']),
            'golden_cross': bool(indicators['golden_cross']),
            'cross_signal': int(indicators['cross_signal']),
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=hold, 1=buy all, 2=sell all
            
        Returns:
            obs, reward, done, info
        """
        price = self.df.loc[self.current_idx, 'Close']
        prev_pv = self._portfolio_value()
        
        # Execute trade with transaction cost
        if action == 1:
            # Buy: spend available cash, minus transaction cost
            shares_to_buy = int(self.cash / (price * (1 + self.transaction_cost)))
            if shares_to_buy > 0:
                cost = shares_to_buy * price * (1 + self.transaction_cost)
                self.shares += shares_to_buy
                self.cash -= cost
        elif action == 2:
            # Sell: get proceeds minus transaction cost
            if self.shares > 0:
                proceeds = self.shares * price * (1 - self.transaction_cost)
                self.cash += proceeds
                self.shares = 0
        # action == 0: hold
        
        # Advance to next day
        self.current_idx += 1
        
        curr_pv = self._portfolio_value()
        daily_return = (curr_pv - prev_pv) / prev_pv if prev_pv > 0 else 0.0
        daily_return_pct = daily_return * 100
        
        # Reward: daily return - opportunity cost (risk-free rate)
        reward = daily_return_pct - (self.risk_free_daily * 100)
        
        # Penalties
        if self.cash < 0:
            reward -= 5.0
        if curr_pv < self.initial_cash * 0.5:
            reward -= 10.0
        
        # Bonus: beat 50-day rolling benchmark
        benchmark_return = self._get_rolling_benchmark()
        if daily_return > benchmark_return:
            reward += 5.0  # bonus for beating 50-day rolling benchmark
        
        self.portfolio_values.append(curr_pv)
        self.daily_returns.append(daily_return)
        self.episode_prices.append(self.df.loc[self.current_idx, 'Close'])
        
        done = (self.current_idx >= self.start_idx + self.window_size)
        obs = self._get_observation()
        info = {
            'current_price': float(price),
            'action_taken': action,
            'shares_held': int(self.shares),
            'cash': float(self.cash),
            'portfolio_value': float(curr_pv),
            'daily_return_pct': float(daily_return_pct),
        }
        
        return obs, float(reward), done, info

    def state(self) -> Dict[str, Any]:
        """Return full internal state"""
        return {
            'task_level': self.task_level,
            'current_idx': self.current_idx,
            'start_idx': self.start_idx,
            'cash': float(self.cash),
            'shares': int(self.shares),
            'portfolio_value': float(self._portfolio_value()),
            'portfolio_values': self.portfolio_values,
            'daily_returns': self.daily_returns,
            'episode_prices': self.episode_prices,
            'day_in_episode': self.current_idx - self.start_idx,
            'window_size': self.window_size,
            'indicators': self._compute_indicators(),
            'score': self._compute_episode_score(),
        }

    def _compute_sharpe(self, returns: np.ndarray) -> float:
        """Compute annualized Sharpe ratio"""
        risk_free_daily = self.risk_free_annual / 252.0
        excess = returns - risk_free_daily
        
        if len(excess) < 2 or np.std(excess, ddof=0) == 0:
            return 0.0
        
        sharpe_daily = np.mean(excess) / np.std(excess, ddof=0)
        sharpe_annualized = sharpe_daily * np.sqrt(252)
        return float(sharpe_annualized)

    def _run_momentum_strategy(self) -> float:
        """
        Run simple momentum strategy on episode prices.
        Includes same 0.1% transaction cost as the agent for fair comparison.
        Strategy: sell if today > yesterday, buy if today <= yesterday.
        """
        prices = self.episode_prices
        cash = self.initial_cash
        shares = 0

        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                # Sell with transaction cost
                if shares > 0:
                    proceeds = shares * prices[i] * (1 - self.transaction_cost)
                    cash += proceeds
                    shares = 0
            else:
                # Buy with transaction cost
                new_shares = int(cash / (prices[i] * (1 + self.transaction_cost)))
                if new_shares > 0:
                    cost = new_shares * prices[i] * (1 + self.transaction_cost)
                    cash -= cost
                    shares += new_shares

        # Final liquidation
        if shares > 0:
            cash += shares * prices[-1] * (1 - self.transaction_cost)
            shares = 0

        final_value = cash
        return (final_value - self.initial_cash) / self.initial_cash if self.initial_cash > 0 else 0.0

    def _compute_episode_score(self) -> Dict[str, Any]:
        """Compute all metrics for graders"""
        agent_return = (self._portfolio_value() - self.initial_cash) / self.initial_cash
        start_price = self.episode_prices[0]
        end_price = self.df.loc[self.current_idx, 'Close']
        bh_return = (end_price - start_price) / start_price if start_price > 0 else 0.0
        mom_return = self._run_momentum_strategy()
        
        agent_sharpe = self._compute_sharpe(np.array(self.daily_returns)) if self.daily_returns else 0.0
        
        # Max drawdown
        pv_array = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(pv_array)
        drawdown = (peak - pv_array) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Fixed deposit return
        fd_return = (1 + self.risk_free_annual) ** (self.window_size / 252.0) - 1
        
        return {
            'agent_return': agent_return,
            'bh_return': bh_return,
            'momentum_return': mom_return,
            'agent_sharpe': agent_sharpe,
            'max_drawdown': max_dd,
            'fd_return': fd_return,
            'precomputed_sharpe': self.precomputed_sharpe,
        }

    def _build_text_observation(self, obs: Dict[str, Any]) -> str:
        """Build human-readable text observation for LLM"""
        prices = obs['close_prices']
        price_str = ', '.join([f'₹{p:,.2f}' for p in prices])
        trend = "↑ UP" if prices[-1] > prices[-2] else "↓ DOWN"
        
        rsi = obs['rsi']
        if rsi < 30:
            rsi_hint = "OVERSOLD — potential buy signal"
        elif rsi > 70:
            rsi_hint = "OVERBOUGHT — potential sell signal"
        else:
            rsi_hint = "neutral"
        
        macd_cross = obs['macd_cross']
        if macd_cross == 1:
            macd_hint = "BULLISH CROSSOVER today"
        elif macd_cross == -1:
            macd_hint = "BEARISH CROSSOVER today"
        elif obs['macd'] > obs['macd_signal']:
            macd_hint = "above signal (bullish momentum)"
        else:
            macd_hint = "below signal (bearish momentum)"
        
        cross_signal = obs['cross_signal']
        golden = obs['golden_cross']
        if cross_signal == 1:
            cross_hint = "FRESH GOLDEN CROSS today — strong bullish signal"
        elif cross_signal == -1:
            cross_hint = "FRESH DEATH CROSS today — strong bearish signal"
        elif golden:
            cross_hint = "Golden Cross active — bullish long-term trend"
        else:
            cross_hint = "Death Cross active — bearish long-term trend"
        
        text = f"""
PORTFOLIO STATUS — Day {obs['day_in_episode'] + 1} of {obs['total_days']}

Current Price: ₹{prices[-1]:,.2f} ({trend} from yesterday)
Last 10 Days:  {price_str}

Technical Indicators:
  RSI(14):        {rsi:.1f}  → {rsi_hint}
  MACD:           {obs['macd']:+.4f}  → {macd_hint}
  MACD Histogram: {obs['macd_hist']:+.4f}
  Trend Signal:   {cross_hint}
  SMA50:          ₹{obs['sma50']:,.2f}
  SMA200:         ₹{obs['sma200']:,.2f}

Your Portfolio:
  Cash:            ₹{obs['cash']:,.2f}
  Shares Held:     {obs['shares']}
  Portfolio Value: ₹{obs['portfolio_value']:,.2f}

Performance So Far:
  Your Return:             {obs['episode_return_pct']:+.2f}%
  Buy-and-Hold Would Be:   {obs['benchmark_return_pct']:+.2f}%
  50-Day Avg Daily Return: {obs['rolling_benchmark']*100:+.4f}%

ACTIONS AVAILABLE:
  0 = HOLD  (do nothing)
  1 = BUY   (buy maximum shares with all available cash)
  2 = SELL  (sell all shares for cash)

Respond with ONLY a single integer: 0, 1, or 2.
""".strip()
        return text
