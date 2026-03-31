"""Portfolio Environment OpenEnv Wrapper"""

from typing import Dict, Any
from uuid import uuid4
try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:
    # Fallback if openenv not installed
    Environment = object
    # State = object

try:
    from models import PortfolioAction, PortfolioObservation, PortfolioState
    from .env import PortfolioEnv
except ImportError:
    from ..models import PortfolioAction, PortfolioObservation, PortfolioState
    from .env import PortfolioEnv

class PortfolioEnvironment(Environment):
    """OpenEnv-compatible Portfolio Environment wrapper"""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_level: int = 1):
        """Initialize environment"""
        self.game = PortfolioEnv(task_level=task_level, seed=42)
        self._state  = State(episode_id=str(uuid4()), step_count=0)


    def reset(self) -> PortfolioObservation:
        """Reset environment and return initial observation"""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        obs, info = self.game.reset()
        return self._build_observation(obs, reward=0.0, done=False)

    def step(self, action: PortfolioAction) -> PortfolioObservation:
        """Execute action and return observation"""
        self._state.step_count += 1
        obs, reward, done, info = self.game.step(action.action)
        return self._build_observation(obs, reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    def _build_observation(self, obs: Dict[str, Any], reward: float, done: bool) -> PortfolioObservation:
        """Build PortfolioObservation from raw observation dict"""
        obs_dict = {
            **obs,
            'reward': reward,
            'done': done,
            'text_observation': self.game._build_text_observation(obs),
        }
        return PortfolioObservation(**obs_dict)
