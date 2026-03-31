"""Portfolio Environment Client"""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import PortfolioAction, PortfolioObservation, PortfolioState
except ImportError:
    from models import PortfolioAction, PortfolioObservation, PortfolioState


class PortfolioEnvClient(
    EnvClient[PortfolioAction, PortfolioObservation, PortfolioState]
):
    """
    Client for the Portfolio Management Environment.
    Maintains a persistent WebSocket connection to the server.

    Usage:
        # Connect to running server
        with PortfolioEnvClient(base_url="http://localhost:8000") as client:
            result = client.reset()
            print(result.observation.portfolio_value)

            result = client.step(PortfolioAction(action=1))
            print(result.observation.episode_return_pct)

        # Start from Docker image
        client = PortfolioEnvClient.from_docker_image("portfolio-env:latest")
        try:
            result = client.reset()
            result = client.step(PortfolioAction(action=0))
        finally:
            client.close()
    """

    def _step_payload(self, action: PortfolioAction) -> Dict:
        """Convert PortfolioAction to JSON payload."""
        return {"action": action.action}

    def _parse_result(self, payload: Dict) -> StepResult[PortfolioObservation]:
        """Parse server response into StepResult[PortfolioObservation]."""
        obs_data    = payload.get("observation", {})
        observation = PortfolioObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> PortfolioState:
        """Parse server response into PortfolioState."""
        return PortfolioState(**payload)