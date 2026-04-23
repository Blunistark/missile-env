from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for the tactical combat scenario layout and rules."""
    land_size: float = 400000.0  # 400km x 400km square
    hvt_pos: torch.Tensor = torch.tensor([0.0, 0.0, 0.0])
    s400_pos: torch.Tensor = torch.tensor([-10000.0, 0.0, 0.0])
    spawn_dist_x: float = -150000.0
    cruise_altitude: float = 15000.0
    s400_launch_range: float = 120000.0
    radar_mast_height: float = 40.0
    intercept_proximity: float = 1500.0
    hvt_hit_proximity: float = 500.0

DEFAULT_SCENARIO = ScenarioConfig()
