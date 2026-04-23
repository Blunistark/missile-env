from dataclasses import dataclass, field
import torch

@dataclass(frozen=True)
class MissileConfig:
    """Configuration for missile physical and aerodynamic properties.
    
    All values are in SI units (kg, m, s, N).
    """
    dry_mass: float
    fuel_mass: float
    max_thrust_n: float
    burn_rate_kg_s: float
    scale: torch.Tensor
    spawn_quat: torch.Tensor
    usd_path: str

    def __post_init__(self):
        # Ensure tensors are correctly typed and on the right device if needed
        # Using CPU by default for config, but logic should move to CUDA if specified
        if not isinstance(self.scale, torch.Tensor):
            object.__setattr__(self, "scale", torch.tensor(self.scale, dtype=torch.float32))
        if not isinstance(self.spawn_quat, torch.Tensor):
            object.__setattr__(self, "spawn_quat", torch.tensor(self.spawn_quat, dtype=torch.float32))

BRAHMOS_CONFIG = MissileConfig(
    dry_mass=1500.0,
    fuel_mass=1500.0,
    max_thrust_n=300000.0,
    burn_rate_kg_s=100.0,
    scale=torch.tensor([100.0, 100.0, 100.0]),
    spawn_quat=torch.tensor([0.70711, 0.0, 0.70711, 0.0]),
    usd_path="C:\\IsaacLab\\assests\\BrahMos.usdz"
)

INTERCEPTOR_CONFIG = MissileConfig(
    dry_mass=500.0,
    fuel_mass=1000.0,
    max_thrust_n=400000.0,
    burn_rate_kg_s=150.0,
    scale=torch.tensor([50.0, 50.0, 50.0]),
    spawn_quat=torch.tensor([0.7071068, 0.0, -0.7071068, 0.0]),
    usd_path="C:\\IsaacLab\\assests\\Missile_FATEH_110.usdz"
)

S400_PATH = "C:\\IsaacLab\\assests\\S-400_Triumf_SAM_system.usdz"
