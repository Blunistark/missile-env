import torch
import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import RigidPrim, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.physics import set_rigid_body_enabled

from configs.missile_config import MissileConfig
from models.dynamics import FlightDynamics

class MissileActor:
    """Wrapper for a missile entity in Isaac Sim with integrated flight dynamics."""
    def __init__(self, name: str, config: MissileConfig, is_rigid: bool = True, device: str = "cpu"):
        self.name = name
        self.config = config
        self.device = device
        
        # Initialize dynamics engine
        self.dynamics = FlightDynamics(config, device=device)
        self.max_fuel = torch.tensor(config.fuel_mass, device=device)
        
        prim_path = f"/World/{name}"
        add_reference_to_stage(usd_path=config.usd_path, prim_path=prim_path)
        
        # Scale and physics setup
        scale_np = config.scale.cpu().numpy()
        if is_rigid:
            SingleRigidPrim(prim_path=prim_path, name=f"{name}_setup").set_local_scale(scale_np)
        else:
            SingleXFormPrim(prim_path=prim_path, name=f"{name}_setup").set_local_scale(scale_np)
            set_rigid_body_enabled(True, prim_path)
            
        self.view = RigidPrim(prim_paths_expr=prim_path, name=f"{name}_view")

    @property
    def current_mass(self) -> torch.Tensor:
        return self.dynamics.current_mass
        
    @property
    def fuel_mass(self) -> torch.Tensor:
        return self.dynamics.fuel_mass
        
    @fuel_mass.setter
    def fuel_mass(self, value: torch.Tensor):
        self.dynamics.fuel_mass = value

    def apply_flight_forces(self, throttle: torch.Tensor, lift_cmd: torch.Tensor, forward_dir: torch.Tensor, dt: float):
        """Calculates and applies aerodynamic and thrust forces.
        
        Args:
            throttle (torch.Tensor): Engine throttle [0, 1]
            lift_cmd (torch.Tensor): Lift command [-1, 1]
            forward_dir (torch.Tensor): Heading unit vector (3,)
            dt (float): Simulation timestep
        """
        # Get current state (Isaac Sim returns numpy for world poses/vels usually, convert to torch)
        current_vel_np = self.view.get_linear_velocities()[0]
        current_vel = torch.tensor(current_vel_np, device=self.device, dtype=torch.float32)
        
        thrust_mag, drag_vec, lift_vec = self.dynamics.calculate_forces(
            throttle, lift_cmd, current_vel, dt
        )
        
        # Apply thrust in the forward direction
        thrust_vec = forward_dir * thrust_mag
        total_force = thrust_vec + drag_vec + lift_vec
        
        # Set mass and apply force (RigidPrim expects numpy for these calls)
        self.view.set_masses(np.array([self.current_mass.cpu().item()]))
        self.view.apply_forces(total_force.cpu().numpy().reshape(1, 3))
