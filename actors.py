import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleRigidPrim, RigidPrim, SingleXFormPrim
from isaacsim.core.utils.physics import set_rigid_body_enabled

import config
from physics import FlightDynamics

class MissileActor:
    def __init__(self, name: str, profile_name: str, usd_path: str, is_rigid=True):
        profile = config.MISSILES[profile_name]
        self.name = name
        
        # Initialize the hardcore physics calculator
        self.dynamics = FlightDynamics(profile)
        self.max_fuel = profile["fuel_mass"]
        
        prim_path = f"/World/{name}"
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        
        if is_rigid:
            SingleRigidPrim(prim_path=prim_path, name=f"{name}_setup").set_local_scale(profile["scale"])
        else:
            SingleXFormPrim(prim_path=prim_path, name=f"{name}_setup").set_local_scale(profile["scale"])
            set_rigid_body_enabled(True, prim_path)
            
        self.view = RigidPrim(prim_paths_expr=prim_path, name=f"{name}_view")

    # --- THE FIX: Property Bridges to Physics Engine ---
    @property
    def current_mass(self):
        return self.dynamics.current_mass
        
    @property
    def fuel_mass(self):
        return self.dynamics.fuel_mass
        
    @fuel_mass.setter
    def fuel_mass(self, value):
        self.dynamics.fuel_mass = value

    def apply_flight_forces(self, throttle: float, lift_cmd: float, forward_dir: np.ndarray, dt: float):
        current_vel = self.view.get_linear_velocities()[0]
        thrust_mag, drag_vec, lift_vec = self.dynamics.calculate_forces(throttle, lift_cmd, current_vel, dt)
        
        # Apply thrust exactly in the direction the missile is pointing
        thrust_vec = forward_dir * thrust_mag
        
        total_force = thrust_vec + drag_vec + lift_vec
        
        self.view.set_masses(np.array([self.current_mass]))
        self.view.apply_forces(np.array([total_force]))