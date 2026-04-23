import torch
from configs.missile_config import MissileConfig

class FlightDynamics:
    """Calculates realistic thrust, drag, and lift based on current state.
    
    Mathematical Formulation:
    - Mass: m(t) = m_dry + m_fuel(t)
    - Thrust: F_thrust = T_max * throttle (if fuel > 0)
    - Drag: F_drag = -0.5 * rho * v^2 * Cd * A (simplified as -Cd * |v|^2 * unit(v))
    - Lift: F_lift = L_cmd * max_lift * speed_factor * [0, 0, 1]
    """
    def __init__(self, config: MissileConfig, device: str = "cpu"):
        self.device = device
        self.dry_mass = torch.tensor(config.dry_mass, device=device)
        self.fuel_mass = torch.tensor(config.fuel_mass, device=device)
        self.max_thrust = torch.tensor(config.max_thrust_n, device=device)
        self.burn_rate = torch.tensor(config.burn_rate_kg_s, device=device)
        
        self.drag_coeff = torch.tensor(0.03, device=device)
        self.gravity = torch.tensor(9.81, device=device)

    @property
    def current_mass(self) -> torch.Tensor:
        """Returns: torch.Tensor of shape (1,)"""
        return self.dry_mass + self.fuel_mass

    def calculate_forces(self, throttle: torch.Tensor, lift_cmd: torch.Tensor, current_velocity: torch.Tensor, dt: float):
        """Calculates forces acting on the vehicle.
        
        Args:
            throttle (torch.Tensor): Scalar [0, 1]
            lift_cmd (torch.Tensor): Scalar [-1, 1]
            current_velocity (torch.Tensor): Shape (3,) in world coordinates
            dt (float): Timestep
            
        Returns:
            thrust_mag (torch.Tensor): Scalar magnitude
            drag_vector (torch.Tensor): Shape (3,)
            lift_vector (torch.Tensor): Shape (3,)
        """
        speed = torch.norm(current_velocity)

        # --- 1. THRUST & FUEL BURN ---
        thrust_mag = torch.tensor(0.0, device=self.device)
        if self.fuel_mass > 0:
            throttle = torch.clamp(throttle, 0.0, 1.0)
            burn_amt = self.burn_rate * throttle * dt
            
            if burn_amt > self.fuel_mass:
                burn_amt = self.fuel_mass.clone()
                throttle = burn_amt / (self.burn_rate * dt)
                
            self.fuel_mass -= burn_amt
            thrust_mag = self.max_thrust * throttle

        # --- 2. AERODYNAMIC DRAG ---
        # F_drag = k * v^2
        drag_mag = self.drag_coeff * (speed ** 2)
        drag_vector = torch.zeros(3, device=self.device)
        if speed > 1.0:
            drag_vector = -(current_velocity / speed) * drag_mag

        # --- 3. DYNAMIC LIFT ---
        max_lift_capacity = self.current_mass * self.gravity * 2.0  # Max 2G maneuver
        speed_factor = torch.clamp(speed / 300.0, 0.0, 1.0) 
        
        lift_mag = lift_cmd * max_lift_capacity * speed_factor
        lift_vector = torch.tensor([0.0, 0.0, lift_mag], device=self.device)

        return thrust_mag, drag_vector, lift_vector
