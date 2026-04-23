import numpy as np

class FlightDynamics:
    """Calculates realistic thrust, drag, and lift based on current state."""
    def __init__(self, profile: dict):
        self.dry_mass = profile["dry_mass"]
        self.fuel_mass = profile["fuel_mass"]
        self.max_thrust = profile["max_thrust_n"]
        self.burn_rate = profile["burn_rate_kg_s"]
        
        # Simplified Aerodynamic coefficients
        self.drag_coeff = 0.03  # How fast it loses speed without thrust
        self.gravity = 9.81

    @property
    def current_mass(self):
        return self.dry_mass + self.fuel_mass

    def calculate_forces(self, throttle: float, lift_cmd: float, current_velocity: np.ndarray, dt: float):
        """Returns: (Thrust Scalar, Drag Vector, Lift Vector)"""
        speed = np.linalg.norm(current_velocity)

        # --- 1. THRUST & FUEL BURN ---
        thrust_mag = 0.0
        if self.fuel_mass > 0:
            throttle = np.clip(throttle, 0.0, 1.0)
            burn_amt = self.burn_rate * throttle * dt
            
            if burn_amt > self.fuel_mass:
                burn_amt = self.fuel_mass
                throttle = burn_amt / (self.burn_rate * dt)
                
            self.fuel_mass -= burn_amt
            thrust_mag = self.max_thrust * throttle

        # --- 2. AERODYNAMIC DRAG ---
        # Drag increases with the square of velocity (F_drag = k * v^2)
        drag_mag = self.drag_coeff * (speed ** 2)
        drag_vector = np.array([0.0, 0.0, 0.0])
        if speed > 1.0:
            # Apply drag in the exact opposite direction of movement
            drag_vector = -(current_velocity / speed) * drag_mag

        # --- 3. DYNAMIC LIFT ---
        # You cannot generate lift if you are not moving fast!
        max_lift_capacity = self.current_mass * self.gravity * 2.0  # Max 2G maneuver
        
        # Speed factor: Lift effectiveness drops to 0 if flying below 100 m/s
        speed_factor = np.clip(speed / 300.0, 0.0, 1.0) 
        
        lift_mag = lift_cmd * max_lift_capacity * speed_factor
        lift_vector = np.array([0.0, 0.0, lift_mag])

        return thrust_mag, drag_vector, lift_vector