from isaacsim import SimulationApp

# Initialize simulation app in windowed mode for visualization
simulation_app = SimulationApp({"headless": False})

import torch
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.viewports import set_camera_view
from configs.missile_config import BRAHMOS_CONFIG
from models.missile_actor import MissileActor

def run_physics_bench():
    print("🧪 Starting Physics & Cruise Diagnostic...")
    world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    
    # Initialize a single BrahMos missile
    brahmos = MissileActor(
        prim_path="/World/BrahMos_Test", 
        name="BrahMos_Test", 
        config=BRAHMOS_CONFIG
    )
    world.scene.add(brahmos.view)
    world.reset()
    
    # Initial Setup
    brahmos.fuel_mass = torch.tensor(BRAHMOS_CONFIG.fuel_mass)
    brahmos.setup_physics_properties()
    
    # Spawn at 15km altitude, moving at Mach 1
    spawn_pos = np.array([0.0, 0.0, 15000.0])
    spawn_quat = BRAHMOS_CONFIG.spawn_quat.cpu().numpy()
    brahmos.view.set_world_poses(positions=np.array([spawn_pos]), orientations=np.array([spawn_quat]))
    brahmos.view.set_linear_velocities(np.array([[340.0, 0.0, 0.0]])) # Start at Mach 1
    
    set_camera_view(eye=np.array([-100.0, -100.0, 15100.0]), target=np.array([0.0, 0.0, 15000.0]))

    print(f"{'Step':<6} | {'Alt (m)':<10} | {'Speed (m/s)':<12} | {'Mass (kg)':<10} | {'Pitch (deg)':<10}")
    print("-" * 60)

    for i in range(1200): # 20 seconds of flight
        # Actions: Full throttle, trim lift for level flight
        action = torch.tensor([1.0, 0.04]) # 0.04 lift is approx trim for cruise
        
        # Get state
        vel_np = brahmos.view.get_linear_velocities()[0]
        pos_np, quat_np = brahmos.view.get_world_poses()
        
        # Calculate Forward Direction
        speed = np.linalg.norm(vel_np)
        fwd = torch.tensor(vel_np / speed if speed > 1.0 else [1, 0, 0], dtype=torch.float32)
        
        # Apply physics
        brahmos.apply_flight_forces(
            throttle=action[0], 
            lift_cmd=action[1], 
            forward_dir=fwd, 
            dt=1.0/60.0
        )
        
        world.step(render=True)
        
        # Log telemetry every second (60 steps)
        if i % 60 == 0:
            alt = pos_np[0][2]
            mass = brahmos.current_mass.item()
            # Simple pitch estimation from quaternion (WXYZ)
            q = quat_np[0]
            pitch = np.degrees(np.arcsin(2.0 * (q[0]*q[2] - q[3]*q[1])))
            
            print(f"{i:<6} | {alt:<10.1f} | {speed:<12.1f} | {mass:<10.1f} | {pitch:<10.2f}")

    print("\n✅ Physics test complete.")
    simulation_app.close()

if __name__ == "__main__":
    run_physics_bench()
