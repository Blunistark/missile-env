# ==========================================
# 1. INITIALIZE APP FIRST
# ==========================================
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.util.debug_draw")

import numpy as np

# ==========================================
# 2. IMPORT CUSTOM MODULES SECOND
# ==========================================
from env import TacticalCombatEnv
import config

if __name__ == "__main__":
    env = TacticalCombatEnv()
    
    # Gymnasium API Standard Reset
    obs, info = env.reset()

    while simulation_app.is_running():
        
        # --- SIMPLE AUTOPILOT (Proportional Controller) ---
        # obs array index 1 is Altitude (Z position)
        current_altitude = obs[1] 
        target_altitude = config.SCENARIO["CRUISE_ALTITUDE"]
        
        # Calculate how far off we are from 15,000 meters
        alt_error = target_altitude - current_altitude
        
        # Base lift needed to hover is ~0.5. 
        # If we are too low (positive error), increase lift. If too high, decrease lift.
        lift_cmd = 0.5 + (alt_error * 0.001) 
        lift_cmd = np.clip(lift_cmd, -1.0, 1.0) # Fins can only deflect so much
        
        # Action: [100% Throttle, Dynamic Lift]
        mock_action = np.array([1.0, lift_cmd]) 
        # --------------------------------------------------
        
        # Step the Environment
        obs, reward, terminated, truncated, info = env.step(action=mock_action)
        
        if terminated or truncated:
            print(f"Episode Ended: {info['outcome']} | Final Reward: {reward:.2f}")
            obs, info = env.reset()

    # Clean shutdown
    simulation_app.close()