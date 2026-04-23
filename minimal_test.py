from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage

def run_minimal():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    # Using the exact path from BRAHMOS_CONFIG
    missile_usd = "C:/IsaacLab/assests/BrahMos.usdz"
    prim_path = "/World/StaticMissile"
    
    print(f"🚀 Loading BrahMos from: {missile_usd}")
    add_reference_to_stage(usd_path=missile_usd, prim_path=prim_path)
    
    # Use SingleXFormPrim
    missile = SingleXFormPrim(prim_path=prim_path, name="test_missile")
    
    world.reset()
    
    # Apply a much smaller scale to compensate for the large USD natural size
    # And rotate -90 degrees around Y to point the nose along the X-axis
    missile.set_local_scale(np.array([0.001, 0.001, 0.001]))
    
    # Quaternion for X-forward (W, X, Y, Z)
    # This rotates the vertical (+Z) nose to the horizontal (+X)
    x_forward_quat = np.array([0.70711, 0.0, -0.70711, 0.0])
    
    missile.set_world_pose(
        position=np.array([0.0, 0.0, 2.0]),
        orientation=x_forward_quat
    )
    
    print("✅ Minimal Environment Ready. Missile should be at (0,0,5).")
    
    while simulation_app.is_running():
        world.step(render=True)

if __name__ == "__main__":
    run_minimal()
    simulation_app.close()
