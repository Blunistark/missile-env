from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage

def run_minimal():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    # Simple static reference
    missile_usd = "C:/IsaacLab/BrahMos_Red.usd"  # Using the path from your config
    prim_path = "/World/StaticMissile"
    
    print(f"🚀 Loading missile from: {missile_usd}")
    add_reference_to_stage(usd_path=missile_usd, prim_path=prim_path)
    
    # Use SingleXFormPrim (no physics properties to crash)
    missile = SingleXFormPrim(prim_path=prim_path, name="test_missile")
    
    world.reset()
    
    # Set a test pose: 5m high, looking slightly up
    # Orientation: [w, x, y, z]
    missile.set_world_pose(
        position=np.array([0.0, 0.0, 5.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0])
    )
    
    print("✅ Minimal Environment Ready. Missile should be at (0,0,5).")
    
    while simulation_app.is_running():
        world.step(render=True)

if __name__ == "__main__":
    run_minimal()
    simulation_app.close()
