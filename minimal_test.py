from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf

def run_minimal():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    # 1. Create a Parent Xform at the desired CoM/Pivot location
    actor_path = "/World/BrahMos_Actor"
    actor_prim = SingleXFormPrim(prim_path=actor_path, name="brahmos_actor")
    
    # 2. Add the Missile USD as a CHILD of the Actor
    missile_usd = "C:/IsaacLab/assests/BrahMos.usdz"
    visuals_path = f"{actor_path}/Visuals"
    add_reference_to_stage(usd_path=missile_usd, prim_path=visuals_path)
    
    # 3. Offset the Visuals so the middle of the missile is at the Actor's (0,0,0)
    # The missile is ~8000 units long. We shift it -4000 units back.
    visuals_prim = SingleXFormPrim(prim_path=visuals_path, name="visuals")
    visuals_prim.set_local_scale(np.array([0.001, 0.001, 0.001]))
    
    # Apply offset: Shift -4.0 meters (or -4000 local units) back
    # Also keep the -90 Y rotation to lay it flat
    visuals_prim.set_local_pose(
        translation=np.array([-4.0, 0.0, 0.0]), 
        orientation=np.array([0.70711, 0.0, -0.70711, 0.0])
    )
    
    world.reset()
    
    # Now, moving the Actor moves the WHOLE missile around its center!
    actor_prim.set_world_pose(position=np.array([0.0, 0.0, 1.0]))
    
    print("✅ CoM Fixed. Gizmo should now be in the CENTER of the missile.")
    
    while simulation_app.is_running():
        world.step(render=True)

if __name__ == "__main__":
    run_minimal()
    simulation_app.close()
