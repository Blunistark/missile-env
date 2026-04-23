from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

import numpy as np
import omni.usd
from pxr import UsdPhysics, Gf, UsdLux, Sdf, UsdGeom
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import RigidPrim
from omni.isaac.core import World
from omni.isaac.core.utils.viewports import set_camera_view

# ==========================================
# ⚙️ TWEAK THESE VARIABLES
# ==========================================
COM_OFFSET_X = 50.0   
SPAWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0]) 
# ==========================================

world = World()
stage = omni.usd.get_context().get_stage()

# ==========================================
# 1. LET THERE BE LIGHT & SKY
# ==========================================
sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
sun.CreateIntensityAttr(3000.0)

dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Sky"))
dome.CreateIntensityAttr(1000.0)

# ==========================================
# 2. LOAD, SCALE, AND FIX THE MISSILE
# ==========================================
usd_path = "C:/IsaacLab/assests/BrahMos.usdz" 
prim_path = "/World/BrahMos"
add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

prim = stage.GetPrimAtPath(prim_path)

if not prim.HasAPI(UsdPhysics.MassAPI):
    UsdPhysics.MassAPI.Apply(prim)
mass_api = UsdPhysics.MassAPI(prim)

mass_api.CreateMassAttr().Set(1500.0) 
mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(COM_OFFSET_X, 0.0, 0.0))

# ==========================================
# 3. ATTACH THE MOUNTED CHASE CAMERA
# ==========================================
# We place the camera inside the BrahMos folder so it moves with it!
cam_path = "/World/BrahMos/ChaseCamera"
chase_cam = UsdGeom.Camera.Define(stage, Sdf.Path(cam_path))
chase_cam.CreateClippingRangeAttr().Set(Gf.Vec2f(1.0, 10000000.0)) # Let it see far distances

# Position it 100 meters behind (-10000cm) and 20 meters up (2000cm)
xform = UsdGeom.Xformable(chase_cam)
xform.AddTranslateOp().Set(Gf.Vec3d(-50.0, 0.0, 50.0))
# Rotate it to face forward along the missile's nose
xform.AddRotateXYZOp().Set(Gf.Vec3d(90.0, 0.0, 90.0)) 


# ==========================================
# 4. INITIALIZE PHYSICS
# ==========================================
missile = RigidPrim(prim_paths_expr=prim_path, name="missile_view")

# --- SAFE SCALING! Apply 0.01 scale before the world resets ---
missile.set_local_scales(np.array([[0.01, 0.01, 0.01]]))
world.scene.add(missile)
world.reset()

missile.set_world_poses(
    positions=np.array([[0.0, 0.0, 500.0]]),
    orientations=np.array([SPAWN_QUAT])
)

print(f"🔥 Starting Wind Tunnel Test... CoM Offset: {COM_OFFSET_X}")


# ==========================================
# 5. SIMULATION LOOP
# ==========================================
while app.is_running():
    # 1. Push the missile forward
    thrust_vector = np.array([[300000.0, 0.0, 0.0]])
    missile.apply_forces(thrust_vector)
    
    # 2. Get the exact current position of the missile
    current_positions, current_orientations = missile.get_world_poses()
    m_pos = current_positions[0]
    
    # 3. Force the Viewport Camera to follow it!
    # Put camera 15 meters behind (-15.0 on X) and 3 meters up (+3.0 on Z)
    cam_pos = m_pos + np.array([-15.0, 0.0, 3.0])
    
    # Point the camera directly at the missile's center
    set_camera_view(eye=cam_pos, target=m_pos)
    
    world.step(render=True)
    
app.close()