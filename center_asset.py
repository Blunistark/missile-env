from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": True}) # Headless is fine for math

from isaacsim.core.utils.stage import add_reference_to_stage, save_stage
from isaacsim.core.api import World
from pxr import UsdGeom, Gf

def center_missile():
    world = World()
    usd_path = "C:/IsaacLab/assests/BrahMos.usdz"
    save_path = "C:/IsaacLab/assests/BrahMos_Centered.usd"
    prim_path = "/World/BrahMos"
    
    print(f"🔧 Centering asset: {usd_path}")
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    
    # Get the prim and its bounding box
    stage = world.stage
    prim = stage.GetPrimAtPath(prim_path)
    bbox_cache = UsdGeom.BBoxCache(UsdGeom.GetStageTimeCodesRange(stage)[0], [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    
    # Calculate the exact center of the bounding box
    range = bbox.GetRange()
    center = range.GetMidpoint()
    
    print(f"📍 Original Center found at: {center}")
    
    # Apply the inverse translation to center it perfectly at (0,0,0)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder() # Reset any existing offsets
    xform.AddTranslateOp().Set(-center)
    
    # Save the new file
    save_stage(save_path)
    print(f"✅ Success! Centered model saved to: {save_path}")

if __name__ == "__main__":
    center_missile()
    simulation_app.close()
