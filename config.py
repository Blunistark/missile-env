import numpy as np

PATHS = {
    "BRAHMOS": "C:\\IsaacLab\\assests\\BrahMos.usdz",
    "S400": "C:\\IsaacLab\\assests\\S-400_Triumf_SAM_system.usdz",
    "INTERCEPTOR": "C:\\IsaacLab\\assests\\Missile_FATEH_110.usdz"
}

SCENARIO = {
    # NEW SPATIAL LAYOUT
    "LAND_SIZE": 400000.0,                         # 400km x 400km square
    "HVT_POS": np.array([0.0, 0.0, 0.0]),          # Target exactly at the center
    "S400_POS": np.array([-10000.0, 0.0, 0.0]),    # S-400 placed 10km in front of target
    
    # NEW SPAWN DISTANCE
    "SPAWN_DIST_X": -150000.0,        # BrahMos spawns 150km away (Just outside 120km barrier)
    
    # EXISTING PARAMETERS
    "CRUISE_ALTITUDE": 15000.0,       
    "S400_LAUNCH_RANGE": 120000.0,    
    "RADAR_MAST_HEIGHT": 40.0,        
    "INTERCEPT_PROXIMITY": 1500.0,    
    "HVT_HIT_PROXIMITY": 500.0        
}

MISSILES = {
    "BRAHMOS": {
        "dry_mass": 1500.0,
        "fuel_mass": 1500.0,
        "max_thrust_n": 300000.0,
        "burn_rate_kg_s": 100.0,
        "scale": np.array([100.0, 100.0, 100.0]),
        "spawn_quat": np.array([0.70711, 0.0, 0.70711, 0.0])
    },
    "INTERCEPTOR": {
        "dry_mass": 500.0,
        "fuel_mass": 1000.0,
        "max_thrust_n": 400000.0,
        "burn_rate_kg_s": 150.0,
        "scale": np.array([50.0, 50.0, 50.0]),
        "spawn_quat": np.array([0.7071068, 0.0, -0.7071068, 0.0])
    }
}