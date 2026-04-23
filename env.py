import numpy as np

from pxr import UsdLux, Sdf
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, VisualSphere
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.viewports import set_camera_view 
from isaacsim.util.debug_draw import _debug_draw as omni_debug_draw
from isaacsim.core.utils.stage import add_reference_to_stage
import config
from actors import MissileActor

class TacticalCombatEnv:
    def __init__(self):
        self.dt = 1.0 / 60.0
        self.world = World(physics_dt=self.dt, rendering_dt=self.dt)
        self.draw = omni_debug_draw.acquire_debug_draw_interface()
        
        self.brahmos_history = []
        self.interceptor_history = []
        self.interceptor_launched = False
        
        self._setup_world()

    def _setup_world(self):
        stage = self.world.stage
        UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun")).CreateIntensityAttr(6000.0)

        # 1. SQUARE GROUND MAP (Centered at origin)
        land_size = config.SCENARIO["LAND_SIZE"]
        FixedCuboid(
            prim_path="/World/Battlefield", name="land", 
            position=np.array([0.0, 0.0, -500.0]), # Centered
            scale=np.array([land_size, land_size, 1000.0]), 
            color=np.array([0.15, 0.15, 0.15])
        )

        # 2. OFFSET S-400 BASE (-10km X)
        s400_pos = config.SCENARIO["S400_POS"]
        add_reference_to_stage(usd_path=config.PATHS["S400"], prim_path="/World/S400_System")
        s400_base = SingleXFormPrim(prim_path="/World/S400_System", name="s400_base")
        s400_base.set_local_scale(np.array([50.0, 50.0, 50.0]))
        s400_base.set_world_pose(position=s400_pos)
        # 3. CENTERED HVT (0,0,0)
        hvt_pos = config.SCENARIO["HVT_POS"]
        VisualSphere(
            prim_path="/World/HVT_City", name="target_city", 
            position=hvt_pos, 
            radius=2000.0, color=np.array([0.0, 0.0, 1.0])
        )

        # 4. LOAD ACTORS
        self.brahmos = MissileActor("BrahMos", "BRAHMOS", config.PATHS["BRAHMOS"], is_rigid=True)
        self.interceptor = MissileActor("Interceptor", "INTERCEPTOR", config.PATHS["INTERCEPTOR"], is_rigid=False)
        
        self.world.scene.add(self.brahmos.view)
        self.world.scene.add(self.interceptor.view)

    def _get_observations(self):
        b_pos, _ = self.brahmos.view.get_world_poses()
        b_vel = self.brahmos.view.get_linear_velocities()
        
        dist_to_target = np.linalg.norm(b_pos[0])
        fuel_pct = self.brahmos.fuel_mass / self.brahmos.max_fuel
        
        # This is what your neural network will receive
        return np.array([
            b_pos[0][0], b_pos[0][2],  # X, Z pos
            b_vel[0][0], b_vel[0][2],  # X, Z vel
            dist_to_target,
            fuel_pct,
            float(self.interceptor_launched)
        ], dtype=np.float32)

    def _compute_reward(self, b_pos, i_pos, terminated, outcome):
        reward = 0.0
        
        # Dense Reward (Encourage forward movement)
        dist_to_target = np.linalg.norm(b_pos[0])
        reward -= (dist_to_target / 100000.0) 
        
        # Sparse Rewards
        if terminated:
            if outcome == "HIT_TARGET": reward += 1000.0
            elif outcome == "INTERCEPTED": reward -= 500.0
            elif outcome == "CRASHED": reward -= 500.0
                
        return reward

    def reset(self):
        self.world.reset()
        self.draw.clear_lines()
        self.brahmos_history.clear()
        self.interceptor_history.clear()
        
        self.interceptor_launched = False
        self.interceptor_ignition = False 
        
        self.brahmos.fuel_mass = self.brahmos.max_fuel
        self.interceptor.fuel_mass = self.interceptor.max_fuel
        
        b_quat = config.MISSILES["BRAHMOS"]["spawn_quat"]

        # Spawn BrahMos at closer distance
        self.brahmos.view.set_masses(np.array([self.brahmos.current_mass]))
        self.brahmos.view.set_world_poses(
            positions=np.array([[config.SCENARIO["SPAWN_DIST_X"], 0.0, config.SCENARIO["CRUISE_ALTITUDE"]]]),
            orientations=np.array([b_quat]) 
        )

        # Spawn Interceptor at S-400 location (NOT 0,0)
        s400_pos = config.SCENARIO["S400_POS"]
        i_quat = config.MISSILES["INTERCEPTOR"]["spawn_quat"]
        vertical_quat = np.array([0.7071068, 0.0, -0.7071068, 0.0])
        self.interceptor.view.set_masses(np.array([self.interceptor.current_mass]))
        self.interceptor.view.set_world_poses(
            positions=np.array([[s400_pos[0], s400_pos[1], 10.0]]), 
            orientations=np.array([i_quat])
        )
        
        # Move Camera to view the new centered arena
        set_camera_view(eye=np.array([-50000.0, -100000.0, 40000.0]), target=np.array([0.0, 0.0, 0.0]))
        return self._get_observations(), {}

    def step(self, action):
        # 1. Apply BrahMos Actions
        b_vel = self.brahmos.view.get_linear_velocities()[0]
        speed = np.linalg.norm(b_vel)
        b_forward = (b_vel / speed) if speed > 1.0 else np.array([1.0, 0.0, 0.0])
        self.brahmos.apply_flight_forces(throttle=action[0], lift_cmd=action[1], forward_dir=b_forward, dt=self.dt)

        self.world.step(render=True)
        
        b_pos, _ = self.brahmos.view.get_world_poses()
        i_pos, _ = self.interceptor.view.get_world_poses()
        
        # NEW DISTANCE MATH: Measure relative to specific assets, not origin
        dist_to_s400 = np.linalg.norm(b_pos[0] - config.SCENARIO["S400_POS"])
        dist_to_target = np.linalg.norm(b_pos[0] - config.SCENARIO["HVT_POS"])

        # 2. S-400 Radar Logic
        h_missile = max(0.1, b_pos[0][2])
        max_los_dist = 3570.0 * (np.sqrt(config.SCENARIO["RADAR_MAST_HEIGHT"]) + np.sqrt(h_missile))
        
        # Check if within radar horizon AND within launch range
        is_visible = (dist_to_s400 < max_los_dist)
        if is_visible and dist_to_s400 < config.SCENARIO["S400_LAUNCH_RANGE"] and not self.interceptor_launched:
            self.interceptor_launched = True

        # 3. S-400 Cold Launch State Machine
        s400_pos = config.SCENARIO["S400_POS"]
        if self.interceptor_launched:
            alt = i_pos[0][2]
            
            if alt < 100.0 and not self.interceptor_ignition:
                ejection_force = self.interceptor.current_mass * 9.81 * 10.0 
                self.interceptor.view.apply_forces(np.array([[0.0, 0.0, ejection_force]]))
            else:
                if not self.interceptor_ignition:
                    print(f"🔥 S-400 MAIN MOTOR IGNITION at {alt:.1f}m altitude!")
                    self.interceptor_ignition = True
                    
                dir_vec = (b_pos[0] - i_pos[0])
                dir_vec /= np.linalg.norm(dir_vec)
                self.interceptor.apply_flight_forces(throttle=1.0, lift_cmd=0.0, forward_dir=dir_vec, dt=self.dt)
        else:
            self.interceptor.view.set_world_poses(
                positions=np.array([[s400_pos[0], s400_pos[1], 10.0]]), 
                orientations=np.array([[0.7071068, 0.0, -0.7071068, 0.0]])
            )
            self.interceptor.view.set_linear_velocities(np.array([[0.0, 0.0, 0.0]]))

        self._orient_missiles(b_pos, i_pos)

        # 4. Terminations
        terminated, outcome = False, "IN_FLIGHT"
        if self.interceptor_launched and np.linalg.norm(b_pos[0] - i_pos[0]) < config.SCENARIO["INTERCEPT_PROXIMITY"]:
            terminated, outcome = True, "INTERCEPTED"
        elif dist_to_target < config.SCENARIO["HVT_HIT_PROXIMITY"]: # Use dist_to_target now!
            terminated, outcome = True, "HIT_TARGET"
        elif b_pos[0][2] < 0:
            terminated, outcome = True, "CRASHED"
            
        obs = self._get_observations()
        reward = self._compute_reward(b_pos, i_pos, terminated, outcome)

        return obs, reward, terminated, False, {"outcome": outcome}
    
    def _orient_missiles(self, b_pos, i_pos):
        def _get_velocity_quat(vel):
            speed = np.linalg.norm(vel)
            if speed < 1.0: return np.array([1.0, 0.0, 0.0, 0.0]) 
            
            fwd = vel / speed
            v1 = np.array([1.0, 0.0, 0.0])
            dot = np.dot(v1, fwd)
            if dot > 0.9999: return np.array([1.0, 0.0, 0.0, 0.0])
            if dot < -0.9999: return np.array([0.0, 0.0, 0.0, 1.0])
            
            cross = np.cross(v1, fwd)
            q = np.array([1.0 + dot, cross[0], cross[1], cross[2]])
            return q / np.linalg.norm(q)

        # NEW: Bulletproof native WXYZ Quaternion Multiplication
        def _quat_mult(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])

        # 1. BRAHMOS ORIENTATION
        b_vels = self.brahmos.view.get_linear_velocities()
        if np.linalg.norm(b_vels[0]) > 1.0:
            vel_quat = _get_velocity_quat(b_vels[0])
            model_offset_quat = config.MISSILES["BRAHMOS"]["spawn_quat"]
            
            # Use our custom math function instead!
            final_b_quat = _quat_mult(vel_quat, model_offset_quat)
            
            self.brahmos.view.set_world_poses(positions=b_pos, orientations=np.array([final_b_quat]))
        
        # 2. INTERCEPTOR ORIENTATION
        if self.interceptor_launched:
            i_vels = self.interceptor.view.get_linear_velocities()
            if np.linalg.norm(i_vels[0]) > 1.0:
                vel_quat = _get_velocity_quat(i_vels[0])
                
                # Default identity offset if it's already facing +X
                model_offset_quat = np.array([1.0, 0.0, 0.0, 0.0]) 
                
                # Use our custom math function instead!
                final_i_quat = _quat_mult(vel_quat, model_offset_quat)
                self.interceptor.view.set_world_poses(positions=i_pos, orientations=np.array([final_i_quat]))