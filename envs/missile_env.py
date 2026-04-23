import torch
import numpy as np
from pxr import UsdLux, Sdf

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, VisualSphere
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.viewports import set_camera_view 
from isaacsim.util.debug_draw import _debug_draw as omni_debug_draw
from isaacsim.core.utils.stage import add_reference_to_stage

from configs.missile_config import BRAHMOS_CONFIG, INTERCEPTOR_CONFIG, S400_PATH
from configs.scenario_config import DEFAULT_SCENARIO, ScenarioConfig
from models.missile_actor import MissileActor
from training.rewards import compute_tactical_reward

class TacticalCombatEnv:
    """Integrated Isaac Sim environment for missile combat scenarios."""
    def __init__(self, scenario: ScenarioConfig = DEFAULT_SCENARIO, device: str = "cpu"):
        self.device = device
        self.config = scenario
        self.dt = 1.0 / 60.0
        self.world = World(physics_dt=self.dt, rendering_dt=self.dt)
        self.draw = omni_debug_draw.acquire_debug_draw_interface()
        
        self.interceptor_launched = False
        self.interceptor_ignition = False
        
        self._setup_world()

    def _setup_world(self):
        stage = self.world.stage
        UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun")).CreateIntensityAttr(6000.0)

        # 1. Ground Map
        land_size = self.config.land_size
        FixedCuboid(
            prim_path="/World/Battlefield", name="land", 
            position=np.array([0.0, 0.0, -500.0]),
            scale=np.array([land_size, land_size, 1000.0]), 
            color=np.array([0.15, 0.15, 0.15])
        )

        # 2. S-400 Base
        s400_pos_np = self.config.s400_pos.cpu().numpy()
        add_reference_to_stage(usd_path=S400_PATH, prim_path="/World/S400_System")
        s400_base = SingleXFormPrim(prim_path="/World/S400_System", name="s400_base")
        s400_base.set_local_scale(np.array([50.0, 50.0, 50.0]))
        s400_base.set_world_pose(position=s400_pos_np)

        # 3. HVT
        hvt_pos_np = self.config.hvt_pos.cpu().numpy()
        VisualSphere(
            prim_path="/World/HVT_City", name="target_city", 
            position=hvt_pos_np, 
            radius=2000.0, color=np.array([0.0, 0.0, 1.0])
        )

        # 4. Actors
        self.brahmos = MissileActor("BrahMos", BRAHMOS_CONFIG, is_rigid=True, device=self.device)
        self.interceptor = MissileActor("Interceptor", INTERCEPTOR_CONFIG, is_rigid=False, device=self.device)
        
        self.world.scene.add(self.brahmos.view)
        self.world.scene.add(self.interceptor.view)

    def _get_observations(self) -> torch.Tensor:
        b_pos_np, _ = self.brahmos.view.get_world_poses()
        b_vel_np = self.brahmos.view.get_linear_velocities()
        
        b_pos = torch.tensor(b_pos_np[0], device=self.device, dtype=torch.float32)
        b_vel = torch.tensor(b_vel_np[0], device=self.device, dtype=torch.float32)
        
        dist_to_target = torch.norm(b_pos - self.config.hvt_pos.to(self.device))
        fuel_pct = self.brahmos.fuel_mass / self.brahmos.max_fuel
        
        obs = torch.tensor([
            b_pos[0], b_pos[2],  # X, Z pos
            b_vel[0], b_vel[2],  # X, Z vel
            dist_to_target,
            fuel_pct,
            float(self.interceptor_launched)
        ], device=self.device, dtype=torch.float32)
        
        return obs

    def reset(self):
        self.world.reset()
        self.interceptor_launched = False
        self.interceptor_ignition = False 
        
        self.brahmos.fuel_mass = torch.tensor(BRAHMOS_CONFIG.fuel_mass, device=self.device)
        self.interceptor.fuel_mass = torch.tensor(INTERCEPTOR_CONFIG.fuel_mass, device=self.device)
        
        # Spawn BrahMos
        b_pos = np.array([self.config.spawn_dist_x, 0.0, self.config.cruise_altitude])
        b_quat = BRAHMOS_CONFIG.spawn_quat.cpu().numpy()
        self.brahmos.view.set_world_poses(positions=np.array([b_pos]), orientations=np.array([b_quat]))

        # Spawn Interceptor at S-400
        s400_pos_np = self.config.s400_pos.cpu().numpy()
        i_pos = np.array([s400_pos_np[0], s400_pos_np[1], 10.0])
        i_quat = INTERCEPTOR_CONFIG.spawn_quat.cpu().numpy()
        self.interceptor.view.set_world_poses(positions=np.array([i_pos]), orientations=np.array([i_quat]))
        
        set_camera_view(eye=np.array([-50000.0, -100000.0, 40000.0]), target=np.array([0.0, 0.0, 0.0]))
        return self._get_observations(), {}

    def step(self, action: torch.Tensor):
        """Action should be torch.Tensor of shape (2,) -> [throttle, lift]"""
        # 1. BrahMos Logic
        b_vel_np = self.brahmos.view.get_linear_velocities()[0]
        b_vel = torch.tensor(b_vel_np, device=self.device, dtype=torch.float32)
        speed = torch.norm(b_vel)
        b_fwd = (b_vel / speed) if speed > 1.0 else torch.tensor([1.0, 0.0, 0.0], device=self.device)
        
        self.brahmos.apply_flight_forces(
            throttle=action[0], lift_cmd=action[1], forward_dir=b_fwd, dt=self.dt
        )

        self.world.step(render=True)
        
        b_pos_np, _ = self.brahmos.view.get_world_poses()
        i_pos_np, _ = self.interceptor.view.get_world_poses()
        
        b_pos = torch.tensor(b_pos_np[0], device=self.device, dtype=torch.float32)
        i_pos = torch.tensor(i_pos_np[0], device=self.device, dtype=torch.float32)
        
        # 2. S-400 Radar & Launch
        s400_pos = self.config.s400_pos.to(self.device)
        dist_to_s400 = torch.norm(b_pos - s400_pos)
        
        h_missile = torch.clamp(b_pos[2], min=0.1)
        max_los = 3570.0 * (np.sqrt(self.config.radar_mast_height) + torch.sqrt(h_missile))
        
        if (dist_to_s400 < max_los) and (dist_to_s400 < self.config.s400_launch_range) and not self.interceptor_launched:
            self.interceptor_launched = True

        # 3. Interceptor Logic
        if self.interceptor_launched:
            alt = i_pos[2]
            if alt < 100.0 and not self.interceptor_ignition:
                ejection_force = self.interceptor.current_mass * 9.81 * 10.0 
                self.interceptor.view.apply_forces(np.array([[0.0, 0.0, ejection_force.cpu().item()]]))
            else:
                if not self.interceptor_ignition:
                    self.interceptor_ignition = True
                
                i_fwd = (b_pos - i_pos)
                i_fwd /= torch.norm(i_fwd)
                self.interceptor.apply_flight_forces(
                    throttle=torch.tensor(1.0, device=self.device), 
                    lift_cmd=torch.tensor(0.0, device=self.device), 
                    forward_dir=i_fwd, dt=self.dt
                )
        else:
            # Keep on launcher
            i_fixed_pos = np.array([s400_pos_np[0], s400_pos_np[1], 10.0])
            i_fixed_quat = INTERCEPTOR_CONFIG.spawn_quat.cpu().numpy()
            self.interceptor.view.set_world_poses(positions=np.array([i_fixed_pos]), orientations=np.array([i_fixed_quat]))
            self.interceptor.view.set_linear_velocities(np.array([[0.0, 0.0, 0.0]]))

        self._orient_missiles(b_pos, i_pos)

        # 4. Terminations
        terminated, outcome = False, "IN_FLIGHT"
        dist_to_interceptor = torch.norm(b_pos - i_pos)
        dist_to_target = torch.norm(b_pos - self.config.hvt_pos.to(self.device))
        
        if self.interceptor_launched and dist_to_interceptor < self.config.intercept_proximity:
            terminated, outcome = True, "INTERCEPTED"
        elif dist_to_target < self.config.hvt_hit_proximity:
            terminated, outcome = True, "HIT_TARGET"
        elif b_pos[2] < 0:
            terminated, outcome = True, "CRASHED"
            
        obs = self._get_observations()
        reward = compute_tactical_reward(
            b_pos, self.config.hvt_pos.to(self.device), 
            self.interceptor_launched, terminated, outcome, device=self.device
        )

        return obs, reward, terminated, False, {"outcome": outcome}

    def _orient_missiles(self, b_pos, i_pos):
        # Utility to align missile model with velocity vector using quaternions
        def _get_velocity_quat(vel: torch.Tensor):
            speed = torch.norm(vel)
            if speed < 1.0: return torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device) 
            fwd = vel / speed
            v1 = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            dot = torch.dot(v1, fwd)
            if dot > 0.9999: return torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            if dot < -0.9999: return torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            cross = torch.cross(v1, fwd)
            q = torch.tensor([1.0 + dot, cross[0], cross[1], cross[2]], device=self.device)
            return q / torch.norm(q)

        def _quat_mult(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return torch.tensor([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], device=self.device)

        # BrahMos Orientation
        b_vel_np = self.brahmos.view.get_linear_velocities()[0]
        b_vel = torch.tensor(b_vel_np, device=self.device, dtype=torch.float32)
        if torch.norm(b_vel) > 1.0:
            vel_quat = _get_velocity_quat(b_vel)
            final_quat = _quat_mult(vel_quat, BRAHMOS_CONFIG.spawn_quat.to(self.device))
            self.brahmos.view.set_world_poses(orientations=np.array([final_quat.cpu().numpy()]))
        
        # Interceptor Orientation
        if self.interceptor_launched:
            i_vel_np = self.interceptor.view.get_linear_velocities()[0]
            i_vel = torch.tensor(i_vel_np, device=self.device, dtype=torch.float32)
            if torch.norm(i_vel) > 1.0:
                vel_quat = _get_velocity_quat(i_vel)
                self.interceptor.view.set_world_poses(orientations=np.array([vel_quat.cpu().numpy()]))
