import torch
from envs.missile_env import TacticalCombatEnv

def main():
    # Initialize the modular environment
    # Use "cuda" if available, else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing Tactical Combat Environment on {device}...")
    
    env = TacticalCombatEnv(device=device)
    obs, info = env.reset()
    
    # Simple control loop
    for i in range(1000):
        # Action: [Throttle (0 to 1), Lift (-1 to 1)]
        # For now, just cruise forward with full throttle and slight lift
        action = torch.tensor([1.0, 0.05], device=device)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 60 == 0:
            print(f"Step {i} | Pos: {obs[0]:.1f}, {obs[1]:.1f} | Reward: {reward.item():.4f} | Outcome: {info['outcome']}")
            
        if terminated:
            print(f"💥 Episode Finished! Outcome: {info['outcome']} | Final Reward: {reward.item():.2f}")
            break

    env.world.close()

if __name__ == "__main__":
    main()