import torch

def compute_tactical_reward(
    brahmos_pos: torch.Tensor,
    target_pos: torch.Tensor,
    interceptor_launched: bool,
    terminated: bool,
    outcome: str,
    device: str = "cpu"
) -> torch.Tensor:
    """Computes a dense and sparse reward for the tactical combat scenario.
    
    Mathematical Formulation:
    - Dense Reward: R_dense = -||p_brahmos - p_target|| / 100,000
    - Sparse Rewards:
        - Outcome "HIT_TARGET": +1000.0
        - Outcome "INTERCEPTED": -500.0
        - Outcome "CRASHED": -500.0
    
    Args:
        brahmos_pos (torch.Tensor): Shape (3,)
        target_pos (torch.Tensor): Shape (3,)
        interceptor_launched (bool): True if interceptor is in flight
        terminated (bool): True if episode ended
        outcome (str): One of ["HIT_TARGET", "INTERCEPTED", "CRASHED", "IN_FLIGHT"]
        
    Returns:
        reward (torch.Tensor): Scalar reward
    """
    reward = torch.tensor(0.0, device=device)
    
    # 1. Dense Reward (Proximity to target)
    dist_to_target = torch.norm(brahmos_pos - target_pos)
    reward -= (dist_to_target / 100000.0)
    
    # 2. Sparse Rewards on Termination
    if terminated:
        if outcome == "HIT_TARGET":
            reward += 1000.0
        elif outcome == "INTERCEPTED":
            reward -= 500.0
        elif outcome == "CRASHED":
            reward -= 500.0
            
    return reward
