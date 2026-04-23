---
description: Inspects code for PyTorch tensor shape mismatches, device allocation errors, and memory bottlenecks across the RL pipeline.
---

Scan the current file for data flow and tensor management issues. Specifically, look for tensor shape mismatches, incorrect device allocations (CPU vs. CUDA conflicts), and inefficient memory usage during the simulation stepping process. Trace the data pipeline from the physics state observation, through the environment, and to the PPO action outputs. Suggest optimized PyTorch operations and corrections to ensure a bottleneck-free training loop.