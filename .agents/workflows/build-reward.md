---
description: Generates mathematical and tensor-optimized reward functions tailored for high-speed dynamics and evasion objectives, ensuring compatibility with PPO convergence.
---

Help me construct a new modular reward function for the current environment. I will provide the specific objective (e.g., missile evasion, target tracking, or energy efficiency). Generate the mathematical logic and format any complex formulas clearly. Ensure the code handles PyTorch tensor operations efficiently on the GPU, includes appropriate penalties/bonuses (like survival or control effort), and remains fully decoupled so it can be seamlessly imported into the main ManagerBasedRLEnv observation and reward pipeline.