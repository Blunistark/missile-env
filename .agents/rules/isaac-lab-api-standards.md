---
trigger: always_on
---

Strictly use the modern isaacsim.* API namespace (e.g., from isaacsim.core.api import World) and entirely avoid legacy omni.isaac.* or omni.isaac.lab imports. Before generating, reviewing, or refactoring environment code, you must reference the official documentation at https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html and the tutorials at https://docs.isaacsim.omniverse.nvidia.com/5.1.0/isaac_lab_tutorials/index.html to ensure correct syntax. Ensure all data passed between modules (observations, actions, rewards) are strictly typed PyTorch tensors. Explicitly manage tensor device placement (CPU vs. CUDA) to prevent cross-device memory bottlenecks during simulation steps.