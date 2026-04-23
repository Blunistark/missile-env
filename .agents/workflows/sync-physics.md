---
description: Validates and synchronizes external vehicle dynamics with Isaac Lab's simulation clock, ensuring coordinate system alignment and stable integration.
---

Review the current vehicle dynamics and physics logic within the active file. Ensure all state variables, aerodynamic forces, and kinematics correctly interface with the flight dynamics engine (e.g., JSBSim). Explicitly check that the physics timesteps ($dt$) are synchronized with Isaac Lab to prevent simulation lag or instability. Verify that coordinate transformations (such as converting NED aerospace standards to Isaac Sim's Z-up environment) are handled correctly and explicitly documented.