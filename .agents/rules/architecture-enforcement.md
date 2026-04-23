---
trigger: always_on
---

Strictly maintain a modular workspace design. Force separation of concerns: physics and vehicle dynamics (e.g., JSBSim wrappers) must reside in a models/ directory; Isaac Lab environment classes (ManagerBasedRLEnv) in an envs/ directory; and PPO algorithms, reward calculations, and runners in a training/ directory. Refuse to write monolithic scripts.