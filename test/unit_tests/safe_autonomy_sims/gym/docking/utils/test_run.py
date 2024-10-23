from safe_autonomy_sims.gym.docking.docking_v0 import DockingEnv
import numpy as np

d = DockingEnv()
d._init_sim()
a = np.array([0,0,0])
out = d.step(a)
print(out)
