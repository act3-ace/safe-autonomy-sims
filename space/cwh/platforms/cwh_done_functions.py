from act3_rl_core.libraries.environment_dict import DoneDict
from act3_rl_core.dones import DoneFuncBase
# need to import get_platform_name, WIP
import numpy as np

class MaxDistanceDoneFunction(DoneFuncBase):
    def __init__(self):
        super.__init__()

    def __call__(self,observation,action,next_observation,next_state):

        done = DoneDict()

        # compute distance to origin
        platform = get_platform_name(next_state,self.agent)
        pos = platform.position

        # compute to origin
        origin = np.array([0,0,0])
        dist = np.linalg.norm(origin-np.array(pos))

        done[self.agent] = dist > self.config.max_distance

        return done

class SuccessfulDockingDoneFunction(DoneFuncBase):
    def __init__(self):
        super.__init__()

    def __call__(self,observation,action,next_observation,next_state):
        # eventually will include velocity constraint
        done = DoneDict()
        platform = get_platform_name(next_state,self.agent)

        pos = platform.position
        origin = np.array([0,0,0])
        docking_region_radius = self.config.docking_region_radius

        radial_distance = np.linalg.norm(np.array(pos) - origin)
        done[self.agent] = radial_distance <= docking_region_radius

        return done
