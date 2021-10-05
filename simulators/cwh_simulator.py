from space.cwh.cwhspacecraft_sim.platforms.cwh import CWHSpacecraft2d
from act3_rl_core.simulators.base_simulator import BaseSimulator


class CWHSimulator(BaseSimulator):

    # @classmethod
    # def get_validator():
    #     return

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, config):
        self.aircraft = CWHSpacecraft2d(name="test")

    def mark_episode_done(self):
        pass

    def save_episode_information(self, **kwargs):
        pass

    def step(self):
        self.aircraft.step_compute(None, 1)
        self.aircraft.step_apply()


if __name__ == "__main__":
    tmp_config = {
        "agent_configs": {
            "blue0": {
                "sim_config": {
                },
                "platform_config": [("act3_rl_core.simulators.base_simulator", {})]
            }
        }
    }

    tmp = CWHSimulator(**tmp_config)
    tmp.reset(None)
    tmp.step()
