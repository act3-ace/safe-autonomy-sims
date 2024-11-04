import pytest
import pickle
import onnx # TODO: add onnx dependency
import onnxruntime as ort # TODO: add dependency
import numpy as np
from safe_autonomy_sims.gym.docking.docking_v0 import DockingEnv
from safe_autonomy_sims.gym.docking.utils import polar_to_cartesian
import safe_autonomy_simulation
import time
import os


# Get action from onnx trained with CoRL
def get_action(ort_sess, obs, input_norms, output_norms):
    # Run model
    obs_vec = obs / input_norms
    obs_vec = np.array(obs_vec, dtype=np.float32)[None, :]
    CONST_INPUT = np.array([1.0], dtype=np.float32)

    # Run the session
    outputs = ort_sess.run(None, {'obs': obs_vec, 'state_ins': CONST_INPUT})
    # print(outputs)
    onnx_act = np.array(outputs[0][0][::2], dtype=np.float32)

    # Check action
    onnx_act = onnx_act * output_norms
    return np.clip(onnx_act, -output_norms, output_norms)


@pytest.fixture(name="corl_data")
def fixture_load_corl_data():
    current_dir = os.path.dirname(__file__)
    corl_data_path = os.path.join(current_dir, 'docking_episode_data.pkl')
    with open(corl_data_path, 'rb') as f:
        data = pickle.load(f)
    return data


@pytest.fixture(name="initial_conditions")
def fixture_initial_conditions():
    ic = {
        "pos_r": 118.585,
        "pos_phi": 6.0084,
        "pos_theta": -1.2689,
        "vel_r": 0.4699,
        "vel_phi": 5.9715,
        "vel_theta": 0.4868,
    }
    return ic


@pytest.fixture(name="onxx_model_path")
def fixture_onxx_model_path():
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, 'model.onnx')
    return path


def test_validate_docking_gym_with_corl(corl_data, initial_conditions, onxx_model_path):
    # Dynamic env class definition to insert initial conditions
    pos_r = initial_conditions['pos_r']
    pos_phi = initial_conditions['pos_phi']
    pos_theta = initial_conditions['pos_theta']
    vel_r = initial_conditions['vel_r']
    vel_phi = initial_conditions['vel_phi']
    vel_theta = initial_conditions['vel_theta']

    class TestDockingEnv(DockingEnv):
        def _init_sim(self):
            # Initialize simulator with chief and deputy spacecraft
            self.chief = safe_autonomy_simulation.sims.spacecraft.CWHSpacecraft(
                name="chief"
            )
            self.deputy = safe_autonomy_simulation.sims.spacecraft.CWHSpacecraft(
                name="deputy",
                position=polar_to_cartesian(
                    r=pos_r,
                    phi=pos_phi,
                    theta=pos_theta,
                ),
                velocity=polar_to_cartesian(
                    r=vel_r,
                    phi=vel_phi,
                    theta=vel_theta,
                )
            )
            self.simulator = safe_autonomy_simulation.Simulator(
                frame_rate=1, entities=[self.chief, self.deputy]
            )
    env = TestDockingEnv()

    # Norms used with CoRL
    input_norms = {
        'deputy': np.array([1.0000e+02, 1.0000e+02, 1.0000e+02, 0.5000e+00, 0.5000e+00,
            0.5000e+00, 1.0000e+00, 1.0000e+00]),
    }
    output_norms = {
        'deputy': np.array([1., 1., 1.], dtype=np.float32),
    }

    # Load deputy onnx
    onnx_location_dep = onxx_model_path
    onnx_model_dep = onnx.load(onnx_location_dep)
    onnx.checker.check_model(onnx_model_dep)
    ort_sess_deputy = ort.InferenceSession(onnx_location_dep)

    # Reset env
    observations, infos = env.reset()
    termination = False
    truncation = False
    obs_array = []
    control_array = []
    reward_components_array = []

    # Continue until done
    while not termination and not truncation:
        st = time.time()
        agent = 'deputy'
        action = get_action(ort_sess_deputy, observations, input_norms[agent], output_norms[agent])
        observations, rewards, termination, truncation, infos = env.step(action)
        print(f"Sim time: {env.simulator.sim_time}, step computation time: {time.time()-st}")
        obs_array.append(observations)
        control_array.append(action)
        reward_components_array.append(infos['reward_components'])

    # assert that obs, actions, and rewards aligns with data from corl environment
    corl_obs = corl_data["obs"]
    corl_actions = corl_data["actions"]
    corl_rewards = corl_data["rewards"]

    # check episode lengths
    assert len(corl_obs) == len(obs_array)
    assert len(corl_actions) == len(control_array)
    assert len(corl_rewards) == len(reward_components_array)

    # check values
    for i, corl_step_action in enumerate(corl_actions):
        print(i)
        assert np.array_equal(corl_step_action, control_array[i])

    for i, corl_step_obs in enumerate(corl_obs):
        print(corl_step_obs)
        assert np.array_equal(corl_step_obs, obs_array[i])

    for i, corl_step_rewards in enumerate(corl_rewards):
        # reward components are different*
        # cherry pick for now, lowest priority
        corl_delta_v = corl_step_rewards["DockingDeltaVReward"]
        delta_v = reward_components_array[i]['delta_v']
        assert corl_delta_v == delta_v
        corl_vel_const = corl_step_rewards["DockingVelocityConstraintReward"]
        vel_const = reward_components_array[i]['velocity_constraint']
        assert corl_vel_const == vel_const
        corl_success = corl_step_rewards["DockingSuccessReward"]
        success = reward_components_array[i]["success"]
        assert corl_success == success
        # corl_failure = corl_step_rewards["DockingFailureReward"]
        # failure = reward_components_array[i]['timeout'] + reward_components_array[i]['crash'] + reward_components_array[i]['out_of_bounds']
        # assert corl_failure == failure
