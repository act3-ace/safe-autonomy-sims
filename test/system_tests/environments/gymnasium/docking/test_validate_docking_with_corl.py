import pytest
import pickle
import onnx # TODO: add onnx dependency
import onnxruntime as ort # TODO: add dependency
import numpy as np
from safe_autonomy_sims.gym.docking.docking_v0 import DockingEnv
from safe_autonomy_sims.simulators.initializers.cwh import Docking3DRadialInitializer
import safe_autonomy_simulation
import os


# Get action from onnx trained with CoRL
def get_action(ort_sess, obs, input_norms, output_norms):
    # Run model
    obs_vec = obs / input_norms
    obs_vec = np.array(obs_vec, dtype=np.float32)[None, :]
    CONST_INPUT = np.array([1.0], dtype=np.float32)

    # Run the session
    outputs = ort_sess.run(None, {'obs': obs_vec, 'state_ins': CONST_INPUT})
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


@pytest.mark.system_test
def test_validate_docking_gym_with_corl(corl_data, initial_conditions, onxx_model_path):
    # Dynamic env class definition to insert initial conditions
    pos_r = initial_conditions['pos_r']
    pos_phi = initial_conditions['pos_phi']
    pos_theta = initial_conditions['pos_theta']
    vel_r = initial_conditions['vel_r']
    vel_phi = initial_conditions['vel_phi']
    vel_theta = initial_conditions['vel_theta']

    # Gym uses different initializer logic than default CoRL
    config = {
        "threshold_distance": 0.5,
        "velocity_threshold": 0.2,
        "mean_motion": 0.001027,
        "slope": 2.0,
    }
    corl_initializer = Docking3DRadialInitializer(config)
    initial_conditions_dict = corl_initializer.compute(
        radius=pos_r,
        azimuth_angle=pos_phi,
        elevation_angle=pos_theta,
        vel_max_ratio=vel_r,
        vel_azimuth_angle=vel_phi,
        vel_elevation_angle=vel_theta,
    )

    class TestDockingEnv(DockingEnv):
        def _init_sim(self):
            # Initialize simulator with chief and deputy spacecraft
            self.chief = safe_autonomy_simulation.sims.spacecraft.CWHSpacecraft(
                name="chief"
            )
            self.deputy = safe_autonomy_simulation.sims.spacecraft.CWHSpacecraft(
                name="deputy",
                position=initial_conditions_dict["position"],
                velocity=initial_conditions_dict["velocity"]
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
    np.random.seed(3)
    observations, infos = env.reset()
    termination = False
    truncation = False
    obs_array = []
    control_array = []
    reward_components_array = []

    # Continue until done
    while not termination and not truncation:
        agent = 'deputy'
        action = get_action(ort_sess_deputy, observations, input_norms[agent], output_norms[agent])
        observations, rewards, termination, truncation, infos = env.step(action)
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
        assert np.allclose(corl_step_action, control_array[i], rtol=1e-02, atol=1e-08)

    for i, corl_step_obs in enumerate(corl_obs):
        assert np.allclose(corl_step_obs, obs_array[i], rtol=1e-04, atol=1e-08)

    for i, corl_step_rewards in enumerate(corl_rewards):
        if i > 0:
            # CoRL distance reward is always 0 on the first step
            corl_distance = corl_step_rewards["DockingDistanceExponentialChangeReward"]
            distance = reward_components_array[i]['distance_pivot']
            assert corl_distance == pytest.approx(distance, rel=1e-04, abs=1e-10)
        corl_delta_v = corl_step_rewards["DockingDeltaVReward"]
        delta_v = reward_components_array[i]['delta_v']
        assert corl_delta_v == pytest.approx(delta_v, rel=1e-04)
        corl_vel_const = corl_step_rewards["DockingVelocityConstraintReward"]
        vel_const = reward_components_array[i]['velocity_constraint']
        assert corl_vel_const == pytest.approx(vel_const, rel=1e-04)
        corl_success = corl_step_rewards["DockingSuccessReward"]
        success = reward_components_array[i]["success"]
        assert corl_success == pytest.approx(success, rel=1e-04)
        corl_failure = corl_step_rewards["DockingFailureReward"]
        failure = reward_components_array[i]['timeout'] + reward_components_array[i]['crash'] + reward_components_array[i]['out_of_bounds']
        assert corl_failure == pytest.approx(failure, rel=1e-04)
