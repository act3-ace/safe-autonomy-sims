import pytest
import pickle
import onnx # TODO: add onnx dependency
import onnxruntime as ort # TODO: add dependency
import numpy as np
from safe_autonomy_sims.pettingzoo.docking.multidocking_v0 import MultiDockingEnv
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
    corl_data_path = os.path.join(current_dir, 'multidocking_v0_episode_data.pkl')
    with open(corl_data_path, 'rb') as f:
        data = pickle.load(f)
    return data


@pytest.fixture(name="onnx_model_path")
def fixture_onnx_model_path():
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, 'models')
    return path


@pytest.mark.integration
def test_validate_multiagent_docking_pettingzoo_with_corl(corl_data, onnx_model_path):
    # Dynamic env class definition to insert initial conditions
    deputies = [f"deputy_{i}" for i in range(3)]
    models = ["blue0_ctrl", "blue1_ctrl", "blue2_ctrl"]

    class TestDockingEnv(MultiDockingEnv):
        def _init_sim(self):
            # Initialize simulator with chief and deputy spacecraft
            self.chief = safe_autonomy_simulation.sims.spacecraft.CWHSpacecraft(
                name="chief"
            )
            self.possible_agents = deputies
            self.deputies = {
                a: safe_autonomy_simulation.sims.spacecraft.CWHSpacecraft(
                    name=a,
                    position=corl_data['IC'][m]['Obs_Sensor_Position']['direct_observation'].value,
                    velocity=corl_data['IC'][m]['Obs_Sensor_Velocity']['direct_observation'].value
                )
                for a, m in zip(self.possible_agents, models)
            }
            self.simulator = safe_autonomy_simulation.Simulator(
                frame_rate=1, entities=[self.chief] + list(self.deputies.values())
            )
    env = TestDockingEnv(num_agents=3)

    # Norms used with CoRL
    input_norms = np.array([1.0000e+02, 1.0000e+02, 1.0000e+02, 0.5000e+00, 0.5000e+00, 0.5000e+00, 1.0000e+00, 1.0000e+00])
    output_norms = np.array([1., 1., 1.], dtype=np.float32)

    # Load deputy onnx models
    ort_sessions = {}
    for deputy, model in zip(deputies, models):
        onnx_location_dep = os.path.join(onnx_model_path, model, 'model.onnx')
        onnx_model_dep = onnx.load(onnx_location_dep)
        onnx.checker.check_model(onnx_model_dep)
        ort_sessions[deputy] = ort.InferenceSession(onnx_location_dep)

    # Reset env
    np.random.seed(3)
    observations, infos = env.reset()
    termination = dict.fromkeys(deputies, False)
    truncation = dict.fromkeys(deputies, False)
    obs_array = []
    control_array = []
    info_array = []

    # Continue until done
    while not any(termination.values()) and not any(truncation.values()):
        action = {}
        for agent in deputies:
            action[agent] = get_action(ort_sessions[agent], observations[agent], input_norms, output_norms)
        observations, rewards, termination, truncation, infos = env.step(action)
        obs_array.append(observations)
        control_array.append(action)
        info_array.append(infos)

    # assert that obs, actions, and rewards aligns with data from corl environment
    corl_obs = corl_data["obs"]
    corl_actions = corl_data["actions"]
    corl_rewards = corl_data["rewards"]

    # check episode lengths
    assert len(corl_obs) == len(obs_array)
    assert len(corl_actions) == len(control_array)
    assert len(corl_rewards) == len(info_array)

    # check values
    for i, corl_step_action in enumerate(corl_actions):
        for deputy, model in zip(deputies, models):
            # TODO: grade tolerance such that it increases with episode length
            assert np.allclose(corl_step_action[model], control_array[i][deputy], rtol=8e-01, atol=1e-08)

    for i, corl_step_obs in enumerate(corl_obs):
        for deputy, model in zip(deputies, models):
            assert np.allclose(corl_step_obs[model], obs_array[i][deputy], rtol=1e-02, atol=1e-08)

    for i, corl_step_rewards in enumerate(corl_rewards):
        for deputy, model in zip(deputies, models):
            if i > 0:
                # CoRL distance reward is always 0 on the first step
                corl_distance = corl_step_rewards[model]["DockingDistanceExponentialChangeReward"]
                distance = info_array[i][deputy]["reward_components"]['distance_pivot']
                assert corl_distance == pytest.approx(distance, rel=1e-03, abs=1e-10)
            corl_delta_v = corl_step_rewards[model]["DockingDeltaVReward"]
            delta_v = info_array[i][deputy]["reward_components"]['delta_v']
            assert corl_delta_v == pytest.approx(delta_v, rel=1e-04, abs=1e-08)
            corl_vel_const = corl_step_rewards[model]["DockingVelocityConstraintReward"]
            vel_const = info_array[i][deputy]["reward_components"]['velocity_constraint']
            assert corl_vel_const == pytest.approx(vel_const, rel=1e-04)
            corl_success = corl_step_rewards[model]["DockingSuccessReward"]
            success = info_array[i][deputy]["reward_components"]["success"]
            assert corl_success == pytest.approx(success, rel=1e-04)
            corl_failure = corl_step_rewards[model]["DockingFailureReward"]
            failure = info_array[i][deputy]["reward_components"]['timeout'] + info_array[i][deputy]["reward_components"]['crash'] + info_array[i][deputy]["reward_components"]['out_of_bounds']
            assert corl_failure == pytest.approx(failure, rel=1e-04)
