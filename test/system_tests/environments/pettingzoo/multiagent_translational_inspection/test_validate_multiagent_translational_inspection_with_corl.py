import pytest
import pickle
import onnx # TODO: add onnx dependency
import onnxruntime as ort # TODO: add dependency
import numpy as np
from safe_autonomy_sims.pettingzoo.inspection.multi_inspection_v0 import MultiInspectionEnv
import os
import safe_autonomy_simulation.sims.inspection as sim


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
    corl_data_path = os.path.join(current_dir, 'multiagent_translational_inspection_episode_data.pkl')
    with open(corl_data_path, 'rb') as f:
        data = pickle.load(f)
    return data


@pytest.fixture(name="onnx_model_path")
def fixture_onnx_model_path():
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, 'models')
    return path


@pytest.fixture(name="initial_conditions")
def fixture_initial_conditions():
    ic = {
        'sun_angle': 1.424975716637098,
        'blue0_position': np.array([-12.6520983,   1.842438, -32.60466323]),
        'blue0_velocity': np.array([-0.05525816, -0.16368131, -0.07830456]),
        'blue1_position': np.array([-15.54117335,   2.26315411,  35.33016436]),
        'blue1_velocity': np.array([-0.07330804, -0.21714724,  0.06216107]),
        'blue2_position': np.array([-29.18383274,   4.24984071, -24.75601503]),
        'blue2_velocity': np.array([-0.03754564, -0.11121469, -0.12248139]),
    }
    return ic


@pytest.mark.system_test
def test_validate_multiagent_inspection_pettingzoo_with_corl(corl_data, onnx_model_path, initial_conditions):
    # Dynamic env class definition to insert initial conditions
    deputies = [f"deputy_{i}" for i in range(3)]
    models = ["blue0_ctrl", "blue1_ctrl", "blue2_ctrl"]

    class TestMultiInspectionEnv(MultiInspectionEnv):
        def _init_sim(self):
            # Initialize spacecraft, sun, and simulator
            self.chief = sim.Target(
                name="chief",
                num_points=100,
                radius=10,
            )
            self.deputies = {
                deputies[0]: sim.Inspector(
                    name=deputies[0],
                    position=initial_conditions['blue0_position'],
                    velocity=initial_conditions['blue0_velocity']
                ),
                deputies[1]: sim.Inspector(
                    name=deputies[1],
                    position=initial_conditions['blue1_position'],
                    velocity=initial_conditions['blue1_velocity']
                ),
                deputies[2]: sim.Inspector(
                    name=deputies[2],
                    position=initial_conditions['blue2_position'],
                    velocity=initial_conditions['blue2_velocity']
                ),
            }
            self.sun = sim.Sun(theta=initial_conditions["sun_angle"])
            self.simulator = sim.InspectionSimulator(
                frame_rate=0.1,
                inspectors=list(self.deputies.values()),
                targets=[self.chief],
                sun=self.sun,
            )
    env = TestMultiInspectionEnv(num_agents=3)

    # Norms used with CoRL
    input_norms = np.array([
        1.0000e+02, 1.0000e+02, 1.0000e+02, # position
        0.5000e+00, 0.5000e+00, 0.5000e+00, # velocity
        1.0000e+02, # points
        1.0000e+00, 1.0e+0, 1.0e+0, # uninspected points
        1.0e+0 # sun angle
    ])
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
    corl_obs_order = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 6]
    # first obs not recorded in CoRL's EpisodeArtifact
    for deputy, obs in observations.items():
        observations[deputy] = obs[corl_obs_order]
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
        for deputy, obs in observations.items():
            observations[deputy] = obs[corl_obs_order]
        obs_array.append(observations)
        control_array.append(action)
        info_array.append(infos)

    # assert that obs, actions, and rewards aligns with data from corl environment
    corl_obs0 = corl_data["obs0"]
    corl_actions0 = corl_data["actions0"]
    corl_rewards0 = corl_data["rewards0"]
    corl_obs1 = corl_data["obs1"]
    corl_actions1 = corl_data["actions1"]
    corl_rewards1 = corl_data["rewards1"]
    corl_obs2 = corl_data["obs2"]
    corl_actions2 = corl_data["actions2"]
    corl_rewards2 = corl_data["rewards2"]

    # check episode lengths
    # assert len(corl_obs0) == len(obs_array)
    # assert len(corl_actions0) == len(control_array)
    # assert len(corl_rewards0) == len(info_array)

    # check values
    for i, gym_step_action_dict in enumerate(control_array):
        if gym_step_action_dict['deputy_0'] is not None:
            assert np.allclose(corl_actions0[i], gym_step_action_dict['deputy_0'], rtol=1e-03, atol=1e-08)
        if gym_step_action_dict['deputy_1'] is not None:
            assert np.allclose(corl_actions1[i], gym_step_action_dict['deputy_1'], rtol=1e-03, atol=1e-08)
        if gym_step_action_dict['deputy_2'] is not None:
            assert np.allclose(corl_actions2[i], gym_step_action_dict['deputy_2'], rtol=1e-03, atol=1e-08)

    for i, gym_step_obs_dict in enumerate(obs_array):
        if gym_step_obs_dict['deputy_0'] is not None:
            assert np.allclose(corl_obs0[i], gym_step_obs_dict['deputy_0'], rtol=1e-04, atol=1e-08)
        if gym_step_obs_dict['deputy_1'] is not None:
            assert np.allclose(corl_obs1[i], gym_step_obs_dict['deputy_1'], rtol=1e-04, atol=1e-08)
        if gym_step_obs_dict['deputy_2'] is not None:
            assert np.allclose(corl_obs2[i], gym_step_obs_dict['deputy_2'], rtol=1e-04, atol=1e-08)

    for i, gym_step_rewards in enumerate(info_array):
        if gym_step_obs_dict['deputy_0'] is not None:
            corl_delta_v = corl_rewards0[i]["InspectionDeltaVReward"]
            delta_v = gym_step_rewards['deputy_0']['reward_components']['delta_v']
            assert corl_delta_v == pytest.approx(delta_v, rel=1e-04, abs=1e-10)
            corl_inspected_points = corl_rewards0[i]["ObservedPointsReward"]
            inspected_points = gym_step_rewards['deputy_0']['reward_components']['observed_points']
            assert corl_inspected_points == pytest.approx(inspected_points, rel=1e-04, abs=1e-10)
            corl_success = corl_rewards0[i]["InspectionSuccessReward"]
            success = gym_step_rewards['deputy_0']['reward_components']["success"]
            assert corl_success == pytest.approx(success, rel=1e-04, abs=1e-10)
            corl_crash = corl_rewards0[i]["InspectionCrashReward"]
            crash = gym_step_rewards['deputy_0']['reward_components']['crash']
            assert corl_crash == pytest.approx(crash, rel=1e-04, abs=1e-10)
        
        if gym_step_obs_dict['deputy_1'] is not None:
            corl_delta_v = corl_rewards1[i]["InspectionDeltaVReward"]
            delta_v = gym_step_rewards['deputy_1']['reward_components']['delta_v']
            assert corl_delta_v == pytest.approx(delta_v, rel=1e-04, abs=1e-10)
            corl_inspected_points = corl_rewards1[i]["ObservedPointsReward"]
            inspected_points = gym_step_rewards['deputy_1']['reward_components']['observed_points']
            assert corl_inspected_points == pytest.approx(inspected_points, rel=1e-04, abs=1e-10)
            corl_success = corl_rewards1[i]["InspectionSuccessReward"]
            success = gym_step_rewards['deputy_1']['reward_components']["success"]
            assert corl_success == pytest.approx(success, rel=1e-04, abs=1e-10)
            corl_crash = corl_rewards1[i]["InspectionCrashReward"]
            crash = gym_step_rewards['deputy_1']['reward_components']['crash']
            assert corl_crash == pytest.approx(crash, rel=1e-04, abs=1e-10)
        
        if gym_step_obs_dict['deputy_2'] is not None:
            corl_delta_v = corl_rewards2[i]["InspectionDeltaVReward"]
            delta_v = gym_step_rewards['deputy_2']['reward_components']['delta_v']
            assert corl_delta_v == pytest.approx(delta_v, rel=1e-04, abs=1e-10)
            corl_inspected_points = corl_rewards2[i]["ObservedPointsReward"]
            inspected_points = gym_step_rewards['deputy_2']['reward_components']['observed_points']
            assert corl_inspected_points == pytest.approx(inspected_points, rel=1e-04, abs=1e-10)
            corl_success = corl_rewards2[i]["InspectionSuccessReward"]
            success = gym_step_rewards['deputy_2']['reward_components']["success"]
            assert corl_success == pytest.approx(success, rel=1e-04, abs=1e-10)
            corl_crash = corl_rewards2[i]["InspectionCrashReward"]
            crash = gym_step_rewards['deputy_2']['reward_components']['crash']
            assert corl_crash == pytest.approx(crash, rel=1e-04, abs=1e-10)
