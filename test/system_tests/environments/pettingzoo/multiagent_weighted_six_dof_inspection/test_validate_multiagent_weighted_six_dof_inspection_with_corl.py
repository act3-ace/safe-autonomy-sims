import pytest
import pickle
import onnx # TODO: add onnx dependency
import onnxruntime as ort # TODO: add dependency
import numpy as np
from safe_autonomy_sims.pettingzoo.inspection.sixdof_multi_inspection_v0 import WeightedSixDofMultiInspectionEnv
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
    corl_data_path = os.path.join(current_dir, 'multiagent_weighted_six_dof_inspection_episode_data.pkl')
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
        'sun_angle': 4.638011137123597,
        'priority_vector_azimuth_angle': 1.740797543236975,
        'priority_vector_elevation_angle': -0.2916098416406021,

        'blue0_position': np.array([11.430147336766295, 63.568910155357834, 11.31895934084295]),
        'blue0_velocity': np.array([0.00230551198121581, -0.014048140092895723, 0.23191365557839458]),
        "blue0_angular_velocity": np.array([-0.009014330025620028, -0.007101284824735656, 0.007311050038821246]),

        'blue1_position': np.array([-29.31479904751045, 37.155355533746715, -47.12654798021313]),
        'blue1_velocity': np.array([-0.0013852601336736417, -0.003318044554698742, 0.0031932885738289543]),
        "blue1_angular_velocity": np.array([-0.0047554460525149975, -0.0033270346650434955, 0.007147049058518156]),

        'blue2_position': np.array([-73.41978911241524, 18.715397760419425, 51.90245238760071]),
        'blue2_velocity': np.array([-0.12398404379534131, -0.17514801489112783, -0.07716526666833894]),
        "blue2_angular_velocity": np.array([-0.005104106532254922, 0.0006993988875818591, -0.0012258391469860408]),
    }
    return ic


# TODO: Update the test to the new gymnasium observation space
#       This requires reconfiguring the observations in CoRL, recreating a policy, exporting the onnx model, rerunning CoRL eval, 
#       updating and rerunning the parse_corl_eval.py script, and updating this test to use the new test artifacts + observation space.
@pytest.mark.skip
def test_validate_multiagent_six_dof_inspection_pettingzoo_with_corl(corl_data, onnx_model_path, initial_conditions):
    # Dynamic env class definition to insert initial conditions
    deputies = [f"deputy_{i}" for i in range(3)]
    models = ["blue0_ctrl", "blue1_ctrl", "blue2_ctrl"]

    # priority vector
    init_priority_vector = np.zeros((3,), dtype=np.float32)
    init_priority_vector[0] = np.cos(initial_conditions["priority_vector_azimuth_angle"]) * np.cos(initial_conditions["priority_vector_elevation_angle"])
    init_priority_vector[1] = np.sin(initial_conditions["priority_vector_azimuth_angle"]) * np.cos(initial_conditions["priority_vector_elevation_angle"])
    init_priority_vector[2] = np.sin(initial_conditions["priority_vector_elevation_angle"])

    class TestWeightedSixDofMultiInspectionEnv(WeightedSixDofMultiInspectionEnv):
        def _init_sim(self):
            # Initialize spacecraft, sun, and simulator
            priority_vector = init_priority_vector
            priority_vector /= np.linalg.norm(priority_vector)  # convert to unit vector
            self.chief = sim.SixDOFTarget(
                name="chief",
                num_points=100,
                radius=10,
                priority_vector=priority_vector,
            )
            self.deputies = {
                deputies[0]: sim.SixDOFInspector(
                    name=deputies[0],
                    position=initial_conditions['blue0_position'],
                    velocity=initial_conditions['blue0_velocity'],
                    orientation=corl_data["IC"]["blue0_ctrl"]["Obs_Sensor_Quaternion"]["direct_observation"].value,
                    angular_velocity=initial_conditions["blue0_angular_velocity"],
                    fov=1.0471975511965976,  # 60 degrees
                    focal_length=9.6e-3,
                ),
                deputies[1]: sim.SixDOFInspector(
                    name=deputies[1],
                    position=initial_conditions['blue1_position'],
                    velocity=initial_conditions['blue1_velocity'],
                    orientation=corl_data["IC"]["blue1_ctrl"]["Obs_Sensor_Quaternion"]["direct_observation"].value,
                    angular_velocity=initial_conditions["blue1_angular_velocity"],
                    fov=1.0471975511965976,  # 60 degrees
                    focal_length=9.6e-3,
                ),
                deputies[2]: sim.SixDOFInspector(
                    name=deputies[2],
                    position=initial_conditions['blue2_position'],
                    velocity=initial_conditions['blue2_velocity'],
                    orientation=corl_data["IC"]["blue2_ctrl"]["Obs_Sensor_Quaternion"]["direct_observation"].value,
                    angular_velocity=initial_conditions["blue2_angular_velocity"],
                    fov=1.0471975511965976,  # 60 degrees
                    focal_length=9.6e-3,
                ),
            }
            self.sun = sim.Sun(theta=initial_conditions["sun_angle"])
            self.simulator = sim.InspectionSimulator(
                frame_rate=0.1,
                inspectors=list(self.deputies.values()),
                targets=[self.chief],
                sun=self.sun,
            )

    env = TestWeightedSixDofMultiInspectionEnv(num_agents=3)

    # Norms used with CoRL
    input_norms = {
        'deputy': np.array([
            100.0, 100.0, 100.0, # relative position
            175.0, 1.0, 1.0, 1.0, # relative position magnorm
            0.5, 0.5, 0.5, # relative velocity
            0.866, 1.0, 1.0, 1.0, # relative velocity magnorm
            0.05, 0.05, 0.05, # angular velocity
            1.0, 1.0, 1.0, # camera direction?
            1.0, 1.0, 1.0, # Y axis direction?
            1.0, 1.0, 1.0, # Z axis direction?
            1.0, 1.0, 1.0, # uninspected points
            1.0, 1.0, 1.0, # sun angle
            1.0, 1.0, 1.0, # priority vector
            1.0, # points score
            1.0, # dot product of uninspected points + position
            ]),
    }
    output_norms = {
        'deputy': np.array([0.001, 1., 0.001, 1., 0.001, 1.], dtype=np.float32),
    }

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
    corl_actions_order = [1, 3, 5, 0, 2, 4]
    termination = dict.fromkeys(deputies, False)
    truncation = dict.fromkeys(deputies, False)
    obs_array = []
    control_array = []
    info_array = []

    # Continue until done
    while not any(termination.values()) and not any(truncation.values()):
        corl_actions = {}
        reordered_actions = {}
        for agent in deputies:
            corl_actions[agent] = get_action(ort_sessions[agent], observations[agent], input_norms["deputy"], output_norms["deputy"])
            # reorder action space
            reordered_actions[agent] = corl_actions[agent][corl_actions_order]
            # remove normalization for comparison to CoRL data
            corl_actions[agent] = corl_actions[agent] / output_norms['deputy']
        observations, rewards, termination, truncation, infos = env.step(reordered_actions)
        obs_array.append(observations)
        control_array.append(corl_actions)
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
    # assert len(corl_rewards0) == len(reward_components_array)

    # check values
    for i, gym_step_action_dict in enumerate(control_array):
        if gym_step_action_dict['deputy_0'] is not None:
            assert np.allclose(corl_actions0[i], gym_step_action_dict['deputy_0'], rtol=1e-03, atol=1e-04)
        if gym_step_action_dict['deputy_1'] is not None:
            assert np.allclose(corl_actions1[i], gym_step_action_dict['deputy_1'], rtol=1e-03, atol=1e-04)
        if gym_step_action_dict['deputy_2'] is not None:
            assert np.allclose(corl_actions2[i], gym_step_action_dict['deputy_2'], rtol=1e-03, atol=1e-04)

    for i, gym_step_obs_dict in enumerate(obs_array):
        if gym_step_obs_dict['deputy_0'] is not None:
            assert np.allclose(corl_obs0[i], gym_step_obs_dict['deputy_0'], rtol=1e-02, atol=1e-03)
        if gym_step_obs_dict['deputy_1'] is not None:
            assert np.allclose(corl_obs1[i], gym_step_obs_dict['deputy_1'], rtol=1e-02, atol=1e-03)
        if gym_step_obs_dict['deputy_2'] is not None:
            assert np.allclose(corl_obs2[i], gym_step_obs_dict['deputy_2'], rtol=1e-02, atol=1e-03)

    for i, gym_step_rewards in enumerate(info_array):
        if gym_step_obs_dict['deputy_0'] is not None:
            corl_inspected_points = corl_rewards0[i]["ObservedPointsReward"]
            inspected_points = gym_step_rewards['deputy_0']['reward_components']['observed_points']
            assert corl_inspected_points == pytest.approx(inspected_points, rel=1e-04, abs=1e-10)
            corl_success = corl_rewards0[i]["SafeInspectionSuccessReward"]
            success = gym_step_rewards['deputy_0']['reward_components']["success"]
            assert corl_success == pytest.approx(success, rel=1e-04, abs=1e-10)
            corl_crash = corl_rewards0[i]["InspectionCrashReward"]
            crash = gym_step_rewards['deputy_0']['reward_components']['crash']
            assert corl_crash == pytest.approx(crash, rel=1e-04, abs=1e-10)
            # corl_max_dist = corl_rewards0[i]["MaxDistanceDoneReward"]
            # max_dist = gym_step_rewards['deputy_0']['reward_components']['max_distance']
            # assert corl_max_dist == pytest.approx(max_dist, rel=1e-04, abs=1e-10)
            # corl_live_timestep = corl_rewards0[i]["LiveTimestepReward"]
            # live_timestep = gym_step_rewards['deputy_0']['reward_components']['live_timestep']
            # assert corl_live_timestep == pytest.approx(live_timestep, rel=1e-04, abs=1e-10)
            # corl_facing_chief = corl_rewards0[i]["FacingChiefReward"]
            # facing_chief = gym_step_rewards['deputy_0']['reward_components']['facing_chief']
            # if corl_facing_chief != 0.0:
            #     assert corl_facing_chief == pytest.approx(facing_chief, rel=1e-04, abs=1e-10)
            corl_delta_v = corl_rewards0[i]["InspectionDeltaVReward"]
            delta_v = gym_step_rewards['deputy_0']['reward_components']['delta_v']
            assert corl_delta_v == pytest.approx(delta_v, rel=1e-03, abs=1e-10)
        
        if gym_step_obs_dict['deputy_1'] is not None:
            corl_inspected_points = corl_rewards1[i]["ObservedPointsReward"]
            inspected_points = gym_step_rewards['deputy_1']['reward_components']['observed_points']
            assert corl_inspected_points == pytest.approx(inspected_points, rel=1e-04, abs=1e-10)
            corl_success = corl_rewards1[i]["SafeInspectionSuccessReward"]
            success = gym_step_rewards['deputy_1']['reward_components']["success"]
            assert corl_success == pytest.approx(success, rel=1e-04, abs=1e-10)
            corl_crash = corl_rewards1[i]["InspectionCrashReward"]
            crash = gym_step_rewards['deputy_1']['reward_components']['crash']
            assert corl_crash == pytest.approx(crash, rel=1e-04, abs=1e-10)
            # corl_live_timestep = corl_rewards1[i]["MaxDistanceDoneReward"]
            # live_timestep = gym_step_rewards['deputy_1']['reward_components']['max_distance']
            # assert corl_live_timestep == pytest.approx(live_timestep, rel=1e-04, abs=1e-10)
            # corl_live_timestep = corl_rewards1[i]["LiveTimestepReward"]
            # live_timestep = gym_step_rewards['deputy_1']['reward_components']['live_timestep']
            # assert corl_live_timestep == pytest.approx(live_timestep, rel=1e-04, abs=1e-10)
            # corl_facing_chief = corl_rewards1[i]["FacingChiefReward"]
            # facing_chief = gym_step_rewards['deputy_1']['reward_components']['facing_chief']
            # if corl_facing_chief != 0.0:
            #     assert corl_facing_chief == pytest.approx(facing_chief, rel=1e-04, abs=1e-10)
            corl_delta_v = corl_rewards1[i]["InspectionDeltaVReward"]
            delta_v = gym_step_rewards['deputy_1']['reward_components']['delta_v']
            assert corl_delta_v == pytest.approx(delta_v, rel=1e-03, abs=1e-10)
        
        if gym_step_obs_dict['deputy_2'] is not None:
            corl_inspected_points = corl_rewards2[i]["ObservedPointsReward"]
            inspected_points = gym_step_rewards['deputy_2']['reward_components']['observed_points']
            assert corl_inspected_points == pytest.approx(inspected_points, rel=1e-04, abs=1e-10)
            corl_success = corl_rewards2[i]["SafeInspectionSuccessReward"]
            success = gym_step_rewards['deputy_2']['reward_components']["success"]
            assert corl_success == pytest.approx(success, rel=1e-04, abs=1e-10)
            corl_crash = corl_rewards2[i]["InspectionCrashReward"]
            crash = gym_step_rewards['deputy_2']['reward_components']['crash']
            assert corl_crash == pytest.approx(crash, rel=1e-04, abs=1e-10)
            # corl_live_timestep = corl_rewards2[i]["MaxDistanceDoneReward"]
            # live_timestep = gym_step_rewards['deputy_2']['reward_components']['max_distance']
            # assert corl_live_timestep == pytest.approx(live_timestep, rel=1e-04, abs=1e-10)
            # corl_live_timestep = corl_rewards2[i]["LiveTimestepReward"]
            # live_timestep = gym_step_rewards['deputy_2']['reward_components']['live_timestep']
            # assert corl_live_timestep == pytest.approx(live_timestep, rel=1e-04, abs=1e-10)
            # corl_facing_chief = corl_rewards2[i]["FacingChiefReward"]
            # facing_chief = gym_step_rewards['deputy_2']['reward_components']['facing_chief']
            # if corl_facing_chief != 0.0:
            #     assert corl_facing_chief == pytest.approx(facing_chief, rel=1e-04, abs=1e-10)
            corl_delta_v = corl_rewards2[i]["InspectionDeltaVReward"]
            delta_v = gym_step_rewards['deputy_2']['reward_components']['delta_v']
            assert corl_delta_v == pytest.approx(delta_v, rel=1e-03, abs=1e-10)
