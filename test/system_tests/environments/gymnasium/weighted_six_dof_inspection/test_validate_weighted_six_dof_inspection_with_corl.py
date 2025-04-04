import pytest
import pickle
import onnx # TODO: add onnx dependency
import onnxruntime as ort # TODO: add dependency
import numpy as np
from safe_autonomy_sims.gym.inspection.sixdof_inspection_v0 import WeightedSixDofInspectionEnv
import safe_autonomy_simulation.sims.inspection as sim
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
    corl_data_path = os.path.join(current_dir, 'weighted_sixdof_inspection_episode_data.pkl')
    with open(corl_data_path, 'rb') as f:
        data = pickle.load(f)
    return data


@pytest.fixture(name="initial_conditions")
def fixture_initial_conditions():
    ic = {
        "sun_angle": 4.894076056971742,
        "priority_vector_azimuth_angle": 4.073586928538976,
        "priority_vector_elevation_angle": -1.0966449747644242,

        "angular_velocity": np.array([0.007485587676434926, -0.00784136861399348, 0.0011536854246057757]),
        "position": np.array([22.234393074496325, -48.08033288410433, 53.20879556201181]),
        "velocity": np.array([0.1726241925062788, 0.10827729728538717, -0.15299274875095167]),
    }
    return ic


@pytest.fixture(name="onxx_model_path")
def fixture_onxx_model_path():
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, 'model.onnx')
    return path

# TODO: Update the test to the new gymnasium observation space
#       This requires reconfiguring the observations in CoRL, recreating a policy, exporting the onnx model, rerunning CoRL eval, 
#       updating and rerunning the parse_corl_eval.py script, and updating this test to use the new test artifacts + observation space.
@pytest.mark.skip
def test_validate_sixdof_inspection_gym_with_corl(corl_data, initial_conditions, onxx_model_path):
    # priority vector
    init_priority_vector = np.zeros((3,), dtype=np.float32)
    init_priority_vector[0] = np.cos(initial_conditions["priority_vector_azimuth_angle"]) * np.cos(initial_conditions["priority_vector_elevation_angle"])
    init_priority_vector[1] = np.sin(initial_conditions["priority_vector_azimuth_angle"]) * np.cos(initial_conditions["priority_vector_elevation_angle"])
    init_priority_vector[2] = np.sin(initial_conditions["priority_vector_elevation_angle"])

    class TestWeightedSixDofInspectionEnv(WeightedSixDofInspectionEnv):
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
            self.deputy = sim.SixDOFInspector(
                name="deputy",
                position=initial_conditions["position"],
                velocity=initial_conditions["velocity"],
                angular_velocity=initial_conditions["angular_velocity"],
                orientation=corl_data["IC"]["blue0_ctrl"]["Obs_Sensor_Quaternion"]["direct_observation"].value,
                fov=1.0471975511965976,  # 60 degrees
                focal_length=9.6e-3,
            )
            self.sun = sim.Sun(theta=initial_conditions["sun_angle"])
            self.simulator = sim.InspectionSimulator(
                frame_rate=0.1,
                inspectors=[self.deputy],
                targets=[self.chief],
                sun=self.sun,
            )
    env = TestWeightedSixDofInspectionEnv()

    # Norms used with CoRL
    input_norms = {
        'deputy': np.array([
            100.0, 100.0, 100.0, # position
            0.5, 0.5, 0.5, # velocity
            0.05, 0.05, 0.05, # angular velocity
            1.0, 1.0, 1.0, 1.0, # quaternion
            1.0, # sun angle
            1.0, # points
            1.0, 1.0, 1.0, # uninspected points
            1.0, 1.0, 1.0, # priority vector
            1.0, # points score
            ]),
    }
    output_norms = {
        'deputy': np.array([0.001, 1., 0.001, 1., 0.001, 1.], dtype=np.float32),
    }

    # Load deputy onnx
    onnx_location_dep = onxx_model_path
    onnx_model_dep = onnx.load(onnx_location_dep)
    onnx.checker.check_model(onnx_model_dep)
    ort_sess_deputy = ort.InferenceSession(onnx_location_dep)

    # Reset env
    np.random.seed(3)
    observations, infos = env.reset()
    corl_actions_order = [1, 3, 5, 0, 2, 4]
    termination = False
    truncation = False
    obs_array = []
    control_array = []
    reward_components_array = []

    # Continue until done
    while not termination and not truncation:
        agent = 'deputy'
        corl_action = get_action(ort_sess_deputy, observations, input_norms[agent], output_norms[agent])
        reordered_action = corl_action[corl_actions_order]
        corl_action = corl_action / output_norms['deputy']
        observations, rewards, termination, truncation, infos = env.step(reordered_action)
        obs_array.append(observations)
        control_array.append(corl_action)
        reward_components_array.append(infos['reward_components'])

    # assert that obs, actions, and rewards aligns with data from corl environment
    corl_obs = corl_data["obs"]
    corl_actions = corl_data["actions"]
    corl_rewards = corl_data["rewards"]

    # check episode lengths
    # assert len(corl_obs) == len(obs_array)
    # assert len(corl_actions) == len(control_array)
    # assert len(corl_rewards) == len(reward_components_array)

    # check values
    for i, corl_step_action in enumerate(corl_actions):
        if i <= 350:
            assert np.allclose(corl_step_action, control_array[i], rtol=1e-03, atol=1e-06)
        else:
            # rounding error accumulation becomes too much
            break

    for i, corl_step_obs in enumerate(corl_obs):
        if i <= 100:
            assert np.allclose(corl_step_obs, obs_array[i], rtol=1e-02, atol=1e-5)
        elif i < 350:
            assert np.allclose(corl_step_obs, obs_array[i], rtol=5e-02, atol=1e-03)    
        else:
            # rounding error accumulation becomes too much
            break

    for i, corl_step_rewards in enumerate(corl_rewards):
        if i < 350:
            corl_inspected_points = corl_step_rewards["ObservedPointsReward"]
            inspected_points = reward_components_array[i]['observed_points']
            assert corl_inspected_points == pytest.approx(inspected_points, rel=1e-04, abs=1e-10)
            corl_success = corl_step_rewards["SafeInspectionSuccessReward"]
            success = reward_components_array[i]["success"]
            assert corl_success == pytest.approx(success, rel=1e-04, abs=1e-10)
            corl_crash = corl_step_rewards["InspectionCrashReward"]
            crash = reward_components_array[i]['crash']
            assert corl_crash == pytest.approx(crash, rel=1e-04, abs=1e-10)
            corl_live_timestep = corl_step_rewards["MaxDistanceDoneReward"]
            live_timestep = reward_components_array[i]['max_distance']
            assert corl_live_timestep == pytest.approx(live_timestep, rel=1e-04, abs=1e-10)
            corl_live_timestep = corl_step_rewards["LiveTimestepReward"]
            live_timestep = reward_components_array[i]['live_timestep']
            assert corl_live_timestep == pytest.approx(live_timestep, rel=1e-04, abs=1e-10)
            corl_facing_chief = corl_step_rewards["FacingChiefReward"]
            facing_chief = reward_components_array[i]['facing_chief']
            assert corl_facing_chief == pytest.approx(facing_chief, rel=1e-03, abs=1e-10)
            corl_delta_v = corl_step_rewards["InspectionDeltaVReward"]
            delta_v = reward_components_array[i]['delta_v']
            assert corl_delta_v == pytest.approx(delta_v, rel=1e-03, abs=1e-10)
        else:
            # rounding error accumulation becomes too much
            break
