import pytest
import pickle
import onnx # TODO: add onnx dependency
import onnxruntime as ort # TODO: add dependency
import numpy as np
from safe_autonomy_sims.gym.inspection.inspection_v0 import InspectionEnv
import safe_autonomy_simulation.sims.inspection as sim
from safe_autonomy_sims.simulators.initializers.cwh import CWH3DRadialWithSunInitializer
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
    corl_data_path = os.path.join(current_dir, 'translational_inspection_episode_data.pkl')
    with open(corl_data_path, 'rb') as f:
        data = pickle.load(f)
    return data


@pytest.fixture(name="initial_conditions")
def fixture_initial_conditions():
    ic = {
        "radius": 62.04303711979668,
        "azimuth_angle": 2.5928229798716957,
        "elevation_angle": 1.1217152607083294,
        "vel_mag": 0.2871053440330955,
        "vel_azimuth_angle": 4.88752815681357,
        "vel_elevation_angle": -1.1826284001971126,
        "sun_angle": 1.9078404144016834,
    }
    return ic


@pytest.fixture(name="onxx_model_path")
def fixture_onxx_model_path():
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, 'model.onnx')
    return path


@pytest.mark.system_test
def test_validate_inspection_gym_with_corl(corl_data, initial_conditions, onxx_model_path):
    # Dynamic env class definition to insert initial conditions

    # Gym uses different initializer logic than default CoRL
    config = {}
    corl_initializer = CWH3DRadialWithSunInitializer(config=config)
    initial_conditions_dict = corl_initializer.compute(
        radius=initial_conditions["radius"],
        azimuth_angle=initial_conditions["azimuth_angle"],
        elevation_angle=initial_conditions["elevation_angle"],
        vel_mag=initial_conditions["vel_mag"],
        vel_azimuth_angle=initial_conditions["vel_azimuth_angle"],
        vel_elevation_angle=initial_conditions["vel_elevation_angle"],
        sun_angle=initial_conditions["sun_angle"],
    )

    class TestInspectionEnv(InspectionEnv):
        def _init_sim(self):
            # Initialize spacecraft, sun, and simulator
            self.chief = sim.Target(
                name="chief",
                num_points=100,
                radius=10,
            )
            self.deputy = sim.Inspector(
                name="deputy",
                position=initial_conditions_dict["position"],
                velocity=initial_conditions_dict["velocity"],
                fov=np.pi,
                focal_length=1,
            )
            self.sun = sim.Sun(theta=initial_conditions_dict["sun_angle"])
            self.simulator = sim.InspectionSimulator(
                frame_rate=0.1,
                inspectors=[self.deputy],
                targets=[self.chief],
                sun=self.sun,
            )
    env = TestInspectionEnv()

    # Norms used with CoRL
    input_norms = {
        'deputy': np.array([
            1.0000e+02, 1.0000e+02, 1.0000e+02, # position
            0.5000e+00, 0.5000e+00, 0.5000e+00, # velocity
            1.0000e+02, # points
            1.0000e+00, 1.0e+0, 1.0e+0, # uninspected points
            1.0e+0 # sun angle
            ]),
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
    corl_obs_order = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 6]
    reordered_obs = observations[corl_obs_order] # first obs not recording in CoRL's EpisodeArtifact
    termination = False
    truncation = False
    obs_array = []
    control_array = []
    reward_components_array = []

    # Continue until done
    while not termination and not truncation:
        agent = 'deputy'
        action = get_action(ort_sess_deputy, reordered_obs, input_norms[agent], output_norms[agent])
        observations, rewards, termination, truncation, infos = env.step(action)
        # handle obs element order mismatch
        reordered_obs = observations[corl_obs_order]
        obs_array.append(reordered_obs)
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
        assert np.allclose(corl_step_action, control_array[i], rtol=1e-04, atol=1e-08)

    for i, corl_step_obs in enumerate(corl_obs):
        assert np.allclose(corl_step_obs, obs_array[i], rtol=1e-03, atol=1e-05)

    for i, corl_step_rewards in enumerate(corl_rewards):
        corl_delta_v = corl_step_rewards["InspectionDeltaVReward"]
        delta_v = reward_components_array[i]['delta_v']
        assert corl_delta_v == pytest.approx(delta_v, rel=1e-04, abs=1e-10)
        corl_inspected_points = corl_step_rewards["ObservedPointsReward"]
        inspected_points = reward_components_array[i]['observed_points']
        assert corl_inspected_points == pytest.approx(inspected_points, rel=1e-04, abs=1e-10)
        corl_success = corl_step_rewards["InspectionSuccessReward"]
        success = reward_components_array[i]["success"]
        assert corl_success == pytest.approx(success, rel=1e-04, abs=1e-10)
        corl_crash = corl_step_rewards["InspectionCrashReward"]
        crash = reward_components_array[i]['crash']
        assert corl_crash == pytest.approx(crash, rel=1e-04, abs=1e-10)
