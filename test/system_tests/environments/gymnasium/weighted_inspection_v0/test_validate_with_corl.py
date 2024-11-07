import pytest
import pickle
import onnx # TODO: add onnx dependency
import onnxruntime as ort # TODO: add dependency
import numpy as np
from safe_autonomy_sims.gym.inspection.weighted_inspection_v0 import WeightedInspectionEnv
import safe_autonomy_simulation.sims.inspection as sim
from safe_autonomy_sims.simulators.initializers.cwh import CWH3DRadialWithSunInitializer
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
    corl_data_path = os.path.join(current_dir, 'weighted_inspection_v0_episode_data.pkl')
    with open(corl_data_path, 'rb') as f:
        data = pickle.load(f)
    return data


@pytest.fixture(name="initial_conditions")
def fixture_initial_conditions():
    ic = {
        "radius": 70.4688500686556,
        "azimuth_angle": 5.591767551579794,
        "elevation_angle": 1.3530600468515865,
        "vel_mag": 0.05530428436783764,
        "vel_azimuth_angle": 1.2092531635835906,
        "vel_elevation_angle": -1.1596684186528265,
        "sun_angle": 6.161208309022409,
        "priority_vector_azimuth_angle": 1.823676366683567,
        "priority_vector_elevation_angle": 0.8613711310531718,
    }
    return ic


@pytest.fixture(name="onxx_model_path")
def fixture_onxx_model_path():
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, 'model.onnx')
    return path


def test_validate_docking_gym_with_corl(corl_data, initial_conditions, onxx_model_path):
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

    # priority vector
    init_priority_vector = np.zeros((3,), dtype=np.float32)
    init_priority_vector[0] = np.cos(initial_conditions["priority_vector_azimuth_angle"]) * np.cos(initial_conditions["priority_vector_elevation_angle"])
    init_priority_vector[1] = np.sin(initial_conditions["priority_vector_azimuth_angle"]) * np.cos(initial_conditions["priority_vector_elevation_angle"])
    init_priority_vector[2] = np.sin(initial_conditions["priority_vector_elevation_angle"])

    class TestWeightedInspectionEnv(WeightedInspectionEnv):
        def _init_sim(self):
            # Initialize spacecraft, sun, and simulator
            priority_vector = init_priority_vector
            priority_vector /= np.linalg.norm(priority_vector)  # convert to unit vector
            self.chief = sim.Target(
                name="chief",
                num_points=100,
                radius=1,
                priority_vector=priority_vector,
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
    env = TestWeightedInspectionEnv()

    # Norms used with CoRL
    input_norms = {
        'deputy': np.array([
            1.0000e+02, 1.0000e+02, 1.0000e+02, # position
            0.5000e+00, 0.5000e+00, 0.5000e+00, # velocity
            1.0000e+02, # points
            1.0000e+00, 1.0e+0, 1.0e+0, # uninspected points
            1.0e+0, # sun angle
            1.0000e+00, 1.0e+0, 1.0e+0, # priority vector
            1.0e+0, # points score
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
    observations, infos = env.reset()
    corl_obs_order = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 6, 11, 12, 13, 14]
    reordered_obs = observations[corl_obs_order] # first obs not recording in CoRL's EpisodeArtifact
    print("FIRST")
    print(reordered_obs)
    termination = False
    truncation = False
    obs_array = []
    control_array = []
    reward_components_array = []

    # Continue until done
    while not termination and not truncation:
        # st = time.time()
        agent = 'deputy'
        action = get_action(ort_sess_deputy, reordered_obs, input_norms[agent], output_norms[agent])
        observations, rewards, termination, truncation, infos = env.step(action)
        # handle obs element order mismatch
        reordered_obs = observations[corl_obs_order]
        # print(f"Sim time: {env.simulator.sim_time}, step computation time: {time.time()-st}")
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
        print(i)
        assert np.allclose(corl_step_action, control_array[i], rtol=1e-04, atol=1e-08)

    for i, corl_step_obs in enumerate(corl_obs):
        print(i)
        print("CORL")
        print(corl_step_obs)
        print("GYM")
        print(obs_array[i])
        assert np.allclose(corl_step_obs, obs_array[i], rtol=1e-03, atol=1e-08)

    # for i, corl_step_rewards in enumerate(corl_rewards):
    #     # reward components are different*
    #     # cherry pick for now, lowest priority
    #     print(i)
    #     corl_delta_v = corl_step_rewards["DockingDeltaVReward"]
    #     delta_v = reward_components_array[i]['delta_v']
    #     assert corl_delta_v == delta_v
    #     corl_vel_const = corl_step_rewards["DockingVelocityConstraintReward"]
    #     vel_const = reward_components_array[i]['velocity_constraint']
    #     assert corl_vel_const == vel_const
    #     corl_success = corl_step_rewards["DockingSuccessReward"]
    #     success = reward_components_array[i]["success"]
    #     assert corl_success == success
    #     # corl_failure = corl_step_rewards["DockingFailureReward"]
    #     # failure = reward_components_array[i]['timeout'] + reward_components_array[i]['crash'] + reward_components_array[i]['out_of_bounds']
    #     # assert corl_failure == failure
