import pickle
import numpy as np


# set up
# episode_artifact_path = "/tmp/safe-autonomy-sims/docking_validation_testing/test_case_0/2024-11-05_20-42-31_episode_artifact.pkl"
# episode_artifact_path = "/tmp/safe-autonomy-sims/inspection_v0_validation_testing/test_case_0/2024-11-06_13-57-00_episode_artifact.pkl"
# episode_artifact_path = "/tmp/safe-autonomy-sims/inspection_v0_validation_testing_r/test_case_0/2024-11-06_18-33-44_episode_artifact.pkl"
episode_artifact_path = "/tmp/safe-autonomy-sims/weighted_inspection_v0_validation_testing/test_case_0/2024-11-07_16-56-51_episode_artifact.pkl"

# open
with open(episode_artifact_path, 'rb') as file:
    ea = pickle.load(file)

# store ICs, obs, actions, reward components
corl_episode_info = {
    "IC": None,
    "obs": [],
    "actions": [],
    "rewards": []
}

corl_episode_info["IC"] = ea.initial_state

# # docking env data collection
# for step_info in ea.steps:
#     # Collect obs
#     obs_dict = step_info.agents['blue0_ctrl'].observations
#     position = obs_dict["Obs_Sensor_Position"]["direct_observation"].value
#     velocity = obs_dict["Obs_Sensor_Velocity"]["direct_observation"].value
#     vel_mag = obs_dict["Obs_Sensor_Velocity_Magnitude"]["mag"].value
#     vel_limit = obs_dict["VelocityLimitGlue_VelocityLimit"]["direct_observation"].value

#     obs = np.concatenate((position, velocity, vel_mag, vel_limit))
#     corl_episode_info["obs"].append(obs)

#     # Collect actions
#     actions_dict = step_info.agents['blue0_ctrl'].actions
#     x_thrust = actions_dict["X Thrust_X_thrust"]
#     y_thrust = actions_dict["Y Thrust_Y_thrust"]
#     z_thrust = actions_dict["Z Thrust_Z_thrust"]

#     actions = np.concatenate((x_thrust, y_thrust, z_thrust))
#     corl_episode_info["actions"].append(actions)

#     # Collect rewards
#     rew_dict = step_info.agents['blue0_ctrl'].rewards
#     corl_episode_info["rewards"].append(rew_dict)

# # inspection_v0 env data collection
# for step_info in ea.steps:
#     # Collect obs
#     obs_dict = step_info.agents['blue0_ctrl'].observations
#     position = obs_dict["Obs_Sensor_Position"]["direct_observation"].value
#     velocity = obs_dict["Obs_Sensor_Velocity"]["direct_observation"].value
#     points = obs_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
#     uninspected_points = obs_dict["Obs_Sensor_UninspectedPoints"]["direct_observation"].value
#     sun_angle = obs_dict["Obs_Sensor_SunAngle"]["direct_observation"].value

#     obs = np.concatenate((position, velocity, points, uninspected_points, sun_angle))
#     corl_episode_info["obs"].append(obs)

#     # Collect actions
#     actions_dict = step_info.agents['blue0_ctrl'].actions
#     x_thrust = actions_dict["RTAModule.x_thrust"]
#     y_thrust = actions_dict["RTAModule.y_thrust"]
#     z_thrust = actions_dict["RTAModule.z_thrust"]

#     actions = np.concatenate((x_thrust, y_thrust, z_thrust))
#     corl_episode_info["actions"].append(actions)

#     # Collect rewards
#     rew_dict = step_info.agents['blue0_ctrl'].rewards
#     corl_episode_info["rewards"].append(rew_dict)

# weighted inspection v0
for step_info in ea.steps:
    # Collect obs
    obs_dict = step_info.agents['blue0_ctrl'].observations
    position = obs_dict["Obs_Sensor_Position"]["direct_observation"].value
    velocity = obs_dict["Obs_Sensor_Velocity"]["direct_observation"].value
    points = obs_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
    uninspected_points = obs_dict["Obs_Sensor_UninspectedPoints"]["direct_observation"].value
    sun_angle = obs_dict["Obs_Sensor_SunAngle"]["direct_observation"].value
    priority_vec = obs_dict["Obs_Sensor_PriorityVector"]["direct_observation"].value
    points_score = obs_dict["Obs_Sensor_InspectedPointsScore"]["direct_observation"].value

    obs = np.concatenate((position, velocity, points, uninspected_points, sun_angle, priority_vec, points_score))
    corl_episode_info["obs"].append(obs)

    # Collect actions
    actions_dict = step_info.agents['blue0_ctrl'].actions
    x_thrust = actions_dict["RTAModule.x_thrust"]
    y_thrust = actions_dict["RTAModule.y_thrust"]
    z_thrust = actions_dict["RTAModule.z_thrust"]

    actions = np.concatenate((x_thrust, y_thrust, z_thrust))
    corl_episode_info["actions"].append(actions)

    # Collect rewards
    rew_dict = step_info.agents['blue0_ctrl'].rewards
    corl_episode_info["rewards"].append(rew_dict)

# store dict in pickle for test for now
with open('weighted_inspection_v0_episode_data.pkl', 'wb') as file:
    pickle.dump(corl_episode_info, file)

