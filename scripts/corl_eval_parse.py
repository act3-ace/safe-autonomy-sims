import pickle
import numpy as np


# set up
# episode_artifact_path = "/tmp/safe-autonomy-sims/docking_validation_testing/test_case_0/2024-11-05_20-42-31_episode_artifact.pkl"
# episode_artifact_path = "/tmp/safe-autonomy-sims/inspection_v0_validation_testing/test_case_0/2024-11-06_13-57-00_episode_artifact.pkl"
# episode_artifact_path = "/tmp/safe-autonomy-sims/inspection_v0_validation_testing_r/test_case_0/2024-11-06_18-33-44_episode_artifact.pkl"
# episode_artifact_path = "/tmp/safe-autonomy-sims/weighted_inspection_v0_validation_testing/test_case_0/2024-11-07_16-56-51_episode_artifact.pkl"
# episode_artifact_path = "/tmp/safe-autonomy-sims/multiagent_translational_inspection_v0_validation_testing/test_case_0/2024-11-13_17-15-42_episode_artifact.pkl"
# episode_artifact_path = "/tmp/safe-autonomy-sims/multiagent_translational_inspection_v0_validation_testing_2/test_case_0/2024-11-14_14-22-00_episode_artifact.pkl"
# episode_artifact_path = "/tmp/safe-autonomy-sims/multiagent_weighted_translational_inspection_v0_validation_testing/test_case_0/2024-11-15_14-58-58_episode_artifact.pkl"
episode_artifact_path = "/tmp/safe-autonomy-sims/multiagent_weighted_six_dof_inspection_v0_validation_testing/test_case_0/2024-11-15_16-51-51_episode_artifact.pkl"


# open
with open(episode_artifact_path, 'rb') as file:
    ea = pickle.load(file)

# store ICs, obs, actions, reward components
corl_episode_info = {
    "IC": None,
    "obs0": [],
    "obs1": [],
    "obs2": [],
    "actions0": [],
    "actions1": [],
    "actions2": [],
    "rewards0": [],
    "rewards1": [],
    "rewards2": [],
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


# # weighted inspection v0
# for step_info in ea.steps:
#     # Collect obs
#     obs_dict = step_info.agents['blue0_ctrl'].observations
#     position = obs_dict["Obs_Sensor_Position"]["direct_observation"].value
#     velocity = obs_dict["Obs_Sensor_Velocity"]["direct_observation"].value
#     points = obs_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
#     uninspected_points = obs_dict["Obs_Sensor_UninspectedPoints"]["direct_observation"].value
#     sun_angle = obs_dict["Obs_Sensor_SunAngle"]["direct_observation"].value
#     priority_vec = obs_dict["Obs_Sensor_PriorityVector"]["direct_observation"].value
#     points_score = obs_dict["Obs_Sensor_InspectedPointsScore"]["direct_observation"].value

#     obs = np.concatenate((position, velocity, points, uninspected_points, sun_angle, priority_vec, points_score))
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


# # multiagent inspection v0
# for step_info in ea.steps:
#     # Collect obs
#     obs0_dict = step_info.agents['blue0_ctrl'].observations
#     if obs0_dict:
#         position = obs0_dict["Obs_Sensor_Position"]["direct_observation"].value
#         velocity = obs0_dict["Obs_Sensor_Velocity"]["direct_observation"].value
#         points = obs0_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
#         uninspected_points = obs0_dict["Obs_Sensor_UninspectedPoints"]["direct_observation"].value
#         sun_angle = obs0_dict["Obs_Sensor_SunAngle"]["direct_observation"].value

#         obs0 = np.concatenate((position, velocity, points, uninspected_points, sun_angle))
#     else:
#         obs0 = None
#     corl_episode_info["obs0"].append(obs0)

#     obs1_dict = step_info.agents['blue1_ctrl'].observations
#     if obs1_dict:
#         position = obs1_dict["Obs_Sensor_Position"]["direct_observation"].value
#         velocity = obs1_dict["Obs_Sensor_Velocity"]["direct_observation"].value
#         points = obs1_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
#         uninspected_points = obs1_dict["Obs_Sensor_UninspectedPoints"]["direct_observation"].value
#         sun_angle = obs1_dict["Obs_Sensor_SunAngle"]["direct_observation"].value

#         obs1 = np.concatenate((position, velocity, points, uninspected_points, sun_angle))
#     else:
#         obs1 = None
#     corl_episode_info["obs1"].append(obs1)

#     obs2_dict = step_info.agents['blue2_ctrl'].observations
#     if obs2_dict:
#         position = obs2_dict["Obs_Sensor_Position"]["direct_observation"].value
#         velocity = obs2_dict["Obs_Sensor_Velocity"]["direct_observation"].value
#         points = obs2_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
#         uninspected_points = obs2_dict["Obs_Sensor_UninspectedPoints"]["direct_observation"].value
#         sun_angle = obs2_dict["Obs_Sensor_SunAngle"]["direct_observation"].value

#         obs2 = np.concatenate((position, velocity, points, uninspected_points, sun_angle))
#     else:
#         obs2 = None
#     corl_episode_info["obs2"].append(obs2)

#     # Collect actions
#     actions0_dict = step_info.agents['blue0_ctrl'].actions
#     if actions0_dict:
#         x_thrust = actions0_dict["RTAModule.x_thrust"]
#         y_thrust = actions0_dict["RTAModule.y_thrust"]
#         z_thrust = actions0_dict["RTAModule.z_thrust"]

#         actions0 = np.concatenate((x_thrust, y_thrust, z_thrust))
#     else:
#         actions0 = None
#     corl_episode_info["actions0"].append(actions0)

#     actions1_dict = step_info.agents['blue1_ctrl'].actions
#     if actions1_dict:
#         x_thrust = actions1_dict["RTAModule.x_thrust"]
#         y_thrust = actions1_dict["RTAModule.y_thrust"]
#         z_thrust = actions1_dict["RTAModule.z_thrust"]

#         actions1 = np.concatenate((x_thrust, y_thrust, z_thrust))
#     else:
#         actions1 = None
#     corl_episode_info["actions1"].append(actions1)

#     actions2_dict = step_info.agents['blue2_ctrl'].actions
#     if actions2_dict:
#         x_thrust = actions2_dict["RTAModule.x_thrust"]
#         y_thrust = actions2_dict["RTAModule.y_thrust"]
#         z_thrust = actions2_dict["RTAModule.z_thrust"]

#         actions2 = np.concatenate((x_thrust, y_thrust, z_thrust))
#     else:
#         actions2 = None
#     corl_episode_info["actions2"].append(actions2)

#     # Collect rewards
#     rew0_dict = step_info.agents['blue0_ctrl'].rewards
#     corl_episode_info["rewards0"].append(rew0_dict)

#     # Collect rewards
#     rew1_dict = step_info.agents['blue1_ctrl'].rewards
#     corl_episode_info["rewards1"].append(rew1_dict)

#     # Collect rewards
#     rew2_dict = step_info.agents['blue2_ctrl'].rewards
#     corl_episode_info["rewards2"].append(rew2_dict)


# multiagent inspection v0
# for step_info in ea.steps:
#     # Collect obs
#     obs0_dict = step_info.agents['blue0_ctrl'].observations
#     if obs0_dict:
#         position = obs0_dict["Obs_Sensor_Position"]["direct_observation"].value
#         velocity = obs0_dict["Obs_Sensor_Velocity"]["direct_observation"].value
#         points = obs0_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
#         uninspected_points = obs0_dict["Obs_Sensor_UninspectedPoints"]["direct_observation"].value
#         sun_angle = obs0_dict["Obs_Sensor_SunAngle"]["direct_observation"].value
#         priority_vec = obs0_dict["Obs_Sensor_PriorityVector"]["direct_observation"].value
#         points_score = obs0_dict["Obs_Sensor_InspectedPointsScore"]["direct_observation"].value

#         obs0 = np.concatenate((position, velocity, points, uninspected_points, sun_angle, priority_vec, points_score))
#     else:
#         obs0 = None
#     corl_episode_info["obs0"].append(obs0)

#     obs1_dict = step_info.agents['blue1_ctrl'].observations
#     if obs1_dict:
#         position = obs1_dict["Obs_Sensor_Position"]["direct_observation"].value
#         velocity = obs1_dict["Obs_Sensor_Velocity"]["direct_observation"].value
#         points = obs1_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
#         uninspected_points = obs1_dict["Obs_Sensor_UninspectedPoints"]["direct_observation"].value
#         sun_angle = obs1_dict["Obs_Sensor_SunAngle"]["direct_observation"].value
#         priority_vec = obs1_dict["Obs_Sensor_PriorityVector"]["direct_observation"].value
#         points_score = obs1_dict["Obs_Sensor_InspectedPointsScore"]["direct_observation"].value

#         obs1 = np.concatenate((position, velocity, points, uninspected_points, sun_angle, priority_vec, points_score))
#     else:
#         obs1 = None
#     corl_episode_info["obs1"].append(obs1)

#     obs2_dict = step_info.agents['blue2_ctrl'].observations
#     if obs2_dict:
#         position = obs2_dict["Obs_Sensor_Position"]["direct_observation"].value
#         velocity = obs2_dict["Obs_Sensor_Velocity"]["direct_observation"].value
#         points = obs2_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
#         uninspected_points = obs2_dict["Obs_Sensor_UninspectedPoints"]["direct_observation"].value
#         sun_angle = obs2_dict["Obs_Sensor_SunAngle"]["direct_observation"].value
#         priority_vec = obs2_dict["Obs_Sensor_PriorityVector"]["direct_observation"].value
#         points_score = obs2_dict["Obs_Sensor_InspectedPointsScore"]["direct_observation"].value

#         obs2 = np.concatenate((position, velocity, points, uninspected_points, sun_angle, priority_vec, points_score))
#     else:
#         obs2 = None
#     corl_episode_info["obs2"].append(obs2)


#     # Collect actions
#     actions0_dict = step_info.agents['blue0_ctrl'].actions
#     if actions0_dict:
#         x_thrust = actions0_dict["RTAModule.x_thrust"]
#         y_thrust = actions0_dict["RTAModule.y_thrust"]
#         z_thrust = actions0_dict["RTAModule.z_thrust"]

#         actions0 = np.concatenate((x_thrust, y_thrust, z_thrust))
#     else:
#         actions0 = None
#     corl_episode_info["actions0"].append(actions0)

#     actions1_dict = step_info.agents['blue1_ctrl'].actions
#     if actions1_dict:
#         x_thrust = actions1_dict["RTAModule.x_thrust"]
#         y_thrust = actions1_dict["RTAModule.y_thrust"]
#         z_thrust = actions1_dict["RTAModule.z_thrust"]

#         actions1 = np.concatenate((x_thrust, y_thrust, z_thrust))
#     else:
#         actions1 = None
#     corl_episode_info["actions1"].append(actions1)

#     actions2_dict = step_info.agents['blue2_ctrl'].actions
#     if actions2_dict:
#         x_thrust = actions2_dict["RTAModule.x_thrust"]
#         y_thrust = actions2_dict["RTAModule.y_thrust"]
#         z_thrust = actions2_dict["RTAModule.z_thrust"]

#         actions2 = np.concatenate((x_thrust, y_thrust, z_thrust))
#     else:
#         actions2 = None
#     corl_episode_info["actions2"].append(actions2)


#     # Collect rewards
#     rew0_dict = step_info.agents['blue0_ctrl'].rewards
#     corl_episode_info["rewards0"].append(rew0_dict)

#     # Collect rewards
#     rew1_dict = step_info.agents['blue1_ctrl'].rewards
#     corl_episode_info["rewards1"].append(rew1_dict)

#     # Collect rewards
#     rew2_dict = step_info.agents['blue2_ctrl'].rewards
#     corl_episode_info["rewards2"].append(rew2_dict)


# multiagent six dof inspection v0
for step_info in ea.steps:
    # Collect obs
    obs0_dict = step_info.agents['blue0_ctrl'].observations
    if obs0_dict:
        position = obs0_dict["Obs_Sensor_Position"]["direct_observation"].value
        velocity = obs0_dict["Obs_Sensor_Velocity"]["direct_observation"].value
        points = obs0_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
        uninspected_points = obs0_dict["Obs_Sensor_UninspectedPoints"]["direct_observation"].value
        sun_angle = obs0_dict["Obs_Sensor_SunAngle"]["direct_observation"].value
        priority_vec = obs0_dict["Obs_Sensor_PriorityVector"]["direct_observation"].value
        points_score = obs0_dict["Obs_Sensor_InspectedPointsScore"]["direct_observation"].value
        quat = obs0_dict["Obs_Sensor_Quaternion"]["direct_observation"].value
        angular_vel = obs0_dict["Obs_Sensor_AngularVelocity"]["direct_observation"].value

        obs0 = np.concatenate((position, velocity, points, uninspected_points, sun_angle, priority_vec, points_score, quat, angular_vel))
    else:
        obs0 = None
    corl_episode_info["obs0"].append(obs0)

    obs1_dict = step_info.agents['blue1_ctrl'].observations
    if obs1_dict:
        position = obs1_dict["Obs_Sensor_Position"]["direct_observation"].value
        velocity = obs1_dict["Obs_Sensor_Velocity"]["direct_observation"].value
        points = obs1_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
        uninspected_points = obs1_dict["Obs_Sensor_UninspectedPoints"]["direct_observation"].value
        sun_angle = obs1_dict["Obs_Sensor_SunAngle"]["direct_observation"].value
        priority_vec = obs1_dict["Obs_Sensor_PriorityVector"]["direct_observation"].value
        points_score = obs1_dict["Obs_Sensor_InspectedPointsScore"]["direct_observation"].value
        quat = obs1_dict["Obs_Sensor_Quaternion"]["direct_observation"].value
        angular_vel = obs1_dict["Obs_Sensor_AngularVelocity"]["direct_observation"].value

        obs1 = np.concatenate((position, velocity, points, uninspected_points, sun_angle, priority_vec, points_score, quat, angular_vel))
    else:
        obs1 = None
    corl_episode_info["obs1"].append(obs1)

    obs2_dict = step_info.agents['blue2_ctrl'].observations
    if obs2_dict:
        position = obs2_dict["Obs_Sensor_Position"]["direct_observation"].value
        velocity = obs2_dict["Obs_Sensor_Velocity"]["direct_observation"].value
        points = obs2_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
        uninspected_points = obs2_dict["Obs_Sensor_UninspectedPoints"]["direct_observation"].value
        sun_angle = obs2_dict["Obs_Sensor_SunAngle"]["direct_observation"].value
        priority_vec = obs2_dict["Obs_Sensor_PriorityVector"]["direct_observation"].value
        points_score = obs2_dict["Obs_Sensor_InspectedPointsScore"]["direct_observation"].value
        quat = obs2_dict["Obs_Sensor_Quaternion"]["direct_observation"].value
        angular_vel = obs2_dict["Obs_Sensor_AngularVelocity"]["direct_observation"].value

        obs2 = np.concatenate((position, velocity, points, uninspected_points, sun_angle, priority_vec, points_score, quat, angular_vel))
    else:
        obs2 = None
    corl_episode_info["obs2"].append(obs2)


    # Collect actions
    actions0_dict = step_info.agents['blue0_ctrl'].actions
    if actions0_dict:
        x_moment = actions0_dict["X Moment_X_moment"]
        x_thrust = actions0_dict["X Thrust_X_thrust"]
        Y_moment = actions0_dict["Y Moment_Y_moment"]
        Y_thrust = actions0_dict["Y Thrust_Y_thrust"]
        Z_moment = actions0_dict["Z Moment_Z_moment"]
        Z_thrust = actions0_dict["Z Thrust_Z_thrust"]

        actions0 = np.concatenate((x_moment, x_thrust,Y_moment, Y_thrust, Z_moment, Z_thrust))
    else:
        actions0 = None
    corl_episode_info["actions0"].append(actions0)

    actions1_dict = step_info.agents['blue1_ctrl'].actions
    if actions1_dict:
        x_moment = actions1_dict["X Moment_X_moment"]
        x_thrust = actions1_dict["X Thrust_X_thrust"]
        Y_moment = actions1_dict["Y Moment_Y_moment"]
        Y_thrust = actions1_dict["Y Thrust_Y_thrust"]
        Z_moment = actions1_dict["Z Moment_Z_moment"]
        Z_thrust = actions1_dict["Z Thrust_Z_thrust"]

        actions1 = np.concatenate((x_moment, x_thrust, Y_moment, Y_thrust, Z_moment, Z_thrust))
    else:
        actions1 = None
    corl_episode_info["actions1"].append(actions1)

    actions2_dict = step_info.agents['blue2_ctrl'].actions
    if actions2_dict:
        x_moment = actions2_dict["X Moment_X_moment"]
        x_thrust = actions2_dict["X Thrust_X_thrust"]
        Y_moment = actions2_dict["Y Moment_Y_moment"]
        Y_thrust = actions2_dict["Y Thrust_Y_thrust"]
        Z_moment = actions2_dict["Z Moment_Z_moment"]
        Z_thrust = actions2_dict["Z Thrust_Z_thrust"]

        actions2 = np.concatenate((x_moment, x_thrust, Y_moment, Y_thrust, Z_moment, Z_thrust))
    else:
        actions2 = None
    corl_episode_info["actions2"].append(actions2)


    # Collect rewards
    rew0_dict = step_info.agents['blue0_ctrl'].rewards
    corl_episode_info["rewards0"].append(rew0_dict)

    # Collect rewards
    rew1_dict = step_info.agents['blue1_ctrl'].rewards
    corl_episode_info["rewards1"].append(rew1_dict)

    # Collect rewards
    rew2_dict = step_info.agents['blue2_ctrl'].rewards
    corl_episode_info["rewards2"].append(rew2_dict)


# store dict in pickle for test for now
with open('multiagent_weighted_six_dof_inspection_v0_episode_data.pkl', 'wb') as file:
    pickle.dump(corl_episode_info, file)

