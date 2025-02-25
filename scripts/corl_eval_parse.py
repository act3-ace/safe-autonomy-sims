"""
This module parses evaluation data (EpisodeArtifacts) from safe-autonomy-sims CoRL tasks and saves
observations, actions, rewards, and initial conditions to disk. This module is used to generate
data used in gymnasium and pettingzoo environment validation tests.

Author: John McCarroll
"""

import pickle
import numpy as np


# Define episode artifact path
episode_artifact_path = "/absolute/path/to/test_case_0/<date-time>_episode_artifact.pkl"

# Load object
with open(episode_artifact_path, 'rb') as file:
    ea = pickle.load(file)


def parse_multi_docking(episode_artifact):
    # store ICs, obs, actions, reward components
    corl_episode_info = {
        "IC": None,
        "obs": [],
        "actions": [],
        "rewards": [],
    }

    corl_episode_info["IC"] = episode_artifact.initial_state

    # docking env data collection
    for step_info in episode_artifact.steps:
        # Collect obs
        obs = {}
        actions = {}
        rew_dict = {}
        for k in step_info.agents.keys():
            obs_dict = step_info.agents[k].observations
            position = obs_dict["Obs_Sensor_Position"]["direct_observation"].value
            velocity = obs_dict["Obs_Sensor_Velocity"]["direct_observation"].value
            vel_mag = obs_dict["Obs_Sensor_Velocity_Magnitude"]["mag"].value
            vel_limit = obs_dict["VelocityLimitGlue_VelocityLimit"]["direct_observation"].value

            obs[k] = np.concatenate((position, velocity, vel_mag, vel_limit))

            # Collect actions
            actions_dict = step_info.agents[k].actions
            x_thrust = actions_dict["X Thrust_X_thrust"]
            y_thrust = actions_dict["Y Thrust_Y_thrust"]
            z_thrust = actions_dict["Z Thrust_Z_thrust"]

            actions[k] = np.concatenate((x_thrust, y_thrust, z_thrust))

            # Collect rewards
            rew_dict[k] = step_info.agents[k].rewards

        corl_episode_info["obs"].append(obs)
        corl_episode_info["actions"].append(actions)
        corl_episode_info["rewards"].append(rew_dict)

    return corl_episode_info

def parse_inspection(episode_artifact):
    # store ICs, obs, actions, reward components
    corl_episode_info = {
        "IC": None,
        "obs": [],
        "actions": [],
        "rewards": [],
    }

    corl_episode_info["IC"] = episode_artifact.initial_state

    # inspection_v0 env data collection
    for step_info in ea.steps:
        # Collect obs
        obs_dict = step_info.agents['blue0_ctrl'].observations
        position = obs_dict["Obs_Sensor_Position"]["direct_observation"].value
        velocity = obs_dict["Obs_Sensor_Velocity"]["direct_observation"].value
        points = obs_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
        uninspected_points = obs_dict["Obs_Sensor_UninspectedPoints"]["direct_observation"].value
        sun_angle = obs_dict["Obs_Sensor_SunAngle"]["direct_observation"].value

        obs = np.concatenate((position, velocity, points, uninspected_points, sun_angle))
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

    return corl_episode_info

def parse_weighted_inspection(episode_artifact):
    # store ICs, obs, actions, reward components
    corl_episode_info = {
        "IC": None,
        "obs": [],
        "actions": [],
        "rewards": [],
    }

    corl_episode_info["IC"] = episode_artifact.initial_state

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

    return corl_episode_info

def parse_weighted_sixdof_inspection(episode_artifact):
    # store ICs, obs, actions, reward components
    corl_episode_info = {
        "IC": None,
        "obs": [],
        "actions": [],
        "rewards": [],
    }

    corl_episode_info["IC"] = episode_artifact.initial_state

    # weighted six dof inspection v0
    for step_info in ea.steps:
        # Collect obs
        sensors = step_info.platforms[0]['sensors']
        if sensors:
            position = sensors['Sensor_Position']['measurement'].value
            velocity = sensors['Sensor_Velocity']['measurement'].value
            angular_velocity = sensors['Sensor_AngularVelocity']['measurement'].value
            orientation = sensors['Sensor_Quaternion']['measurement'].value
            sun_angle = sensors['Sensor_SunAngle']['measurement'].value
            points = sensors['Sensor_InspectedPoints']['measurement'].value
            cluster = sensors['Sensor_UninspectedPoints']['measurement'].value
            priority_vector = sensors['Sensor_PriorityVector']['measurement'].value
            inspection_weight = sensors['Sensor_InspectedPointsScore']['measurement'].value

            obs0 = np.concatenate((
                position, 
                velocity, 
                angular_velocity, 
                orientation, 
                sun_angle, 
                points, 
                cluster,
                priority_vector,
                inspection_weight,
                ))
        else:
            obs0 = None
        corl_episode_info["obs"].append(obs0)

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
        corl_episode_info["actions"].append(actions0)

        # Collect rewards
        rew0_dict = step_info.agents['blue0_ctrl'].rewards
        corl_episode_info["rewards"].append(rew0_dict)

    return corl_episode_info

def parse_multiagent_inspection(episode_artifact):
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

    corl_episode_info["IC"] = episode_artifact.initial_state

    # multiagent inspection v0
    for step_info in ea.steps:
        # Collect obs
        obs0_dict = step_info.agents['blue0_ctrl'].observations
        if obs0_dict:
            position = obs0_dict["Obs_Sensor_Position"]["direct_observation"].value
            velocity = obs0_dict["Obs_Sensor_Velocity"]["direct_observation"].value
            points = obs0_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
            uninspected_points = obs0_dict["Obs_Sensor_UninspectedPoints"]["direct_observation"].value
            sun_angle = obs0_dict["Obs_Sensor_SunAngle"]["direct_observation"].value

            obs0 = np.concatenate((position, velocity, points, uninspected_points, sun_angle))
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

            obs1 = np.concatenate((position, velocity, points, uninspected_points, sun_angle))
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

            obs2 = np.concatenate((position, velocity, points, uninspected_points, sun_angle))
        else:
            obs2 = None
        corl_episode_info["obs2"].append(obs2)

        # Collect actions
        actions0_dict = step_info.agents['blue0_ctrl'].actions
        if actions0_dict:
            x_thrust = actions0_dict["RTAModule.x_thrust"]
            y_thrust = actions0_dict["RTAModule.y_thrust"]
            z_thrust = actions0_dict["RTAModule.z_thrust"]

            actions0 = np.concatenate((x_thrust, y_thrust, z_thrust))
        else:
            actions0 = None
        corl_episode_info["actions0"].append(actions0)

        actions1_dict = step_info.agents['blue1_ctrl'].actions
        if actions1_dict:
            x_thrust = actions1_dict["RTAModule.x_thrust"]
            y_thrust = actions1_dict["RTAModule.y_thrust"]
            z_thrust = actions1_dict["RTAModule.z_thrust"]

            actions1 = np.concatenate((x_thrust, y_thrust, z_thrust))
        else:
            actions1 = None
        corl_episode_info["actions1"].append(actions1)

        actions2_dict = step_info.agents['blue2_ctrl'].actions
        if actions2_dict:
            x_thrust = actions2_dict["RTAModule.x_thrust"]
            y_thrust = actions2_dict["RTAModule.y_thrust"]
            z_thrust = actions2_dict["RTAModule.z_thrust"]

            actions2 = np.concatenate((x_thrust, y_thrust, z_thrust))
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

    return corl_episode_info

def parse_weighted_multiagent_inspection(episode_artifact):
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

    corl_episode_info["IC"] = episode_artifact.initial_state

    # multiagent inspection v0
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

            obs0 = np.concatenate((position, velocity, points, uninspected_points, sun_angle, priority_vec, points_score))
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

            obs1 = np.concatenate((position, velocity, points, uninspected_points, sun_angle, priority_vec, points_score))
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

            obs2 = np.concatenate((position, velocity, points, uninspected_points, sun_angle, priority_vec, points_score))
        else:
            obs2 = None
        corl_episode_info["obs2"].append(obs2)


        # Collect actions
        actions0_dict = step_info.agents['blue0_ctrl'].actions
        if actions0_dict:
            x_thrust = actions0_dict["RTAModule.x_thrust"]
            y_thrust = actions0_dict["RTAModule.y_thrust"]
            z_thrust = actions0_dict["RTAModule.z_thrust"]

            actions0 = np.concatenate((x_thrust, y_thrust, z_thrust))
        else:
            actions0 = None
        corl_episode_info["actions0"].append(actions0)

        actions1_dict = step_info.agents['blue1_ctrl'].actions
        if actions1_dict:
            x_thrust = actions1_dict["RTAModule.x_thrust"]
            y_thrust = actions1_dict["RTAModule.y_thrust"]
            z_thrust = actions1_dict["RTAModule.z_thrust"]

            actions1 = np.concatenate((x_thrust, y_thrust, z_thrust))
        else:
            actions1 = None
        corl_episode_info["actions1"].append(actions1)

        actions2_dict = step_info.agents['blue2_ctrl'].actions
        if actions2_dict:
            x_thrust = actions2_dict["RTAModule.x_thrust"]
            y_thrust = actions2_dict["RTAModule.y_thrust"]
            z_thrust = actions2_dict["RTAModule.z_thrust"]

            actions2 = np.concatenate((x_thrust, y_thrust, z_thrust))
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


    return corl_episode_info

def parse_weighted_multiagent_sixdof_inspection(episode_artifact):
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

    corl_episode_info["IC"] = episode_artifact.initial_state

    # six dof inspection v0
    for step_info in ea.steps:
        # Collect obs
        obs0_dict = step_info.agents['blue0_ctrl'].observations
        if obs0_dict:
            relative_chief_pos_local_ref = obs0_dict["Obs_Sensor_RelativeChiefPosition_Local_Ref"]["direct_observation"].value
            relative_chief_pos_local_ref_mag = obs0_dict["Obs_Sensor_RelativeChiefPosition_Local_Ref_MagNorm3D"]["direct_observation"].value
            relative_chief_vel_local_ref = obs0_dict["Obs_Sensor_RelativeChiefVelocity_Local_Ref"]["direct_observation"].value
            relative_chief_vel_local_ref_mag = obs0_dict["Obs_Sensor_RelativeChiefVelocity_Local_Ref_MagNorm3D"]["direct_observation"].value
            # quaternion = obs0_dict["Obs_Sensor_Quaternion"]["direct_observation"].value
            angular_vel = obs0_dict["Obs_Sensor_AngularVelocity"]["direct_observation"].value
            orientation_unit_vector_local_ref = obs0_dict["Obs_Sensor_OrientationUnitVector_Local_Ref"]["direct_observation"].value
            # x_axis_local_ref = obs0_dict["Coordinate_Axis_Glue_X-Axis_Local_Ref"]["direct_observation"].value
            y_axis_local_ref = obs0_dict["Coordinate_Axis_Glue_Y-Axis_Local_Ref"]["direct_observation"].value
            z_axis_local_ref = obs0_dict["Coordinate_Axis_Glue_Z-Axis_Local_Ref"]["direct_observation"].value
            # orientation_unit_vector_local_ref_dotproduct_relative_chief_pos = obs0_dict["Obs_Sensor_OrientationUnitVector_Local_Ref_DotProduct_Obs_Sensor_RelativeChiefPosition"]["direct_observation"].value
            # inspected_points = obs0_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
            uninspected_points_local_ref = obs0_dict["Obs_Sensor_UninspectedPoints_Local_Ref"]["direct_observation"].value
            sun_angle_unit_vector_local_ref = obs0_dict["Obs_Sensor_SunAngle_AngleToUnitVector_Local_Ref"]["direct_observation"].value
            priority_vec_local_ref = obs0_dict["Obs_Sensor_PriorityVector_Local_Ref"]["direct_observation"].value
            inspected_points_score = obs0_dict["Obs_Sensor_InspectedPointsScore"]["direct_observation"].value
            uninspected_points_dotproduct_position = obs0_dict["Obs_Sensor_UninspectedPoints_DotProduct_Obs_Sensor_Position"]["direct_observation"].value

            obs0 = np.concatenate((
                relative_chief_pos_local_ref, 
                relative_chief_pos_local_ref_mag, 
                relative_chief_vel_local_ref, 
                relative_chief_vel_local_ref_mag, 
                # quaternion, 
                angular_vel, 
                orientation_unit_vector_local_ref, 
                # x_axis_local_ref, 
                y_axis_local_ref,
                z_axis_local_ref,
                # orientation_unit_vector_local_ref_dotproduct_relative_chief_pos,
                # inspected_points,
                uninspected_points_local_ref,
                sun_angle_unit_vector_local_ref,
                priority_vec_local_ref,
                inspected_points_score,
                uninspected_points_dotproduct_position,
                ))
        else:
            obs0 = None
        corl_episode_info["obs0"].append(obs0)

        # Collect obs
        obs1_dict = step_info.agents['blue1_ctrl'].observations
        if obs1_dict:
            relative_chief_pos_local_ref = obs1_dict["Obs_Sensor_RelativeChiefPosition_Local_Ref"]["direct_observation"].value
            relative_chief_pos_local_ref_mag = obs1_dict["Obs_Sensor_RelativeChiefPosition_Local_Ref_MagNorm3D"]["direct_observation"].value
            relative_chief_vel_local_ref = obs1_dict["Obs_Sensor_RelativeChiefVelocity_Local_Ref"]["direct_observation"].value
            relative_chief_vel_local_ref_mag = obs1_dict["Obs_Sensor_RelativeChiefVelocity_Local_Ref_MagNorm3D"]["direct_observation"].value
            # quaternion = obs1_dict["Obs_Sensor_Quaternion"]["direct_observation"].value
            angular_vel = obs1_dict["Obs_Sensor_AngularVelocity"]["direct_observation"].value
            orientation_unit_vector_local_ref = obs1_dict["Obs_Sensor_OrientationUnitVector_Local_Ref"]["direct_observation"].value
            # x_axis_local_ref = obs1_dict["Coordinate_Axis_Glue_X-Axis_Local_Ref"]["direct_observation"].value
            y_axis_local_ref = obs1_dict["Coordinate_Axis_Glue_Y-Axis_Local_Ref"]["direct_observation"].value
            z_axis_local_ref = obs1_dict["Coordinate_Axis_Glue_Z-Axis_Local_Ref"]["direct_observation"].value
            # orientation_unit_vector_local_ref_dotproduct_relative_chief_pos = obs1_dict["Obs_Sensor_OrientationUnitVector_Local_Ref_DotProduct_Obs_Sensor_RelativeChiefPosition"]["direct_observation"].value
            # inspected_points = obs1_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
            uninspected_points_local_ref = obs1_dict["Obs_Sensor_UninspectedPoints_Local_Ref"]["direct_observation"].value
            sun_angle_unit_vector_local_ref = obs1_dict["Obs_Sensor_SunAngle_AngleToUnitVector_Local_Ref"]["direct_observation"].value
            priority_vec_local_ref = obs1_dict["Obs_Sensor_PriorityVector_Local_Ref"]["direct_observation"].value
            inspected_points_score = obs1_dict["Obs_Sensor_InspectedPointsScore"]["direct_observation"].value
            uninspected_points_dotproduct_position = obs1_dict["Obs_Sensor_UninspectedPoints_DotProduct_Obs_Sensor_Position"]["direct_observation"].value

            obs1 = np.concatenate((
                relative_chief_pos_local_ref, 
                relative_chief_pos_local_ref_mag, 
                relative_chief_vel_local_ref, 
                relative_chief_vel_local_ref_mag, 
                # quaternion, 
                angular_vel, 
                orientation_unit_vector_local_ref, 
                # x_axis_local_ref, 
                y_axis_local_ref,
                z_axis_local_ref,
                # orientation_unit_vector_local_ref_dotproduct_relative_chief_pos,
                # inspected_points,
                uninspected_points_local_ref,
                sun_angle_unit_vector_local_ref,
                priority_vec_local_ref,
                inspected_points_score,
                uninspected_points_dotproduct_position,
                ))
        else:
            obs1 = None
        corl_episode_info["obs1"].append(obs1)

        # Collect obs
        obs2_dict = step_info.agents['blue2_ctrl'].observations
        if obs2_dict:
            relative_chief_pos_local_ref = obs2_dict["Obs_Sensor_RelativeChiefPosition_Local_Ref"]["direct_observation"].value
            relative_chief_pos_local_ref_mag = obs2_dict["Obs_Sensor_RelativeChiefPosition_Local_Ref_MagNorm3D"]["direct_observation"].value
            relative_chief_vel_local_ref = obs2_dict["Obs_Sensor_RelativeChiefVelocity_Local_Ref"]["direct_observation"].value
            relative_chief_vel_local_ref_mag = obs2_dict["Obs_Sensor_RelativeChiefVelocity_Local_Ref_MagNorm3D"]["direct_observation"].value
            # quaternion = obs2_dict["Obs_Sensor_Quaternion"]["direct_observation"].value
            angular_vel = obs2_dict["Obs_Sensor_AngularVelocity"]["direct_observation"].value
            orientation_unit_vector_local_ref = obs2_dict["Obs_Sensor_OrientationUnitVector_Local_Ref"]["direct_observation"].value
            # x_axis_local_ref = obs2_dict["Coordinate_Axis_Glue_X-Axis_Local_Ref"]["direct_observation"].value
            y_axis_local_ref = obs2_dict["Coordinate_Axis_Glue_Y-Axis_Local_Ref"]["direct_observation"].value
            z_axis_local_ref = obs2_dict["Coordinate_Axis_Glue_Z-Axis_Local_Ref"]["direct_observation"].value
            # orientation_unit_vector_local_ref_dotproduct_relative_chief_pos = obs2_dict["Obs_Sensor_OrientationUnitVector_Local_Ref_DotProduct_Obs_Sensor_RelativeChiefPosition"]["direct_observation"].value
            # inspected_points = obs2_dict["Obs_Sensor_InspectedPoints"]["direct_observation"].value
            uninspected_points_local_ref = obs2_dict["Obs_Sensor_UninspectedPoints_Local_Ref"]["direct_observation"].value
            sun_angle_unit_vector_local_ref = obs2_dict["Obs_Sensor_SunAngle_AngleToUnitVector_Local_Ref"]["direct_observation"].value
            priority_vec_local_ref = obs2_dict["Obs_Sensor_PriorityVector_Local_Ref"]["direct_observation"].value
            inspected_points_score = obs2_dict["Obs_Sensor_InspectedPointsScore"]["direct_observation"].value
            uninspected_points_dotproduct_position = obs2_dict["Obs_Sensor_UninspectedPoints_DotProduct_Obs_Sensor_Position"]["direct_observation"].value

            obs2 = np.concatenate((
                relative_chief_pos_local_ref, 
                relative_chief_pos_local_ref_mag, 
                relative_chief_vel_local_ref, 
                relative_chief_vel_local_ref_mag, 
                # quaternion, 
                angular_vel, 
                orientation_unit_vector_local_ref, 
                # x_axis_local_ref, 
                y_axis_local_ref,
                z_axis_local_ref,
                # orientation_unit_vector_local_ref_dotproduct_relative_chief_pos,
                # inspected_points,
                uninspected_points_local_ref,
                sun_angle_unit_vector_local_ref,
                priority_vec_local_ref,
                inspected_points_score,
                uninspected_points_dotproduct_position,
                ))
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

    return corl_episode_info


# Use parsing function appropriate to given task
corl_episode_info = parse_weighted_sixdof_inspection(ea)

# Store data to disk
with open('corl_evaluation_episode_data.pkl', 'wb') as file:
    pickle.dump(corl_episode_info, file)
