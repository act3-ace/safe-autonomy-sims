"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Run an evaluation episode and make animation for the inspection environment
"""
from safe_autonomy_sims.evaluation.animation.inspection_animation import InspectionAnimation


if __name__ == '__main__':
    # Pass path to checkpoint file
    animation = InspectionAnimation(checkpoint_path='/tmp/safe-autonomy-sims/output/tune/TRANSLATIONAL-INSPECTION/TRANSLATIONAL-INSPECTION-PPO_CorlMultiAgentEnv_4d6e5_00000_0_2024-03-19_11-56-52/checkpoint_000002',
                                    expr_config_path='configs/translational-inspection/experiment.yml',
                                    task_config_path='configs/translational-inspection/task.yml',
                                    parameters_path='configs/translational-inspection/parameters.yml')
    # Give filetype: 'png' for image, 'mp4' or 'gif' for animation
    animation.make_animation(filetype='png')
