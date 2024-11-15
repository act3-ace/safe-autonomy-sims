"""
file: export_onnx.py
author: Nate Hamilton

Description: This module has the function necessary for converting rllib checkpoints into onnx
             files. The onnx version is useful for sharing with a wide variety of users because it
             is a standardized format. Additionally, the netron tool can generate visuals from
             onnx files. These are useful for debugging and checking your architecture is what you
             designed. *See https://netron.app/
             As we start generating more types of checkpoints, we will expand the number of
             conversion functions.
"""
import os

from ray.rllib.algorithms.algorithm import Algorithm

from corl.experiments.rllib_experiment import RllibExperiment
from corl.train_rl import parse_corl_args, build_experiment


def rllib_chkpt_to_onnx(
    checkpoint_path: str, output_folder: str = "./onnx_model", policy_id: str = "blue0_ctrl", onnx_opset: int = 11
) -> None:
    """
    Parameters
    ----------
    checkpoint_path : str
        The path to the checkpoint from which each agent's policy will be loaded. The
        directory titled the agent's name will be filled in programmatically during this function.
        An example of the format of this string is:
            '/path/to/experiment/output/checkpoint_00005/policies/{}/policy_state.pkl'.
    output_folder : str
        The new folder that will be created to store the generated model.onnx
        Default: './onnx_model'
    policy_id : str
        A name associated with the policy in the checkpoint file. CoRL convention is to call it...
        Default: 'blue0_ctrl'
    onnx_opset : int
        The specific opset version to use. The list of options is available at
        https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions
        Default: 11
    """
    # Load the algorithm from the provided checkpoint
    agent = Algorithm.from_checkpoint(checkpoint_path, policy_ids=[policy_id])

    # Save the trained model as an onnx file
    agent.export_policy_model(export_dir=output_folder, policy_id=policy_id, onnx=onnx_opset)


def convert_policies_to_onnx(
    policy_dir: str,
    dest_dir: str,
):
    """
    This method converts a collection of rllib checkpoints into onnx models. If you only want to
    convert 1 rllib checkpoint, look above.

    Parameters
    ----------
    policy_dir : str
        The directory where all the rllib checkpoints are to be converted
    dest_dir : str
        The target directory where all the onnx models will be saved
    """
    # Get the list of policies to make
    policy_names = os.listdir(policy_dir)

    # Check for policy id, which will be the name of the subfolder under policies
    policy_id = os.listdir(os.path.join(policy_dir, policy_names[0], 'policies'))[0]

    # Convert them
    for policy_name in policy_names:
        rllib_chkpt_to_onnx(
            checkpoint_path=os.path.join(policy_dir, policy_name), output_folder=os.path.join(dest_dir, policy_name), policy_id=policy_id
        )


if __name__ == "__main__":
    # pylint:disable=line-too-long
    # chkpt_path = "/tmp/safe-autonomy-sims/output/tune/DOCKING/DOCKING-PPO_CorlMultiAgentEnv_3f697_00000_0_2024-10-30_16-33-40/checkpoint_000000"
    chkpt_path = "/tmp/safe-autonomy-sims/output/tune/MULTIAGENT-WEIGHTED-TRANSLATIONAL-INSPECTION/MULTIAGENT-WEIGHTED-TRANSLATIONAL-INSPECTION-test-PPO_CorlMultiAgentEnv_8cd7b_00000_0_2024-11-14_17-34-03/checkpoint_000000"
    
    args = parse_corl_args()
    experiment_class, experiment_file_validated = build_experiment(args) # calls register envs

    rllib_chkpt_to_onnx(chkpt_path, output_folder= "./onnx_model_2", policy_id="blue2_ctrl")
    # convert_policies_to_onnx(policy_dir=policies, dest_dir="./onnx_models")
