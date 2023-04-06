from saferl.evaluation.evaluation_api import run_one_evaluation
from saferl.evaluation.launch.serialize_cwh3d import SerializeCWH3D
from inspection_animation import AnimationFromCheckpoint


if __name__ == '__main__':
    # ****Replace with your paths****
    checkpoint_path = 'example_checkpoint' # Path to checkpoint directory
    expr_config_path = 'safe-autonomy-sims/configs/experiments/inspection/inspection_3d.yml' # Path to experiment config
    launch_dir_of_experiment = 'safe-autonomy-sims'
    task_config_path = 'safe-autonomy-sims/configs/tasks/cwh3d_inspection/cwh3d_task.yml' # Path to task config
    save_dir = 'safe-autonomy-sims/saferl/evaluation/' # Path to directory to save png/mp4
    mode = 'operator' # Plotting mode: 'operator', 'obs_act', or '3d_pos'
    last_step = True # True for png of last step, False for mp4

    metrics_config = {
        "agent": {
            "__default__": [
                {
                    "name": "ObservationVector",
                    "functor": "saferl.evaluation.general_metrics.ObservationVector",
                    "config": {
                        "description": "Observations"
                    }
                },
                {
                    "name": "ControlVector",
                    "functor": "saferl.evaluation.general_metrics.ControlVector",
                    "config": {
                        "description": "Actions"
                    }
                },
            ]
        }
    }

    default_test_case_manager_config = {
        "class_path": "corl.evaluation.runners.section_factories.test_cases.default_strategy.DefaultStrategy",
        "config": {
            "num_test_cases": 1
        }
    }

    dataframe = run_one_evaluation(task_config_path, expr_config_path, launch_dir_of_experiment, metrics_config, checkpoint_path, SerializeCWH3D, test_case_manager_config=default_test_case_manager_config)

    animation = AnimationFromCheckpoint()
    animation.make_animation(dataframe, save_dir, mode, last_step)
