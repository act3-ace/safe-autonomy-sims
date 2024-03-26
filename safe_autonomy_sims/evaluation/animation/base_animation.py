"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

import abc
import os
import sys
import typing

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import pandas as pd
from corl.evaluation.runners.section_factories.plugins.platform_serializer import PlatformSerializer
from numpy.typing import ArrayLike

from safe_autonomy_sims.evaluation.evaluation_api import run_one_evaluation


class BaseAnimationModule(abc.ABC):
    """
    Base module for making animations/plots from a checkpoint

    Parameters
    ----------
    checkpoint_path: str
        The absolute path to the checkpoint from which the policy under evaluation will be loaded
    expr_config_path: str
        The absolute path to the experiment config used in training
    task_config_path: str
        The absolute path to the task_config used in training
    platform_serializer_class: PlatformSerializer
        The PlatformSerializer subclass capable of saving the state of the Platforms used in the evaluation environment
    save_dir: str
        The absolute path to the directory in which evaluation episode(s) data will be saved
    launch_dir_of_experiment: str
        Paths in the experiment config are relative to the directory the corl.train_rl module is intended to be launched from.
        This string captures the path to the intended launch location of the experiment under evaluation.
    tmp_dir: str
        Temporary directory to save animation images to
    start_index: int
        Index to start plotting loop
    """

    def __init__(
        self,
        checkpoint_path: str,
        expr_config_path: str,
        task_config_path: str,
        platform_serializer_class: PlatformSerializer,
        save_dir: str = 'safe-autonomy-sims/safe_autonomy_sims/evaluation/animation/figs/',
        launch_dir_of_experiment: str = 'safe-autonomy-sims',
        tmp_dir: str = '/tmp/eval_results/animation_plots',
        start_index: int = 0
    ):

        self.checkpoint_path = checkpoint_path
        self.expr_config_path = expr_config_path
        self.task_config_path = task_config_path
        self.platform_serializer_class = platform_serializer_class
        self.save_dir = save_dir
        self.launch_dir_of_experiment = launch_dir_of_experiment
        self.tmp_dir = tmp_dir
        self.start_index = start_index

        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        self.filenames: list[str] = []
        self.filetype = 'png'

        self.metrics_config = {
            "agent": {
                "__default__": [
                    {
                        "name": "ObservationVector",
                        "functor": "safe_autonomy_sims.evaluation.general_metrics.ObservationVector",
                        "config": {
                            "description": "Observations"
                        }
                    },
                    {
                        "name": "ControlVector",
                        "functor": "safe_autonomy_sims.evaluation.general_metrics.ControlVector",
                        "config": {
                            "description": "Actions"
                        }
                    },
                ]
            }
        }

        self.default_test_case_manager_config = {
            "type": "corl.evaluation.runners.section_factories.test_cases.default_strategy.DefaultStrategy",
            "config": {
                "num_test_cases": 1
            }
        }

    def make_animation(
        self,
        dataframe: typing.Union[pd.DataFrame, None] = None,
        filetype: str = 'png',
    ):
        """
        Run one evaluation episode using the checkpoint and make an animation.

        Parameters
        ----------
        dataframe: typing.Union[pd.DataFrame, None]
            A pre-existing dataframe to be used, to bypass running and evaluation episode
        filetype: str
            Type of file to save the animation/plot. Use 'mp4' or 'gif' for animations, and 'png' for plots
        """
        self.filetype = filetype
        assert self.filetype in ['png', 'gif', 'mp4'], "filetype must be 'png', 'gif', or 'mp4'"

        if dataframe is None:
            dataframe = run_one_evaluation(
                self.task_config_path,
                self.expr_config_path,
                self.launch_dir_of_experiment,
                self.metrics_config,
                self.checkpoint_path,
                self.platform_serializer_class,
                test_case_manager_config=self.default_test_case_manager_config
            )
            pd.to_pickle(dataframe, os.path.join(self.save_dir, 'episode_dataframe.pkl'))

        data = self.get_data_from_dataframe(dataframe)

        axes = self.setup_plots(data)

        self.filenames = []
        print('Animation Progress:')
        for i in range(self.start_index, len(dataframe['ObservationVector'].values[0])):
            self.make_plots_in_loop(axes, data, i)

            # Print progress
            var = (i + 1) / len(dataframe['ObservationVector'].values[0])
            sys.stdout.write('\r')
            sys.stdout.write(f"[{'=' * int(var * 20) : <20}] {var * 100}%")
            sys.stdout.flush()

            if self.filetype != 'png':
                # Save figure
                filename = os.path.join(self.tmp_dir, 'frame' + str(i) + '.png')
                self.filenames.append(filename)
                plt.tight_layout()
                plt.savefig(filename)

                # Remove temporary elements
                self.remove_temp_elements(axes)

        print('\n')
        self.print_metrics(data)

        if self.filetype == 'png':
            # Make plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'episode_plot.png'))
        else:
            # Make video
            images: list[ArrayLike] = [imageio.imread(filename) for filename in self.filenames]
            imageio.mimwrite(os.path.join(self.save_dir, 'episode_animation.' + self.filetype), ims=images, fps=30)

            # Remove temp files
            for filename in set(self.filenames):
                os.remove(filename)
            os.rmdir(self.tmp_dir)

    def get_data_from_dataframe(self, dataframe: pd.DataFrame) -> typing.Any:
        """
        Get data from the dataframe into another form

        Parameters
        ----------
        dataframe: pd.DataFrame
            Dataframe created by running an evaluation episode

        Returns
        -------
        typing.Any
            Episode data
        """
        return dataframe

    @abc.abstractmethod
    def setup_plots(self, data: typing.Any) -> typing.Any:
        """
        Setup and return all axes

        Parameters
        ----------
        data: typing.Any
            Episode data

        Returns
        -------
        typing.Any
            Axes
        """
        raise NotImplementedError

    @abc.abstractmethod
    def make_plots_in_loop(self, axes: typing.Any, data: typing.Any, i: int):
        """
        Add data to the plots inside the loop.

        Parameters
        ----------
        axes: typing.Any
            Axes
        data: typing.Any
            Episode data
        i: int
            Current loop iteration
        """
        raise NotImplementedError

    def remove_temp_elements(self, axes: typing.Any):
        """
        Remove any temporary animation elements

        Parameters
        ----------
        axes: typing.Any
            Axes
        """

    def print_metrics(self, data: typing.Any):
        """
        Print any final metrics for the episode

        Parameters
        ----------
        data: typing.Any
            Episode data
        """
