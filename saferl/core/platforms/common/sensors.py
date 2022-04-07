"""
Common sensor implementations.

Author: Jamie Cunningham
"""
import abc
import typing

import numpy as np
from act3_rl_core.simulators.base_parts import BaseSensor


class TransformSensor(BaseSensor):
    """
    Sensor part which applies a transformation to the measurement.
    """

    def _calculate_measurement(self, state: typing.Tuple) -> typing.Union[np.ndarray, typing.Tuple, typing.Dict]:
        measurement = self._raw_measurement(state)
        measurement = self._transform(measurement, state)
        return measurement

    def _raw_measurement(self, state):
        """
        Calculate the raw sensor measurement value.
        Parameters
        ----------
        state: typing.Tuple
            The current state of the environment used to obtain the measurement

        Returns
        -------
        typing.Union[np.ndarray, typing.Tuple, typing.Dict]
            The raw measurements from this sensor
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _transform(self, measurement, state):
        """
        Perform a transformation on a measurement.

        Parameters
        ----------
        measurement: typing.Union[np.ndarray, typing.Tuple, typing.Dict]
            The measurement to be transformed
        state: typing.Tuple
            The current state of the environment used to obtain the measurement

        Returns
        -------
        typing.Union[np.ndarray, typing.Tuple, typing.Dict]
            The transformed measurements from this sensor
        """
        raise NotImplementedError
