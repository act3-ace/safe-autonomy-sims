"""Mocks for use in testing the Inspection Gymnasium environment"""

import typing

import numpy as np
from safe_autonomy_simulation.entities.entity import Entity
from safe_autonomy_simulation.sims.inspection.inspection_points import InspectionPointSet
from safe_autonomy_simulation.sims.spacecraft.point_model import CWHDynamics
from safe_autonomy_simulation.sims.spacecraft.sixdof_model import SixDOFDynamics


class MockInspectionPointSet(InspectionPointSet):
    """Mock inspection point set used for testing without having to reach a state where
    points are inspected
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._steps_before_inspection_toggle = 2
        self._all_points_inspected = False

    def _post_step(self, step_size: float):
        self._steps_before_inspection_toggle -= 1

        if self._steps_before_inspection_toggle == 0:
            self._all_points_inspected = True

    def get_total_weight_inspected(self, inspector_entity: Entity | None = None) -> float:
        return 1.0 if self._all_points_inspected else 0.0


class StaticSixDOFDynamics(SixDOFDynamics):
    """A 6DOF dynamics class for testing that keeps the state static to simplify testing
    """

    def _step(self, step_size: float, state: np.ndarray, control: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        return (state, np.zeros(len(state)))


class StaticCWHDynamics(CWHDynamics):
    """A CWH dynamics class for testing that keeps the state static to simplify testing
    """

    def _step(self, step_size: float, state: np.ndarray, control: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        return (state, np.zeros(len(state)))
