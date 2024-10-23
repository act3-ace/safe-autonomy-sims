"""Mocks for use in testing the Inspection Gymnasium environment"""

import typing

import numpy as np
from safe_autonomy_simulation.dynamics.dynamics import Dynamics
from safe_autonomy_simulation.entities.entity import Entity
from safe_autonomy_simulation.entities.physical import PhysicalEntity
from safe_autonomy_simulation.sims.inspection.inspection_points import InspectionPointSet


class MockInspectionPointSet(InspectionPointSet):
    """Mock inspection point set used for testing without having to reach a state where
    points are inspected
    """

    def __init__(self, parent: PhysicalEntity):
        super().__init__("inspection_points", parent, 100, 1, np.array([1, 0, 0]))
        self._steps_before_inspection_toggle = 2
        self._all_points_inspected = False

    def _post_step(self, step_size: float):
        self._steps_before_inspection_toggle -= 1

        if self._steps_before_inspection_toggle == 0:
            self._all_points_inspected = True

    def get_total_weight_inspected(self, inspector_entity: Entity | None = None) -> float:
        return 1.0 if self._all_points_inspected else 0.0


class StaticDynamics(Dynamics):
    """A dynamics class for testing that keeps the state static to simplify testing
    """

    def _step(self, step_size: float, state: np.ndarray, control: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        return (state, np.zeros(len(state)))
