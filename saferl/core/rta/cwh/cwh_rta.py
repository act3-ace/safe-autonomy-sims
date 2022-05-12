"""
This module implements Run Time Assurance for Clohessy-Wiltshire spacecraft
"""
import abc

from run_time_assurance.rta import RTABackupController, RTAModule
from run_time_assurance.zoo.cwh.docking_3d import (
    M_DEFAULT,
    N_DEFAULT,
    V0_DEFAULT,
    V1_COEF_DEFAULT,
    X_VEL_LIMIT_DEFAULT,
    Y_VEL_LIMIT_DEFAULT,
    Z_VEL_LIMIT_DEFAULT,
    Docking3dExplicitOptimizationRTA,
    Docking3dExplicitSwitchingRTA,
    Docking3dImplicitOptimizationRTA,
    Docking3dImplicitSwitchingRTA,
)

from saferl.core.glues.rta_glue import RTAGlue, RTAGlueValidator


class CWHDocking3dRTAGlueValidator(RTAGlueValidator):
    """Base validator for cwh docking 3d rta glues

    Parameters
    ----------
    m : float, optional
        mass in kg of spacecraft, by default M_DEFAULT
    n : float, optional
        orbital mean motion in rad/s of current Hill's reference frame, by default N_DEFAULT
    v0 : float, optional
        Maximum safe docking velocity in m/s, by default V0_DEFAULT
        v0 of v_limit = v0 + v1*n*||r||
    v1_coef : float, optional
        coefficient of linear component of the distance depending speed limit in 1/seconds, by default V1_COEF_DEFAULT
        v1_coef of v_limit = v0 + v1_coef*n*||r||
    x_vel_limit : float, optional
        max velocity magnitude in the x direction, by default X_VEL_LIMIT_DEFAULT
    y_vel_limit : float, optional
        max velocity magnitude in the y direction, by default Y_VEL_LIMIT_DEFAULT
    z_vel_limit : float, optional
        max velocity magnitude in the z direction, by default Z_VEL_LIMIT_DEFAULT
    control_bounds_high : Union[float, np.ndarray], optional
        upper bound of allowable control. Pass a list for element specific limit. By default 1
    control_bounds_low : Union[float, np.ndarray], optional
        lower bound of allowable control. Pass a list for element specific limit. By default -1
    """
    m: float = M_DEFAULT
    n: float = N_DEFAULT
    v0: float = V0_DEFAULT
    v1_coef: float = V1_COEF_DEFAULT
    x_vel_limit: float = X_VEL_LIMIT_DEFAULT
    y_vel_limit: float = Y_VEL_LIMIT_DEFAULT
    z_vel_limit: float = Z_VEL_LIMIT_DEFAULT
    control_bounds_high: float = 1
    control_bounds_low: float = -1


class RTAGlueCWHDocking3d(RTAGlue):
    """General RTA Glue to wrap CWH Docking 3d RTA from the run-time-assurance package"""

    def __init__(self, **kwargs):
        self.config: CWHDocking3dRTAGlueValidator
        super().__init__(**kwargs)

    @property
    def get_validator(cls):
        return CWHDocking3dRTAGlueValidator

    def _create_rta_module(self) -> RTAModule:
        rta_args = self._get_rta_args()
        rta_module = self._instantiate_rta_module(**rta_args)
        return rta_module

    @abc.abstractmethod
    def _instantiate_rta_module(self, **kwargs) -> RTAModule:
        raise NotImplementedError

    def _get_rta_args(self) -> dict:
        return {
            "m": self.config.m,
            "n": self.config.n,
            "v0": self.config.v0,
            "v1_coef": self.config.v1_coef,
            "x_vel_limit": self.config.x_vel_limit,
            "y_vel_limit": self.config.y_vel_limit,
            "z_vel_limit": self.config.z_vel_limit,
            "control_bounds_high": self.config.control_bounds_high,
            "control_bounds_low": self.config.control_bounds_low,
        }


class CWHDocking3dExplicitSwitchingRTAGlueValidator(CWHDocking3dRTAGlueValidator):
    """Validator for CWH Docking 3d Explicit Switching RTA Glue

    Parameters
    ----------
    backup_controller : RTABackupController, optional
        backup controller object utilized by rta module to generate backup control.
        By default Docking2dStopLQRBackupController
    """
    # backup_controller: RTABackupController = None
    test: int = 1


class RTAGlueCHWDocking3dExplicitSwitching(RTAGlueCWHDocking3d):
    """RTA Glue to wrap CWH Docking 3d Explicit Switching RTA from the run-time-assurance package"""

    def __init__(self, **kwargs):
        self.config: CWHDocking3dExplicitSwitchingRTAGlueValidator
        super().__init__(**kwargs)

    @property
    def get_validator(cls):
        return CWHDocking3dExplicitSwitchingRTAGlueValidator

    def _instantiate_rta_module(self, **kwargs) -> RTAModule:
        return Docking3dExplicitSwitchingRTA(**kwargs)

    def _get_rta_args(self) -> dict:
        parent_args = super()._get_rta_args()
        return {
            **parent_args,
            'backup_controller': self.config.backup_controller,
        }


class CWHDocking3dImplicitSwitchingRTAGlueValidator(CWHDocking3dRTAGlueValidator):
    """Validator for CWH Docking 3d Implicit Switching RTA Glue

    Parameters
    ----------
    backup_window : float
        Duration of time in seconds to evaluate backup controller trajectory
    backup_controller : RTABackupController, optional
        backup controller object utilized by rta module to generate backup control.
        By default Docking2dStopLQRBackupController
    """
    backup_window: float = 5
    backup_controller: RTABackupController = None


class RTAGlueCHWDocking3dImplicitSwitching(RTAGlueCWHDocking3d):
    """RTA Glue to wrap CWH Docking 3d Explicit Switching RTA from the run-time-assurance package"""

    def __init__(self, **kwargs):
        self.config: CWHDocking3dImplicitSwitchingRTAGlueValidator
        super().__init__(**kwargs)

    @property
    def get_validator(cls):
        return CWHDocking3dImplicitSwitchingRTAGlueValidator

    def _instantiate_rta_module(self, **kwargs) -> RTAModule:
        return Docking3dImplicitSwitchingRTA(**kwargs)

    def _get_rta_args(self) -> dict:
        parent_args = super()._get_rta_args()
        return {
            **parent_args,
            'backup_window': self.config.backup_window,
            'backup_controller': self.config.backup_controller,
        }


class RTAGlueCHWDocking3dExplicitOptimization(RTAGlueCWHDocking3d):
    """RTA Glue to wrap CWH Docking 3d Explicit Switching RTA from the run-time-assurance package"""

    def _instantiate_rta_module(self, **kwargs) -> RTAModule:
        return Docking3dExplicitOptimizationRTA(**kwargs)


class CWHDocking3dImplicitOptimizationRTAGlueValidator(CWHDocking3dRTAGlueValidator):
    """Validator for CWH Docking 3d Implicit Optimization RTA Glue

    Parameters
    ----------
    backup_window : float
        Duration of time in seconds to evaluate backup controller trajectory
    num_check_all : int
        Number of points at beginning of backup trajectory to check at every sequential simulation timestep.
        Should be <= backup_window.
        Defaults to 0 as skip_length defaults to 1 resulting in all backup trajectory points being checked.
    skip_length : int
        After num_check_all points in the backup trajectory are checked, the remainder of the backup window is filled by
        skipping every skip_length points to reduce the number of backup trajectory constraints. Will always check the
        last point in the backup trajectory as well.
        Defaults to 1, resulting in no skipping.
    backup_controller : RTABackupController, optional
        backup controller object utilized by rta module to generate backup control.
        By default Docking2dStopLQRBackupController
    """
    backup_window: float = 5
    num_check_all: int = 5
    skip_length: int = 1
    backup_controller: RTABackupController = None


class RTAGlueCHWDocking3dImplicitOptimization(RTAGlueCWHDocking3d):
    """RTA Glue to wrap CWH Docking 3d Explicit Switching RTA from the run-time-assurance package"""

    def __init__(self, **kwargs):
        self.config: CWHDocking3dImplicitOptimizationRTAGlueValidator
        super().__init__(**kwargs)

    @property
    def get_validator(cls):
        return CWHDocking3dImplicitOptimizationRTAGlueValidator

    def _instantiate_rta_module(self, **kwargs) -> RTAModule:
        return Docking3dImplicitOptimizationRTA(**kwargs)

    def _get_rta_args(self) -> dict:
        parent_args = super()._get_rta_args()
        return {
            **parent_args,
            'backup_window': self.config.backup_window,
            'num_check_all': self.config.num_check_all,
            'skip_length': self.config.skip_length,
            'backup_controller': self.config.backup_controller,
        }
