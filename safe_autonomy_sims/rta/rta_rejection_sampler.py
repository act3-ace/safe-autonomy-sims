"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning Core (CoRL) Safe Autonomy Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains a parameter wrapper for RTA rejection sampling.
If chosen initial parameters violate safety, they are rejected and resampled.
"""

import typing
from collections import OrderedDict

import numpy as np
from corl.libraries.parameters import ParameterWrapper, ParameterWrapperValidator
from corl.libraries.units import Quantity
from numpy.random import Generator, RandomState
from pydantic import BaseModel, ConfigDict, create_model
from pydantic.types import PyObject
from run_time_assurance.rta import ConstraintBasedRTA
from run_time_assurance.utils import to_jnp_array_jit

from safe_autonomy_sims.simulators.initializers.initializer import BaseInitializer, InitializerValidator
from safe_autonomy_sims.simulators.saferl_simulator import InitializerResetValidator

Randomness = Generator | RandomState
OtherVars = typing.Mapping[tuple[str, ...], Quantity]


class RTAValidator(BaseModel):
    """
    A configuration validator for RTA glue object

    Attributes
    ----------
    functor: PyObject
        The class module of the RTA Glue to be instantiated
    states: list
        States to be passed from the initializer to the RTA module (in order)
    args: dict
        Arguments used to initialize the RTA module
    arg_map: dict
        Mapping of arguments from 'args' to kwargs accepted by the RTA module.
        Useful when 'args' contains extra values or naming is different.
        By default None.
    """
    functor: PyObject
    states: list
    args: dict
    arg_map: typing.Union[dict, None] = None


class RejectionSamplerWrapper(ParameterWrapperValidator):
    """
    A configuration validator for rejection sampler parameter wrapper

    Attributes
    ----------
    initializer: InitializerResetValidator
        Initializer to convert parameters to values used by the RTA
    rta: RTAValidator
        RTA glue used to check safety constraint values
    max_rejections: int
        Maximum number of times to reject a sample before passing it through
    """
    initializer: InitializerResetValidator
    rta: RTAValidator
    max_rejections: int = 1000


class RejectionSampler(ParameterWrapper):
    """
    Rejection Sampling Parameter Wrapper.
    If sampled parameters violate safety constraints, they are rejected and resampled.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Reorganize args if necessary
        if self.config.rta.arg_map is not None:
            new_args = {}
            for k, v in self.config.rta.arg_map.items():
                new_args[k] = self.config.rta.args[v]
            self.config.rta.args = new_args

        # Set unused args for RTA glue
        self.config.rta.args['wrapped'] = []
        self.config.rta.args['agent_name'] = 'None'
        self.config.rta.args['state_observation_names'] = ['None']

    @staticmethod
    def get_validator() -> type[ParameterWrapperValidator]:
        """Get the validator class for this Parameter"""
        return RejectionSamplerWrapper

    def get_value(self, rng: Randomness, other_vars: OtherVars) -> dict[str, str | Quantity]:
        # Create initializer (errors when this is done in __init__)
        initializer = self.config.initializer.functor(self.config.initializer.config)

        safe_state = False
        check_counter = 0
        # Loop until safe state is found or max iterations is reached
        while not safe_state and check_counter < self.config.max_rejections:
            check_counter += 1
            sample = self.get_sample_value(rng, other_vars, initializer)
            state = self.get_state(sample)
            safe_state = self.check_if_safe_state(state)

        # TODO: fixes callback error
        sample['m'] = 0.0

        # We create a dynamic model since the rejection sampler can wrap any parameter as defined
        # in configuration files
        field_defs = OrderedDict({k: (type(v), v) for k, v in sample.items()})
        model_config = ConfigDict(arbitrary_types_allowed=True)
        params_model = create_model('init_state', **field_defs, __config__=model_config)(**sample)

        return params_model  # type: ignore

    def get_sample_value(self, rng: Randomness, other_vars: OtherVars, initializer: BaseInitializer) -> typing.Dict:
        """
        Sample the parameters

        Parameters
        ----------
        rng: Randomness
            Randomness for sampling
        other_vars: OtherVars
            Other variables for sampled parameters
        initializer: BaseInitializer
            Initializer used to get state from parameters

        Returns
        -------
        typing.Dict
            Initializer state
        """
        params = {}
        for k, p in self.config.wrapped.items():
            v = p.get_value(rng, other_vars)
            params[k] = v

        transformed_params = initializer.compute(**params)
        return transformed_params

    def get_state(self, value: typing.Dict) -> np.ndarray:
        """
        Gets RTA state from Initializer state

        Parameters
        ----------
        value: typing.Dict
            Initializer state

        Returns
        -------
        np.ndarray
            RTA state
        """
        array = []
        for k in self.config.rta.states:
            v = value[k] if isinstance(value[k], np.ndarray) else [value[k]]
            array.append(v)
        return np.concatenate(array)

    def check_if_safe_state(self, state: np.ndarray):
        """
        Determines if state is safe or not.

        Parameters
        ----------
        state: np.ndarray
            Current system state

        Returns
        -------
        bool
            True if state is safe, False if not.
        """
        assert isinstance(self.config.rta.functor(**self.config.rta.args).rta,
                          ConstraintBasedRTA), ("Must use constraint based rta for rejection sampling.")
        init_state_safe = True

        # For each constraint, check if satisfied
        for c in self.config.rta.functor(**self.config.rta.args).rta.constraints.values():
            if c.phi(to_jnp_array_jit(state), c.params) < 0 or c(to_jnp_array_jit(state), c.params) < 0:
                init_state_safe = False
                break
        return init_state_safe


class RejectionSamplerInitializerValidator(InitializerValidator):
    """
    A configuration validator for agent initializer to be used with the rejection sampler

    Attributes
    ----------
    states: typing.Optional[typing.List]
        State keys to initialize the agent.
        Useful when the rejection sampler state contains extra values.
        By default None.
    """
    states: typing.Optional[typing.List] = None


class RejectionSamplerInitializer(BaseInitializer):
    """
    Agent initializer to pass through init_state from rejection sampler.
    Example config usage:

    "simulator_reset_parameters": {
      "initializer": {
        "functor": "safe_autonomy_sims.rta.rta_rejection_sampler.RejectionSamplerInitializer",
        "config": {}
      },
      "config": {
        "init_state": {
          "type": "safe_autonomy_sims.simulators.initializers.initializer.SimAttributeAccessor",
          "config": {
            "attribute_name": "init_state",
          }
        }
      }
    }
    """

    @staticmethod
    def get_validator() -> typing.Type[InitializerValidator]:
        return RejectionSamplerInitializerValidator

    def compute(self, **kwargs):
        # Convert to dict
        init_state = dict(kwargs['init_state'])
        if self.config.states is not None:
            states = {}
            for k in self.config.states:
                states[k] = init_state[k]
        else:
            states = init_state

        return states
