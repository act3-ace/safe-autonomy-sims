"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements the base initializer class.
"""

import abc
import typing

from corl.libraries.units import Quantity
from pydantic import BaseModel


def strip_units(value) -> typing.Any:
    """
    Strips units from an input value
    Note that no conversions take place.

    Supports CoRL and Pint units

    Parameters
    ----------
    value : Any
        value to be stripped on units

    Returns
    -------
    typing.Any
        same as input but with all units stripped
    """

    if isinstance(value, Quantity):
        return value.m
    return value


def strip_units_from_dict(input_dict: typing.Dict) -> typing.Dict:
    """
    Strips units from input dict and returns dict of values only.
    Note that no conversions take place.

    Supports CoRL and Pint units

    Parameters
    ----------
    input_dict : typing.Dict
        dict of parameters to be stripped of units

    Returns
    -------
    dict
        same as input dict but with all units stripped
    """
    out = {}
    for k, v in input_dict.items():
        out[k] = strip_units(v)
    return out


class InitializerValidator(BaseModel):
    """
    Base Validator for Initializers

    Currently has no parameters, however, make sure to inherit from it to receive future updates to Initializer interface.
    """


class BaseInitializer(abc.ABC):
    """
    This class defines the template for Initializer classes. Initializers are responsible
    for providing a dictionary complete with all relevant agent_reset_config values. Initializers
    encapsulate the initialization of randomized and conditional (dependent) agent state values.
    """

    def __init__(self, config):
        self.config = self.get_validator()(**config)

    @staticmethod
    def get_validator() -> typing.Type[InitializerValidator]:
        """
        get validator for this Done Functor

        Returns:
            DoneFuncBaseValidator -- validator the done functor will use to generate a configuration
        """
        return InitializerValidator

    def __call__(self, **kwargs) -> typing.Dict:
        return self.compute(**kwargs)

    @abc.abstractmethod
    def compute(self, **kwargs) -> typing.Dict:
        """Computes initialization params from an arbitrary set of input kwargs

        Returns
        -------
        typing.Dict
            initialization params generated from input kwargs
        """
        raise NotImplementedError


class BaseInitializerWithPint(BaseInitializer):
    """Generic initializer that utilize a pint UnitRegistry for quantities and unit conversion
    """


class PassThroughInitializer(BaseInitializer):
    """Initializer to simply pass through input kwargs as initialization parameters
    """

    def compute(self, **kwargs):
        return kwargs


class StripUnitsInitializer(BaseInitializer):
    """Initializer to simply strip units from input kwargs and pass magnitude. Note, does not perform unit checking or conversion"""

    def compute(self, **kwargs):
        return strip_units_from_dict(kwargs)


class Accessor(abc.ABC):
    """Base interface for accessor objects to accesss arbitrary sim values for entity initilizer parameters"""

    def __init__(self):
        ...

    @property
    @abc.abstractmethod
    def dependencies(self) -> typing.Set:
        """
        Returns the set of dependencies of this accessor.
        Allows initialization process to initialize dependencies before utilizing this accessor.

        Returns
        -------
        typing.Set
            set of dependency names for this accessor (e.g. names of other entities)
        """

    @abc.abstractmethod
    def access(self, sim, sim_entities: typing.Dict[str, typing.Any]):
        """Access values in the sim and sim entities

        Parameters
        ----------
        sim
            reference to partially initialized sim from which values may be accessed
        sim_entities : typing.Dict[str, typing.Any]
            dictionary of already initialized entites from which values may be accessed
        """


class AttributeAccessor(Accessor):
    """Base accessor implementation for accessing arbitrary attributes from objects

    Parameters
    ----------
    attribute_name: str
        name of attribute to be queried and returned by accessor. May contain nested attributes chained with "."
            e.g. "first.second.third" accesses obj.first.second.third
    """

    def __init__(self, attribute_name):
        super().__init__()
        self.attribute_name = attribute_name

    def get_attribute(self, obj):
        """gets attribute described in constructor from arbitrary input object

        Parameters
        ----------
        obj : Any
            Object from which attribute is accessed

        Returns
        -------
        Any
            accessed attribute
        """
        output = None
        next_obj = obj
        attribute_name_sections = self.attribute_name.split(".")

        if not attribute_name_sections:
            raise ValueError("Attribute name is empty")

        for attribure_name_section in attribute_name_sections:
            output = getattr(next_obj, attribure_name_section)
            next_obj = output

        return output


class SimAttributeAccessor(AttributeAccessor):
    """Accessor for attributes of parent sim object

    Parameters
    ----------
    attribute_name: str
        name of attribute to be queried and returned by accessor. May contain nested attributes chained with "."
            e.g. "first.second.third" accesses obj.first.second.third
    """

    @property
    def dependencies(self) -> typing.Set:
        return set()

    def access(self, sim, _):
        return self.get_attribute(sim)


class EntityAttributeAccessor(AttributeAccessor):
    """Accessor for accessing the attribute of another entity

    Parameters
    ----------
    attribute_name: str
        name of attribute to be queried and returned by accessor. May contain nested attributes chained with "."
            e.g. "first.second.third" accesses obj.first.second.third
    entity_name : str
        name of entity to retrieve attribute from
    """

    def __init__(self, attribute_name, entity_name):
        super().__init__(attribute_name)
        self.entity_name = entity_name

    @property
    def dependencies(self) -> typing.Set:
        return set((self.entity_name, ))

    def access(self, _, sim_entities: typing.Dict[str, typing.Any]):
        return self.get_attribute(sim_entities[self.entity_name])
