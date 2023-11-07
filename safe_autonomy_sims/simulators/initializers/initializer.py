"""
This module defines the base initializer class.

Author: John McCarroll
"""

import abc
import typing

import pint
from corl.libraries.units import GetStrFromUnit, NoneUnitType, ValueWithUnits
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

    if isinstance(value, ValueWithUnits):
        return value.value
    if isinstance(value, pint.Quantity):
        return value.magnitude
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


def corl_value_to_pint(
    value: typing.Union[typing.Any, ValueWithUnits], ureg: pint.UnitRegistry, to_unit: typing.Union[str, pint.Unit] = None
) -> typing.Any:
    """
    Converts CoRL ValueWithUnits to pint Quantity. Doesn't modify values without units

    Parameters
    ----------
    value : typing.Union[typing.Any, ValueWithUnits]
        input value to convert to pint Quantity
    ureg : pint.UnitRegistry
        pint unit registry to be used for unit conversion
    to_unit: str or pint.Unit
        converts the input value to this unit if the value has units. Optional

    Returns
    -------
    typing.Any
        same as input but converted to pint Quantity if a CoRL ValueWithUnits
    """

    if isinstance(value, ValueWithUnits):
        units = value.units
        if isinstance(units, NoneUnitType):
            unit_str = None
        else:
            unit_str = GetStrFromUnit(units)

        q = ureg.Quantity(value.value, unit_str)
        if to_unit is not None:
            q = q.to(to_unit)

        return q
    return value


def corl_values_to_pint_from_dict(input_dict: typing.Dict, ureg: pint.UnitRegistry) -> typing.Dict:
    """
    Converts CoRL ValueWithUnits dict elements to pint Quantities. Leaves values without units alone

    Note that no conversion takes place

    Parameters
    ----------
    input_dict : typing.Dict
        dict of parameters to be converted to pint quatities
    ureg : pint.UnitRegistry
        pint unit registry to be used for unit conversion

    Returns
    -------
    dict
        same as input_dict but with CoRL ValueWithUnits converted to pint Quantities
    """
    out = {}
    for k, v in input_dict.items():
        out[k] = corl_value_to_pint(v, ureg)

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
    encapsulatie the  initialization of randomized and conditional (dependant) agent state values.
    """

    def __init__(self, config):
        self.config = self.get_validator(**config)

    @property
    def get_validator(self) -> typing.Type[InitializerValidator]:
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

    def __init__(self, config):
        super().__init__(config)
        self.ureg: pint.UnitRegistry = pint.get_application_registry()


class PassThroughInitializer(BaseInitializer):
    """Initializer to simply pass through input kwargs as initialization parameters
    """

    def compute(self, **kwargs):
        return kwargs


class StripUnitsInitializer(BaseInitializer):
    """Initializer to simply strip units from input kwargs and pass magnitude. Note, does not perform unit checking or conversion"""

    def compute(self, **kwargs):
        return strip_units_from_dict(kwargs)


class CorlUnitsToPintInitializer(BaseInitializerWithPint):
    """Initializer to convert all CoRL ValuesWithUnits to pint Quantities"""

    def compute(self, **kwargs):
        return corl_values_to_pint_from_dict(kwargs, self.ureg)


class PintUnitConversionInitializer(BaseInitializerWithPint):
    """
    Generic Initializer for handling input CoRL ValuesWith Units

    Input values are converted to pint Quantities and unit converted to desired values.

    param_units defines which unit each expected input parameter should be converted to.
        Use unit name string or pint.Unit. Use None for unitless values.
    These values are provided as pint Quantities and unit stripped raw values.
    """
    param_units: typing.Dict[str, typing.Union[str, pint.Unit]] = {}

    def compute(self, **kwargs):
        kwargs_with_converted_units = {}
        for k, v in kwargs.items():
            if k not in self.param_units:
                raise TypeError(f"Unexpected Argument {k}")
            kwargs_with_converted_units[k] = corl_value_to_pint(v, self.ureg, to_unit=self.param_units[k])

        kwargs_with_stripped_units = strip_units_from_dict(kwargs_with_converted_units)

        return self.compute_with_units(kwargs_with_converted_units, kwargs_with_stripped_units)

    @abc.abstractmethod
    def compute_with_units(self, kwargs_with_converted_units, kwargs_with_stripped_units):
        """Computes initialization params from input kwargs with unit conversion applied

        Parameters
        ----------
        kwargs_with_converted_units : dict
            input dictionary composed of pint.Quantities converted to unit specified in self.param_units
        kwargs_with_stripped_units : dict
            input dictionary composed of raw values after conversion to unit specified in self.param_units
        """
        raise NotImplementedError


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
        ...

    @abc.abstractmethod
    def access(self, sim, sim_entities: typing.Dict[str, typing.Any]):
        """_summary_

        Parameters
        ----------
        sim
            reference to partially initialized sim from which values may be accessed
        sim_entities : typing.Dict[str, typing.Any]
            dictionary of already initialized entites from which values may be accessed
        """
        ...


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
