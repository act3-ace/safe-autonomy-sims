import typing

import factory
from corl.simulators.base_platform import BasePlatform


class BasePlatformFactory(factory.Factory):
    class Meta:
        model = BasePlatform

    platform_name = None
    platform = None
    platform_config: typing.List = []
