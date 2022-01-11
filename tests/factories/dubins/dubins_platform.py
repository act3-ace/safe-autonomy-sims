import factory

import saferl.platforms.dubins.dubins_platform as p


class DubinsPlatformFactory(factory.Factory):
    class Meta:
        model = p.DubinsPlatform

    _exclusive_parts = None
    _platform = None
    name = None
    sensors = None
    controllers = None
