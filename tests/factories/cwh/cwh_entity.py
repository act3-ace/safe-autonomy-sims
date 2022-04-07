import factory

from saferl.backend.cwh import cwh as e


class CWHSpacecraftFactory(factory.Factory):
    class Meta:
        model = e.CWHSpacecraft

    name = "blue0"
    integration_method = "RK45"
