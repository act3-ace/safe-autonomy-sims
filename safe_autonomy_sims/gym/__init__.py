from gymnasium.envs.registration import register


# ---- single agent environments ----

register(
    id="Docking-v0",
    entry_point="safe_autonomy_sims.gymnasium.docking.docking_v0:DockingEnv",
)

register(
    id="Inspection-v0",
    entry_point="safe_autonomy_sims.gymnasium.inspection.inspection_v0:InspectionEnv",
)

register(
    id="WeightedInspection-v0",
    entry_point="safe_autonomy_sims.gymnasium.inspection.weighted_inspection_v0:WeightedInspectionEnv",
)

register(
    id="SixDofInspection-v0",
    entry_point="safe_autonomy_sims.gymnasium.inspection.sixdof_inspection_v0:SixDofInspectionEnv",
)
