from gymnasium.envs.registration import register


# ---- single agent environments ----

register(
    id="Docking-v0",
    entry_point="safe_autonomy_sims.gym.docking.docking_v0:DockingEnv",
)

register(
    id="Inspection-v0",
    entry_point="safe_autonomy_sims.gym.inspection.inspection_v0:InspectionEnv",
)

register(
    id="WeightedInspection-v0",
    entry_point="safe_autonomy_sims.gym.inspection.weighted_inspection_v0:WeightedInspectionEnv",
)

register(
    id="SixDofInspection-v0",
    entry_point="safe_autonomy_sims.gym.inspection.sixdof_inspection_v0:WeightedSixDofInspectionEnv",
)
