# 3D Inspection - Multiagent

## Description

The 3D CWH Inspection task objective is to maneuver multiple deputy
spacecraft to inspect points around the chief spacecraft. A
finite number of points are generated as a sphere around the
chief in a uniform manner. It is assumed that each deputy spacecraft
always points its sensor towards the chief, and can inspect all
visible points on the sphere. The deputies successfully complete
the task once all points are inspected.

## Training

To launch a training loop, the module `corl/train_rl.py` in the `corl/` repository
is used. This module must be passed the necessary experiment config file at launch.
From the root of the repository, execute the following command:

```commandline
python -m corl.train_rl --cfg /path/to/safe-autonomy-sims/configs/experiments/multiagent/cwh_3d_inspection_multiagent.yml
```
