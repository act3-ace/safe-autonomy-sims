# 3D CWH Docking

## Description

The 3D CWH Docking task objective is to maneuver a deputy
satellite from a random location into a docking region 
surrounding a chief satellite.

## Training

To launch a training loop, the module `corl/train_rl.py` in the `corl/` repository 
is used. This module must be passed the necessary experiment config file at launch. 
From the root of the repository, execute the following command:

```commandline
python -m corl.train_rl --cfg /path/to/safe-autonomy-sims/configs/experiments/docking/docking_3d.yml
```