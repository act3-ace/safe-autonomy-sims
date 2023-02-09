# 2D Rejoin

## Description

This task uses a simplified 2D Dubins flight dynamics model to calculate
transitions in state based on aircraft velocity and heading. The ML agent's controls
include throttle and heading turn rate. The agent wingman must remain in formation
for a specified amount of time in order to succeed.

## Training

To launch a training loop, the module `corl/train_rl.py` in the `corl/` repository 
is used. This module must be passed the necessary experiment config file at launch. 
From the root of the repository, execute the following command:

```commandline
python -m corl.train_rl --cfg /path/to/safe-autonomy-sims/configs/experiments/rejoin/rejoin2d/rejoin_2d.yml
```