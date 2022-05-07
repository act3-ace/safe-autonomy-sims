# 2D Dubins Rejoin

## Description

This task challenges an ML agent to learn to perform a basic maneuver
used in aircraft formation flight: the rejoin. Default rejoin environments
contain two aircraft, one lead and one wingman. The lead aircraft, by default, 
flies on a scripted path, while the wingman, controlled by an ML agent, attempts
to reach and maintain a relative position to the lead.

This task uses a simplified 2D Dubins flight dynamics model to calculate
transitions in state based on aircraft velocity and heading. The ML agent's controls
include throttle and heading turn rate. The agent wingman must remain in formation
for a specified amount of time in order to succeed.


## Training

To launch a training loop, the module `corl/train_rl.py` in the `corl/` repository 
is used. This module must be passed the necessary experiment config file at launch. 
From the root of the repository, execute the following command:

```commandline
python -m corl.train_rl --cfg /path/to/safe-autonomy-sims/configs/experiments/dubins2d.yml
```

