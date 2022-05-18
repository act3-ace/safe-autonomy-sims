# 3D Dubins Rejoin

## Description

Similar to the 2D Dubins Rejoin task, this task challenges an ML agent to learn to perform a rejoin maneuver,
flying in formation relative to a lead aircraft for a specified period of time. 
Again, the default task contains one scripted lead and one ML agent controlled wingman.

The key difference in this task is, as the name suggests, dimensionality.
This task uses a 3D Dubins flight dynamics model to calculate
transitions in state based on aircraft velocity and orientation. 
The ML agent's controls include roll rate, pitch rate, and throttle. 


## Training

To launch a training loop, the module `corl/train_rl.py` in the `corl/` repository 
is used. This module must be passed the necessary experiment config file at launch. 
From the root of the repository, execute the following command:

```commandline
python -m corl.train_rl --cfg /path/to/safe-autonomy-sims/configs/experiments/rejoin/rejoin3d/rejoin_3d.yml
```