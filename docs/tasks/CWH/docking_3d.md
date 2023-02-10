# 3D Docking

## Description
The 3D CWH Docking task objective is to maneuver a deputy
spacecraft from a random location into a docking region 
surrounding the chief spacecraft.

## Training
To launch a training loop, the module `corl/train_rl.py` in the `corl/` repository 
is used. This module must be passed the necessary experiment config file at launch. 
From the root of the repository, execute the following command:

```commandline
python -m corl.train_rl --cfg /path/to/safe-autonomy-sims/configs/experiments/docking/docking_3d.yml
```

## Success Criteria
The deputy is considered successfully docked when its distance to the chief is less than a desired distance $\rho_d$.  

$$
\varphi_{\text{docking}}: (\Vert {r_{\rm H}}\Vert \leq  \rho_d)
$$

where

$$
\Vert {r_{\rm H}}\Vert =(x^2+y^2+z^2)^{1/2}
$$
