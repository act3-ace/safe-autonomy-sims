# safe-autonomy-sims


## Intro
The Safe-Autonomy-Sims library provides the components and
tools to build modular, integration-focused Reinforcement 
Learning environments with Run Time Assurance (RTA). 
This repo is designed to work hand-in-glove with the CoRL 
library.

## Installation
The following instructions detail how to install 
the safe-autonomy-sims library on your local system.
It is recommended to install the python modules within 
a [virtualenv](https://virtualenv.pypa.io/en/stable/#)
or [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.

### Installing CoRL
Clone a copy of the CoRL source code onto
your local machine:
```shell
# via SSH
git clone git@github.com/act3-ace:act3-rl/corl.git

# via HTTPS
git clone https://github.com/act3-ace/act3-rl/corl.git
```

Install the CoRL (act3-rl-core) module into your 
environment using `pip`:
```shell
pip install path/to/corl/
```

### Installing safe-autonomy-sims
Clone a copy of the safe-autonomy-sims source code 
onto your local machine:
```shell
# via SSH
git clone git@github.com/act3-ace:rta/safe-autonomy-sims.git

# via HTTPS
git clone https://github.com/act3-ace/safe-autonomy-sims.git
```

Install the safe-autonomy-sims module into your 
environment using `pip`:
```shell
pip install path/to/safe-autonomy-sims/
```

For a local development version, please install 
using the `-e, --editable` option:
```shell
pip install -e path/to/safe-autonomy-sims/
```


## Usage

### Training

Training experiments are conducted in safe-autonomy-sims via
[configuration files](configs). These files can be manipulated
to define experiment parameters, agent configurations,
environments, tasks, policies, and platforms.

The CoRL library provides a training endpoint script which
uses the RLLib reinforcement learning library to train agents
in an environment.

As an example, you can launch a training loop for the
provided Docking 3D environment using the following command:
```shell
python path/to/corl/corl train.py --cfg /path/to/safe-autonomy-sims/configs/experiments/docking/docking_3d.yml --compute_platform local
```

### Evaluation

TO BE DETERMINED

## Environments

This library includes the following environments:
  - Rejoin 2D
  - Rejoin 3D
  - Docking 3D


### Rejoin
Aircraft formation flight rejoin where a wingman aircraft controlled by the agent must join a formation relative to a lead aircraft. The formation is defined by a rejoin region relative to the lead's position and orientation which the wingman must enter and remain within. Comes in the following flavors:

-  **Rejoin 2D**  
Throttle and heading control.  

-  **Rejoin 3D**  
Throttle, heading, flight angle, roll control.  


### Docking
Spacecraft docking scenario where an agent controlled deputy spacecraft must dock with a stationary chief spacecraft while both orbit a central body. This is accomplished by approaching the chief to within a predefined docking distance while maintaining a safe relative velocity within that distance. The motion of the deputy spacecraft is governed by the Clohessy-Wiltshire linearized dynamics model. Comes in the following flavors: 

-  **Docking 3D**
Static 1N thrusters in $`\pm x, \pm y, \pm z`$.


## Team
Jamie Cunningham,
Vardaan Gangal,
Umberto Ravaioli,
John McCarroll,
Kerianne Hobbs
