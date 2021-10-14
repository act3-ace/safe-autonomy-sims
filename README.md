# safe-autonomy-sims


# Intro
The Safe-Autonomy-Sims library provides the components and tools to build modular,  compatible Reinforcement Learning environments with Run Time Assurance (RTA). This repo is designed to work hand-in-glove with the ACT3-RL-Core library.

## Installation
Inside of the repo's root directory, simply install using the `setup.py` with:
```shell
pip install .
```

For a local development version, please install using the `-e, --editable` option:
```shell
pip install -e .
```

## Usage


### Training

TO BE DETERMINED



### Evaluation

TO BE DETERMINED 

## Environments

This library will include the following environments:
  - Rejoin 2D
  - Rejoin 3D
  - Docking 2D
  - Docking 3D


### Rejoin
Aircraft formation flight rejoin where a wingman aircraft controlled by the agent must join a formation relative to a lead aircraft. The formation is defined by a rejoin region relative to the lead's position and orientation which the wingman must enter and remain within. Comes in the following flavors:

-  **Rejoin 2D**  
Throttle and heading control.  

-  **Rejoin 3D**  
Throttle, heading, flight angle, roll control.  


### Docking
Spacecraft docking scenario where an agent controlled deputy spacecraft must dock with a stationary chief spacecraft while both orbit a central body. This is accomplished by approaching the chief to within a predefined docking distance while maintaining a safe relative velocity within that distance. The motion of the deputy spacecraft is governed by the Clohessy-Wiltshire linearlized dynamics model. Comes in the following flavors:

-  **Docking 2D**  
Static 1N thrusters in $`\pm x`$ and  $`\pm y`$.    

-  **Docking 3D**
Static 1N thrusters in $`\pm x, \pm y, \pm z`$.


## Team
Jamie Cunningham,
Vardaan Gangal
Umberto Ravaioli
John McCarroll,
Kerianne Hobbs,
