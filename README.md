# safe-autonomy-sims


## Intro
The Safe-Autonomy-Sims library provides the components and
tools to build modular, integration-focused Reinforcement 
Learning environments with Run Time Assurance (RTA). 
This repo is designed to work hand-in-glove with the CoRL,
safe-autonomy-dynamics, and run-time-assurance libraries.

## Installation
The following instructions detail how to install 
the safe-autonomy-sims library and its dependencies on your local system.
This repository provides a helper installation script `install.sh`
which allows users to install the safe-autonomy-sims package and
its dependencies onto a local machine from the secure ACT3
GitLab instance. It is recommended to run `install.sh` within 
a [virtualenv](https://virtualenv.pypa.io/en/stable/#)
or [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.

### Normal installation

1. Navigate to the 
[install.sh](https://github.com/act3-ace/safe-autonomy-sims/-/blob/main/install.sh)
file in the browser and click the download button located above 
the top right of the file contents.
2. Generate a GitLab [personal access token](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html#create-a-personal-access-token)
with at least `read_api` access if you don't already have one (save this token!).
3. Make your local copy of `install.sh` executable:
```bash
# Replace path/to/install.sh with the actual path
chmod +x path/to/install.sh
```
4. **(Recommended)** Activate your desired environment:
```bash
# Example using conda environment named myenv
conda activate myenv
```
5. Run `install.sh`, passing in a personal access token:
```bash
# Replace path/to/install.sh with the actual path
# Replace <my-access-token> with your personal access token
path/to/install.sh -p <my-access-token>
```

This should install the safe-autonomy-sims python package and its
dependencies into your local activated environment.

### Developer installation
The following instructions are for active developers of safe-autonomy-sims.
These instructions will guide you through creating an editable
local installation of this repository.

1. Clone the safe-autonomy-sims repository
```bash
# via SSH (recommended)
git clone git@github.com/act3-ace:rta/safe-autonomy-stack/safe-autonomy-sims.git
```
```bash
# via HTTPS
git clone https://github.com/act3-ace/safe-autonomy-sims.git
```
2. Generate a GitLab [personal access token](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html#create-a-personal-access-token)
with at least `read_api` access if you don't already have one (save this token!).
3. Make your local copy of `install.sh` executable:
```bash
# Replace path/to/install.sh with the actual path
chmod +x path/to/install.sh
```
4. **(Recommended)** Activate your desired environment:
```bash
# Example using conda environment named myenv
conda activate myenv
```
5. Run `install.sh` with the developer `-d` option inside the root of the repository,
passing in a personal access token:
```bash
# Run inside safe-autonomy-sims root
# Replace <my-access-token> with your personal access token
./install.sh -p <my-access-token> -d
```

This should install the safe-autonomy-sims package as an
editable pip installation in your local activated environment.
Any changes you make in the repository should be reflected live in
your environment.

## Local Documentation

This repository is setup to use [MKDOCS](https://www.mkdocs.org/)
which is a static site generator geared towards building 
project documentation. Documentation source files are 
written in Markdown, and configured with a single YAML 
configuration file.

**NOTE**: In order to properly build the documentation locally, you must first 
have CoRL and safe-autonomy-sims installed in your container/environment!

Install the Mkdocs modules in a container/virtual environment via pip:
```shell
pip install -U -r mkdocs-requirements.txt
```
To build the documentation locally without serving it, use
the following command from within your container/virtual environment:
```shell
python -m  mkdocs build
```
To serve the documentation on a local port, use the following
command from within your container/virtual environment: 
```shell
python -m mkdocs serve 
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

## Environments

This library includes the following environments:

- Inspection 3D
- Inspection 3D - Multiagent
- Docking 3D
- Docking 3D - Multiagent
- Rejoin 2D
- Rejoin 3D
- Rejoin 3D - Multiagent


### Inspection
Spacecraft inspection scenario where an agent controlled deputy spacecraft must inspect points on a stationary chief spacecraft while both orbit a central body. This is accomplished by approaching and navigating around the chief to view all points on a sphere, assuming the deputy always points a sensor towards the chief. The motion of the deputy spacecraft is governed by the Clohessy-Wiltshire linearized dynamics model. Comes in the following flavors: 

-  **Inspection 3D**
Static 1N thrusters in $\pm x, \pm y, \pm z$.

-  **Inspection 3D - Multiagent**
Multiple agent controlled deputy spaceraft. Each controlled by static 1N thrusters in $\pm x, \pm y, \pm z$.


### Docking
Spacecraft docking scenario where an agent controlled deputy spacecraft must dock with a stationary chief spacecraft while both orbit a central body. This is accomplished by approaching the chief to within a predefined docking distance while maintaining a safe relative velocity within that distance. The motion of the deputy spacecraft is governed by the Clohessy-Wiltshire linearized dynamics model. Comes in the following flavors: 

-  **Docking 3D**
Static 1N thrusters in $\pm x, \pm y, \pm z$.

-  **Docking 3D - Multiagent**
Multiple agent controlled deputy spaceraft. Each controlled by static 1N thrusters in $\pm x, \pm y, \pm z$.


## Team
Jamie Cunningham,
Umberto Ravaioli,
John McCarroll,
Kyle Dunlap,
Nate Hamilton,
Kerianne Hobbs