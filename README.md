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
It is recommended to install the python modules within
a [virtualenv](https://virtualenv.pypa.io/en/stable/#)
or [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.
safe-autonomy-sims utilizes [Poetry](https://python-poetry.org/) to handle installation.
Poetry can install safe-autonomy-sims into an auto-generated virtualenv or within the currently active environment.

1. Clone the safe-autonomy-sims repository

    ```bash
    # via SSH (recommended)
    git clone git@github.com/act3-ace:safe-autonomy-sims.git
    ```

    ```bash
    # via HTTPS
    git clone https://github.com/act3-ace/safe-autonomy-sims.git
    ```

2. **(Recommended)** Activate your desired environment:

    ```bash
    # Example using conda environment named myenv
    conda activate myenv
    ```

3. Install the safe-autonomy-sims module into your environment using `poetry`:

    ```bash
    cd safe-autonomy-sims
    poetry install
    ```

    Poetry will handle installing appropriate versions of the dependencies for safe-autonomy-dynamics into your environment, if they aren't already installed.  Poetry will install an editable version of safe-autonomy-sims to the environment.

## Local Documentation

This repository is setup to use [MkDocs](https://www.mkdocs.org/)
which is a static site generator geared towards building
project documentation. Documentation source files are
written in Markdown, and configured with a single YAML
configuration file.

**NOTE**: In order to properly build the documentation locally, you must first
have CoRL and safe-autonomy-sims installed in your container/environment!

Install the MkDocs modules in a container/virtual environment via Poetry:

```shell
poetry install --with docs
```

To build the documentation locally without serving it, use
the following command from within your container/virtual environment:

```shell
poetry run mkdocs build
```

To serve the documentation on a local port, use the following
command from within your container/virtual environment:

```shell
poetry run mkdocs serve 
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
python -m corl.train_rl_ --cfg /path/to/safe-autonomy-sims/configs/cw3hd/docking/experiments/experiment.yml --compute-platform local
```

## Environments

This library includes the following environments:

- Translational Inspection v0:
- Translational Inspection v1:
- Translational Inspection v2:
- Translational Inspection Multiagent v2:
- Six DoF Inspection v2:
- Six DoF Inspection Multiagent v2:
- Docking 3D
- Docking 3D - Multiagent

### Inspection

Spacecraft inspection scenario where an agent controlled deputy spacecraft must inspect points on a stationary chief spacecraft while both orbit a central body. This is accomplished by approaching and navigating around the chief to view all points on a sphere. Inspection 3D environments assume the deputy always points a sensor towards the chief, while Inspection Six DoF environments allow the deputy to control the orientation of the sensor. The translational motion of the deputy spacecraft is governed by the Clohessy-Wiltshire linearized dynamics model, and the Six DoF environments use a quaternion formulation to model attitude. All have static 1N thrusters in $\pm x, \pm y, \pm z$, and Six DoF environments also have moment controllers in $\pm x, \pm y, \pm z$. Comes in the following flavors:

- **Translational Inspection v0**
No illumination from sun. Agent can inspect any point at any time.

- **Translational Inspection v1**
Sun illuminates the chief, which is assumed to rotate in the $x-y$ plane. Agent can only inspect illuminated points.

- **Translational Inspection v2**
Sun illuminates the chief. Points are prioritizied through a directional unit vector, and are assigned weights/scores based on their angular distance to this vector. Inspected points are rewarded based on score. Success is determined by reaching a score threshold, rather than all points inspected.

- **Translational Inspection Multiagent v2**
Same as translational v2 environment, with multiple agent controlled deputy spacecraft.

- **Six DoF Inspection v2**
Same as translational v2 environment, but agent can control attitude (does not always point at chief).

- **Six DoF Inspection Multiagent v2**
Same as Six DoF v2 environment, with multiple agent controlled deputy spacecraft.

### Docking

Spacecraft docking scenario where an agent controlled deputy spacecraft must dock with a stationary chief spacecraft while both orbit a central body. This is accomplished by approaching the chief to within a predefined docking distance while maintaining a safe relative velocity within that distance. The motion of the deputy spacecraft is governed by the Clohessy-Wiltshire linearized dynamics model. Comes in the following flavors:

- **Docking 3D**
Static 1N thrusters in $\pm x, \pm y, \pm z$.

- **Docking 3D - Multiagent**
Multiple agent controlled deputy spaceraft. Each controlled by static 1N thrusters in $\pm x, \pm y, \pm z$.

## Team

Jamie Cunningham,
Umberto Ravaioli,
John McCarroll,
Kyle Dunlap,
Nate Hamilton,
Kerianne Hobbs
