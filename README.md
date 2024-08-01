# safe-autonomy-sims

The `safe-autonomy-sims` package provides components and tools to build modular, integration-focused Reinforcement Learning environments with Run Time Assurance (RTA). This repo is designed to work hand-in-glove with the [`corl`](https://github.com/act3-ace/CoRL), [`safe-autonomy-simulation`](https://github.com/act3-ace/safe-autonomy-simulation), and [`run-time-assurance`](https://github.com/act3-ace/run-time-assurance) packages.

## Installation

The following instructions detail how to install the `safe-autonomy-sims` library on your local system. It is recommended to install the python modules within a virtual environment.

The easiest way to install `safe-autonomy-sims` into your environment is via `pip`:

```shell
pip install safe-autonomy-sims
```

## Usage

The `safe-autonomy-sims` package provides RL training environments and example training configurations using [Gymnasium](https://gymnasium.farama.org/), [PettingZoo](https://pettingzoo.farama.org/), and [CoRL](https://github.com/act3-ace/CoRL). These environments are designed to provide challenge problems for safe autonomous control.

### Gym

This package provides the following single-agent Gymnasium environments:

* [`Docking-v0`](safe_autonomy_sims/gym/docking/docking_v0.py)
* [`Inspection-v0`](safe_autonomy_sims/gym/inspection/inspection_v0.py)
* [`WeightedInspection-v0`](safe_autonomy_sims/gym/inspection/weighted_inspection_v0.py)
* [`SixDofInspection-v0`](safe_autonomy_sims/gym/inspection/sixdof_inspection_v0.py)

These environments can be built using the `gymnasium.make()` function:

```python
import gymnasium
import safe_autonomy_sims.gym

# Build the Docking-v0 environment
env = gymnasium.make("Docking-v0")
```

See the [Gymnasium documentation](https://gymnasium.farama.org/) for more information.

### PettingZoo

This package also provides the following multi-agent PettingZoo environments:

* [`MultiDocking-v0`](safe_autonomy_sims/pettingzoo/docking/multidocking_v0.py)
* [`MultiInspection-v0`](safe_autonomy_sims/pettingzoo/inspection/multi_inspection_v0.py)
* [`MultiWeightedInspection-v0`](safe_autonomy_sims/pettingzoo/inspection/weighted_multi_inspection_v0.py)
* [`SixDofMultiInspection-v0`](safe_autonomy_sims/pettingzoo/inspection/sixdof_multi_inspection_v0.py)

These environments can be built using the following syntax:

```python
import safe_autonomy_sims

# Build the MultiDocking-v0 environment
env = safe_autonomy_sims.pettingzoo.MultiDockingEnv()
```

See the [PettingZoo documentation](https://pettingzoo.farama.org/) for more information.

### CoRL

This package provides several environments designed to use the [CoRL](https://github.com/act3-ace/CoRL) library for RL training. The following sections give an overview on using CoRL for training and the provided CoRL-compatible environments.

#### Training

Training experiments using CoRL are conducted in safe-autonomy-sims via [configuration files](configs). These files can be manipulated to define experiment parameters, agent configurations, environments, tasks, policies, and platforms.

The `corl` package provides a training endpoint script which uses the RLLib reinforcement learning library to train agents in an environment.

As an example, you can launch a training loop for the provided Docking  environment using the following command:

```shell
# from root of safe-autonomy-sims
python -m corl.train_rl --cfg configs/docking/experiment.yml
```

Further information on training and experiment configuration can be found [here](docs/configuration.md).

#### Environments

This package includes the following CoRL-compatible environments:

* [Docking](docs/tasks/CWH/docking.md)
* [Multiagent Docking](docs/tasks/CWH/multiagent_docking.md)
* [Translational Inspection](docs/tasks/CWH/translational_inspection.md)
* [Multiagent Translational Inspection](docs/tasks/CWH/multiagent_translational_inspection.md)
* [Weighted Translational Inspection](docs/tasks/CWH/weighted_translational_inspection.md)
* [Multiagent Weighted Translational Inspection](docs/tasks/CWH/multiagent_weighted_translational_inspection.md)
* [Weighted Six DoF Inspection](docs/tasks/CWH/six_dof_inspection.md)
* [Multiagent Weighted Six DoF Inspection](docs/tasks/CWH/multiagent_six_dof_inspection.md)

##### Docking

Spacecraft docking scenario where an agent controlled deputy spacecraft must dock with a stationary chief spacecraft while both orbit a central body. This is accomplished by approaching the chief to within a predefined docking distance while maintaining a safe relative velocity within that distance. The motion of the deputy spacecraft is governed by the Clohessy-Wiltshire linearized dynamics model. Comes in the following flavors:

* **Docking**
Static 1N thrusters in $\pm x, \pm y, \pm z$.

* **Multiagent Docking**
Multiple agent controlled deputy spaceraft. Each controlled by static 1N thrusters in $\pm x, \pm y, \pm z$.

##### Inspection

Spacecraft inspection scenario where an agent controlled deputy spacecraft must inspect points on a stationary chief spacecraft while both orbit a central body. This is accomplished by approaching and navigating around the chief to view all points on a sphere. Points on the sphere can be illuminated by the sun, and only illuminated points can be inspected. Inspection 3D environments assume the deputy always points a sensor towards the chief, while Inspection Six DoF environments allow the deputy to control the orientation of the sensor. The translational motion of the deputy spacecraft is governed by the Clohessy-Wiltshire linearized dynamics model, and the Six DoF environments use a quaternion formulation to model attitude. All have static 1N thrusters in $\pm x, \pm y, \pm z$, and Six DoF environments also have moment controllers in $\pm x, \pm y, \pm z$. Comes in the following flavors:

* **Translational Inspection**
Agent can only control its translational motion. Orientation is assumed to be pointing at the chief. All points are weighted equally.

* **Weighted Translational Inspection**
Agent can only control its translational motion. Points are prioritizied through a directional unit vector, and are assigned weights/scores based on their angular distance to this vector. Inspected points are rewarded based on score. Success is determined by reaching a score threshold, rather than all points inspected.

* **Multiagent Translational Inspection**
Same as translational-inspection environment, with multiple agent controlled deputy spacecraft.

* **Weighted Six DoF Inspection**
Same as translational-inspection environment, but agent can control attitude (does not always point at chief).

* **Multiagent Weighted Six DoF Inspection**
Same as weighted-six-dof-inspection environment, with multiple agent controlled deputy spacecraft.

## Development

If you are interested in contributing to the development of `safe-autonomy-sims`, the following sections outline the recommended process for setting up a development environment and building the package documentation.

### Developer Installation

The `safe-autonomy-sims` library was developed using the python packaging tool `poetry`. It is recommended to perform a local development installation of this project using `poetry` if you plan on contributing.

```shell
git clone <safe-autonomy-sims-url>
cd safe-autonomy-sims
poetry install
```

Poetry will handle installing appropriate dependencies into your environment, if they aren't already installed.  Poetry will also install an editable version of `safe-autonomy-sims` into the environment. For more information on managing Poetry environments see the [official documentation](https://python-poetry.org/docs/managing-environments/).

### Local Documentation

This repository is setup to use [MkDocs](https://www.mkdocs.org/) which is a static site generator geared towards building project documentation. Documentation source files are written in Markdown, and configured with a single YAML configuration file.

**NOTE**: In order to properly build the documentation locally, you must first have `safe-autonomy-sims` and its dependencies installed in your container/environment!

Install the MkDocs modules in a container/virtual environment via Poetry:

```shell
poetry install --with docs
```

To build the documentation locally without serving it, use the following command from within your container/virtual environment:

```shell
poetry run mkdocs build
```

To serve the documentation on a local port, use the following command from within your container/virtual environment:

```shell
poetry run mkdocs serve 
```

## Public Release

Approved for public release; distribution is unlimited. Case Number: AFRL-2023-6156

## Team

Jamie Cunningham,
Umberto Ravaioli,
John McCarroll,
Kyle Dunlap,
Nate Hamilton,
Charles Keating,
Kochise Bennett,
Aditesh Kumar,
Kerianne Hobbs
