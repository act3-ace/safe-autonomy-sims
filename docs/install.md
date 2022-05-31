# Installation
The following instructions detail how to install 
the safe-autonomy-sims library on your local system.
It is recommended to install the python modules within 
a [virtualenv](https://virtualenv.pypa.io/en/stable/#)
or [conda](https://docs.conda.io/projects/conda/en/latest/index.html)
environment.

The safe-autonomy-sims library is built on the 
[CoRL](https://github.com/act3-ace/act3-rl/corl)
reinforcement learning framework using the 
[run-time-assurance](https://github.com/act3-ace/run-time-assurance)
and [safe-autonomy-dynamics](https://github.com/act3-ace/safe-autonomy-dynamics) 
packages. These dependencies are required in order to use
safe-autonomy-sims. At the time of writing,
none of these dependencies are available on any public package
repos and must be installed from source.

## Installing CoRL
Clone a copy of the CoRL source code onto
your local machine via SSH:
```shell
git clone git@github.com/act3-ace:act3-rl/corl.git
```
or HTTPS:
```shell
git clone https://github.com/act3-ace/act3-rl/corl.git
```

Install the CoRL module into your environment using `pip`:
```shell
pip install path/to/corl/
```

## Installing run-time-assurance
Clone a copy of the run-time-assurance source code onto your local machine via SSH:
```shell
git clone git@github.com/act3-ace:rta/run-time-assurance.git
```
or HTTPS:
```shell
git clone https://github.com/act3-ace/run-time-assurance.git
```

Install the run-time-assurance module into your 
environment using `pip`:
```shell
pip install path/to/run-time-assurance/
```

## Installing safe-autonomy-dynamics
Clone a copy of the safe-autonomy-dynamics source code onto your local machine via SSH:
```shell
git clone git@github.com/act3-ace:rta/safe-autonomy-dynamics.git
```
or HTTPS:
```shell
git clone https://github.com/act3-ace/safe-autonomy-dynamics.git
```

Install the safe-autonomy-dynamics module into your 
environment using `pip`:
```shell
pip install path/to/safe-autonomy-dynamics/
```

## Installing safe-autonomy-sims
Once CoRL is installed in a local environment, clone a
copy of the safe-autonomy-sims source code onto your local
machine via SSH (recommended):
```shell
git clone git@github.com/act3-ace:rta/safe-autonomy-sims.git
```
or HTTPS:
```shell
git clone https://github.com/act3-ace/safe-autonomy-sims.git
```

Install the safe-autonomy-sims module into your local
environment using `pip`:
```shell
pip install path/to/safe-autonomy-sims/
```

## Development
For a local development version, you can install the
safe-autonomy-sims package using `pip`'s 
`-e, --editable` option:
```shell
pip install -e path/to/safe-autonomy-sims/
```
This will install the package in an editable mode within
your environment, allowing any changes you make to the
source to persist.

## Questions or Issues?
If you have any trouble installing the safe-autonomy-sims
package in your local environment, please feel free to
submit an [issue](https://github.com/act3-ace/safe-autonomy-sims/-/issues).

For more information on what's available in safe-autonomy-sims,
see our [API](api/index.md).
