# Installation
The following instructions detail how to install 
the safe-autonomy-sims library on your local system.
It is recommended to install the python modules within 
a [virtualenv](https://virtualenv.pypa.io/en/stable/#)
or [conda](https://docs.conda.io/projects/conda/en/latest/index.html)
environment.

The safe-autonomy-sims library is built on the 
[CoRL]({{corl_url}}/corl)
reinforcement learning framework using the 
[run-time-assurance]({{git_url}}/run-time-assurance)
and [safe-autonomy-dynamics]({{git_url}}/safe-autonomy-dynamics) 
packages. These dependencies are required in order to use
safe-autonomy-sims. At the time of writing,
none of these dependencies are available on any public package
repos and must be installed from source.

## Installing CoRL
Clone a copy of the CoRL source code onto
your local machine via SSH:
```shell
git clone {{corl_ssh}}/corl.git
```
or HTTPS:
```shell
git clone {{corl_url}}/corl.git
```

### CoRL Version
CoRL is a developing library which is continuously introducing new
features. We have pinned the version of CoRL which is compatible
with safe-autonomy-sims to prevent breaking changes from CoRL. 

If you receive an error regarding the required CoRL version needed for
safe-autonomy-sims, you can use the correct version by running the
following command:
```shell
cd path/to/corl/
git checkout tags/vX.XX.X  # replace with needed version number
```

Once you have the correct version checked out, install the CoRL
module into your environment using `pip`:
```shell
pip install path/to/corl/
```

## Installing safe-autonomy-dynamics
Clone a copy of the safe-autonomy-dynamics source code onto your local machine via SSH:
```shell
git clone {{git_ssh}}/safe-autonomy-dynamics.git
```
or HTTPS:
```shell
git clone {{git_url}}/safe-autonomy-dynamics.git
```

Install the safe-autonomy-dynamics module into your 
environment using `pip`:
```shell
pip install path/to/safe-autonomy-dynamics/
```

## Installing run-time-assurance
Clone a copy of the run-time-assurance source code onto your local machine via SSH:
```shell
git clone {{git_ssh}}/run-time-assurance.git
```
or HTTPS:
```shell
git clone {{git_url}}/run-time-assurance.git
```

Install the run-time-assurance module into your 
environment using `pip`:
```shell
pip install path/to/run-time-assurance/
```

## Installing safe-autonomy-sims
Once CoRL is installed in a local environment, clone a
copy of the safe-autonomy-sims source code onto your local
machine via SSH (recommended):
```shell
git clone {{git_ssh}}/safe-autonomy-sims.git
```
or HTTPS:
```shell
git clone {{git_url}}/safe-autonomy-sims.git
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
submit an [issue]({{git_url}}/safe-autonomy-sims/-/issues).

For more information on what's available in safe-autonomy-sims,
see our [API](api/index.md).
