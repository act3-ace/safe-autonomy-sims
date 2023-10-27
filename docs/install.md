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
safe-autonomy-sims.  Poetry can install safe-autonomy-sims and its dependendencies
into an auto-generated virtualenv or within the currently active environment.

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

Install the safe-autonomy-sims module into your
environment using `poetry`:

```shell
cd safe-autonomy-sims
poetry install
```

Poetry will handle installing appropriate versions of the dependencies for safe-autonomy-dynamics into your environment, if they aren't already installed.  Poetry will install an editable version of safe-autonomy-sims to the environment.

## Questions or Issues?

If you have any trouble installing the safe-autonomy-sims
package in your local environment, please feel free to
submit an [issue]({{git_url}}/safe-autonomy-sims/issues).

For more information on what's available in safe-autonomy-sims,
see our [API](api/index.md).
