# Installation

The following instructions detail how to install
the safe-autonomy-sims library on your local system. It is recommended to install the python modules within a [virtualenv](https://virtualenv.pypa.io/en/stable/#) or [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.

These instructions assumes installation using [Poetry](https://python-poetry.org/docs/), a tool for dependency management and packaging in Python. Poetry will automatically create a virtual environment for you during installation if one does not already exist. If you don't wish to use Poetry, any `poetry install` commands can be safely replaced by a standard `pip install` command.

## Installing safe-autonomy-sims

Install the safe-autonomy-sims module into your
environment using `poetry`:

```shell
cd safe-autonomy-sims
poetry install
```

Poetry will handle installing appropriate dependencies into your environment, if they aren't already installed.  Poetry will install an editable version of safe-autonomy-sims to the environment. For more information on managing Poetry environments see the [official documentation](https://python-poetry.org/docs/managing-environments/).

## Questions or Issues?

If you have any trouble installing the safe-autonomy-sims package in your local environment, please feel free to submit an [issue]({{git_url}}/safe-autonomy-sims/issues).
