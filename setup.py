#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import setup, find_packages


def parse_requirements(filename: str):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

reqs = parse_requirements("requirements.txt")


if __name__ == '__main__':
    tests_require = [
        'flake8',
        'mypy',
        'mypy-extensions',
        'mypy-protobuf',
        'pylint',
        'pytest',
        'pytest-mock',
        'pytest-cov',
        'pytest-order',
        'yapf',
        'isort',
        'rope',
        'pre-commit',
        'pre-commit-hooks',
        'detect-secrets',
        'blacken-docs',
        'bashate',
        'fish',
        'watchdog',
        'speedscope',
        'pandas-profiling',
        'factory',
    ]

    docs_require = parse_requirements("mkdocs-requirements.txt")

    setup(
        name="safe-autonomy-sims",
        author="ACT3",
        description="ACT3 Safe Autonomy RL Benchmarks",

        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",

        url="https://github.com/act3-ace/safe-autonomy-sims",

        license="",

        setup_requires=[
            'setuptools_scm',
            'pytest-runner'
        ],
        use_scm_version={
            'fallback_version': '0.0.0',
        },

        # add in package_data
        include_package_data=True,
        package_data={
            'saferl': ['*.yml', '*.yaml']
        },

        packages=find_packages(),

        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],

        install_requires=reqs,

        extras_require={
            "testing":  tests_require,
            "docs":  docs_require,
        },
        python_requires='>=3.8',
    )


