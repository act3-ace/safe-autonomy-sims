#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import setup


if __name__ == '__main__':
    tests_require = [
        'flake8',
        'mypy',
        'mypy-extensions',
        'mypy-protobuf',
        'pylint',
        'pytest',
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
        'pandas-profiling'
    ]

    docs_require = [
        'mkdocs',
        'mkdocs-macros-plugin',
        'mkdocs-mermaid-plugin',
        'inari[mkdocs]',
        'pymdown-extensions',
    ]

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
            'safe_autonomy_sims': ['*.yml', '*.yaml']
        },

        packages=[
                'safe_autonomy_sims',
        ],

        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],

        install_requires=[
            'act3-rl-core'
        ],
        extras_require={
            "testing":  tests_require,
            "docs":  docs_require,
        },
        python_requires='>=3.8',
    )


