import setuptools

setuptools.setup(
    name="saferl",
    version="0.0.1",
    author="",
    author_email="",
    description="Contains implementation of the  compatible SafeRL environments",
    packages=setuptools.find_packages(include=["saferl"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6, <3.9",
    install_requires=[
        "act3-rl-core",
    ],
)
