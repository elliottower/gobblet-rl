from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gobblet-rl",
    version="1.0.0",
    description="A PettingZoo implementation of the Gobblet board game.",
    url="https://github.com/elliottower/gobblet-rl",
    author="Elliot Tower",
    author_email="elliot@elliottower.com",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["gobblet-rl"],
    python_requires=">=3.8",
    install_requires=["pettingzoo"],
    tests_require=["pytest"],
)