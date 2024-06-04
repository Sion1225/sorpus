import toml
import re

from src.sorpus.__version__ import __version__


# pyproject.toml
def get_version_from_pyproject():
    with open("pyproject.toml") as f:
        pyproject_data = toml.load(f)
    return pyproject_data["project"]["version"]


# README.md
def get_version_from_readme():
    with open("README.md") as f:
        for line in f:
            if "PyPI" in line:
                version = re.search(r"v(.+?)-", line).group(1)
                return version


# version check
def test_version():
    assert __version__ == get_version_from_pyproject() == get_version_from_readme()
