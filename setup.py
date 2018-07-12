try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import sys

if sys.version_info < (3, 0):
    print("Python 3.0 or higher required, please upgrade.")
    sys.exit(1)

version = "2018.1.0.dev0"

setup(name="minidolfin",
      description="Minimal FE library",
      version=version,
      author="Jan Blechta",
      author_email="blechta@karlin.mff.cuni.cz",
      license="LGPL v3 or later",
      packages=["minidolfin"],
      install_requires=["tsfc", "fenics-ffc", "numba", "petsc4py", "matplotlib"])
