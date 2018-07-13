==========
minidolfin
==========

minidolfin is experimental minimal FE library written in
Python and using JIT-compilation techniques of FEniCS stack,
TSFC, Numba.

Currently it uses PETSc as a linear algebra backend which
is maybe a bit too heavy weight considering that minidolfin
does not run with MPI parallelization (yet).

Getting started
===============

To install in FEniCS `dev-env` Docker image::

    git clone https://github.com/blechta/minidolfin.git
    cd minidolfin
    fenicsproject run quay.io/fenicsproject/dev-env

and in the container::

    pip3 install --user -r requirements.txt
    pip3 install --user -e .

To run a demo::

    cd demo
    python3 helmholtz.py

Authors
=======

* @blechta
* @w1th0utnam3

License
=======

GNU LGPL v3 or any later version: `COPYING`_, `COPYING.LESSER`_.

Testing
=======

.. image:: https://circleci.com/gh/blechta/minidolfin.svg?style=svg
    :target: https://circleci.com/gh/blechta/minidolfin
