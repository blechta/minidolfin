To install in FEniCS `dev-env` Docker image::

    git clone https://github.com/blechta/minidolfin.git
    fenicsproject run quay.io/fenicsproject/dev-env

and in the container::

    pip3 install --user -r requirements.txt
    pip3 install --user -e .
    export LD_LIBRARY_PATH=/usr/local/petsc-32/lib:$LD_LIBRARY_PATH

To run a demo::

    cd demo
    python3 helmholtz.py
