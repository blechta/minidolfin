from petsc4py import PETSc
import numpy


def create_matrix_from_csr(csr):

    # Prepare parameters
    i, j = csr
    m = n = i.size - 1
    bs = 1

    # Allocate matrix
    A = PETSc.Mat().createAIJ((m, n), bs)
    A.setPreallocationCSR((i, j))

    # Forbid further nonzero insertions
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, True)

    return A


def create_vector(N):
    return PETSc.Vec().createSeq(N, bsize=1)


def create_scalar():
    return numpy.zeros(shape=(), dtype=PETSc.ScalarType)


if PETSc.Sys.getVersion()[0:2] <= (3, 8) and PETSc.Sys.getVersionInfo()['release']:
    def set_solver_package(pc, package):
        pc.setFactorSolverPackage(package)
else:
    def set_solver_package(pc, package):
        pc.setFactorSolverType(package)
