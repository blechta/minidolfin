from petsc4py import PETSc


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
