import numpy
import numba

import collections


def build_unit_cube_mesh(nx, ny, nz):
    x = numpy.linspace(0, 1, nx + 1)
    y = numpy.linspace(0, 1, ny + 1)
    z = numpy.linspace(0, 1, nz + 1)
    c = numpy.meshgrid(x, y, z, copy=False, indexing='ij')
    vertices = numpy.transpose(c).reshape((nx + 1)*(ny + 1)*(nz + 1), 3)

    cells = numpy.ndarray((6*nx*ny*nz, 4), dtype=numpy.uintc)
    @numba.jit(nopython=True)
    def build_topology(nx, ny, nz, cells):
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    v0 = iz*(nx + 1)*(ny + 1) + iy*(nx + 1) + ix
                    v1 = v0 + 1
                    v2 = v0 + (nx + 1)
                    v3 = v1 + (nx + 1)
                    v4 = v0 + (nx + 1)*(ny + 1)
                    v5 = v1 + (nx + 1)*(ny + 1)
                    v6 = v2 + (nx + 1)*(ny + 1)
                    v7 = v3 + (nx + 1)*(ny + 1)

                    c0 = 6*(iz*nx*ny + iy*nx + ix)
                    cells[c0+0,:] = [v0, v1, v3, v7]
                    cells[c0+1,:] = [v0, v1, v7, v5]
                    cells[c0+2,:] = [v0, v5, v7, v4]
                    cells[c0+3,:] = [v0, v3, v2, v7]
                    cells[c0+4,:] = [v0, v6, v4, v7]
                    cells[c0+5,:] = [v0, v2, v6, v7]
    build_topology(nx, ny, nz, cells)

    Mesh = collections.namedtuple("Mesh", "vertices cells")
    return Mesh(vertices=vertices, cells=cells)
