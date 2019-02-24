# minidolfin
# Copyright (C) 2019 Chris Richardson and Jan Blechta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import matplotlib.pyplot as plt
import matplotlib.tri as tri


def plot(mesh, *args, **kwargs):
    """ Plot a mesh with matplotlib, possibly with associated data,
        which may be associated with points or triangles """

    # FIXME: check keys of mesh contain geometry and topology
    tdim = mesh.reference_cell.get_dimension()
    geom, topo = mesh.vertices, mesh.topology[(tdim, 0)]
    x = geom[:, 0]
    y = geom[:, 1]

    plt.gca(aspect='equal')

    if args:
        data = args[0]
        if len(data) == len(geom):
            plt.tricontourf(x, y, topo, data, 40, **kwargs)
        elif len(data) == len(topo):
            tr = tri.Triangulation(x, y, topo)
            plt.tripcolor(tr, data, **kwargs)
        else:
            raise RuntimeError("Data is wrong length")

    plt.triplot(x, y, topo, color='k', alpha=0.5)

    xmax = x.max()
    xmin = x.min()
    ymax = y.max()
    ymin = y.min()
    dx = 0.1 * (xmax - xmin)
    dy = 0.1 * (ymax - ymin)
    plt.xlim(xmin - dx, xmax + dx)
    plt.ylim(ymin - dy, ymax + dy)

    return
