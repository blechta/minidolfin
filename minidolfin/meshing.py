import FIAT
import numpy
import numba


class Mesh(object):

    def __init__(self, reference_cell, vertices, cells):
        tdim = reference_cell.get_dimension()

        self.reference_cell = reference_cell
        self.vertices = vertices
        self.topology = {(tdim, 0): cells}

        # TODO: Numbify me?
        # TODO: Or rather do the actual sorting?
        try:
            for i in range(cells.shape[0]):
                for j in range(cells.shape[1] - 1):
                    assert cells[i, j+1] > cells[i, j]
        except AssertionError:
            raise ValueError("Cell-vertex topology is not sorted")

        # FIXME: Compute only needed connectivity
        for d in range(1, tdim):
            self.topology[(tdim, d)], self.topology[(d, 0)] = \
                    self._compute_connectivity_tdim_d_0(d)
        self._compute_boundary_facets()

    def get_connectivity(self, dim0, dim1):
        if dim0 == dim1:
            num_entities = self.num_entities(dim1)
            return numpy.arange(num_entities,
                                dtype=numpy.uintc).reshape(num_entities, 1)
        else:
            try:
                return self.topology[(dim0, dim1)]
            except KeyError:
                raise ValueError("Connectivity {}-{} has not been computed"
                                 .format(dim0, dim1))

    def num_entities(self, dim):
        if dim == 0:
            return self.vertices.shape[0]
        else:
            return self.get_connectivity(dim, 0).shape[0]

    def _compute_connectivity_tdim_d_0(self, d):
        # Fetch data
        tdim = self.reference_cell.get_dimension()
        cell_vertex_connectivity = self.topology[(tdim, 0)]
        ent_vert_conn_local = self.reference_cell.get_connectivity()[(d, 0)]
        ent_per_cell = len(ent_vert_conn_local)
        vertices_per_ent = len(ent_vert_conn_local[0])
        num_cells = self.num_entities(tdim)

        # Compute ent-vertex connectivity cell-by-cell
        ent_vert_conn = cell_vertex_connectivity[:, ent_vert_conn_local] \
            .reshape(ent_per_cell*num_cells, vertices_per_ent)

        # Gather cells togeter, pick unique, that gives ent-numbering
        tmp = {}
        for ent_ind, ent_verts in enumerate(ent_vert_conn):
            tmp.setdefault(tuple(ent_verts), []).append(ent_ind)
        cell_ent_conn = numpy.ndarray((ent_vert_conn.shape[0],),
                                      dtype=ent_vert_conn.dtype)
        ent_vert_conn = numpy.ndarray((len(tmp), ent_vert_conn.shape[1]),
                                      dtype=ent_vert_conn.dtype)
        cnt = 0
        for ent_verts, ent_inds in tmp.items():
            ent_vert_conn[cnt] = ent_verts
            cell_ent_conn[ent_inds] = cnt
            cnt += 1
        del tmp
        # NB: With NumPy >= 1.13 just maybe:
        # ent_vert_conn, cell_ent_conn = numpy.unique(ent_vert_conn,
        #                                    axis=0, return_inverse=True)

        # Adapt data into desired shape
        cell_ent_conn = cell_ent_conn.reshape(num_cells, ent_per_cell)

        return cell_ent_conn, ent_vert_conn

    def _compute_boundary_facets(self):
        # Fetch data
        tdim = self.reference_cell.get_dimension()
        cell_facet_connectivity = self.get_connectivity(tdim, tdim - 1)

        # Determine boundary facets by counting the number of cells
        # they apper in (1 = boundary, 2 = interior)
        counts = numpy.bincount(cell_facet_connectivity.flat)
        self.boundary_facets = set(f for f,
                                   count in enumerate(counts) if count == 1)


def build_unit_cube_mesh(nx, ny, nz):
    x = numpy.linspace(0, 1, nx + 1)
    y = numpy.linspace(0, 1, ny + 1)
    z = numpy.linspace(0, 1, nz + 1)
    c = numpy.meshgrid(x, y, z, copy=False, indexing='ij')
    vertices = numpy.transpose(c).reshape((nx + 1) * (ny + 1) * (nz + 1), 3)
    cells = numpy.ndarray((6 * nx * ny * nz, 4), dtype=numpy.uintc)

    @numba.jit(nopython=True)
    def build_topology(nx, ny, nz, cells):
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    v0 = iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix
                    v1 = v0 + 1
                    v2 = v0 + (nx + 1)
                    v3 = v1 + (nx + 1)
                    v4 = v0 + (nx + 1) * (ny + 1)
                    v5 = v1 + (nx + 1) * (ny + 1)
                    v6 = v2 + (nx + 1) * (ny + 1)
                    v7 = v3 + (nx + 1) * (ny + 1)

                    c0 = 6 * (iz * nx * ny + iy * nx + ix)
                    cells[c0+0, :] = [v0, v1, v3, v7]
                    cells[c0+1, :] = [v0, v1, v5, v7]
                    cells[c0+2, :] = [v0, v4, v5, v7]
                    cells[c0+3, :] = [v0, v2, v3, v7]
                    cells[c0+4, :] = [v0, v4, v6, v7]
                    cells[c0+5, :] = [v0, v2, v6, v7]
    build_topology(nx, ny, nz, cells)

    fiat_cell = FIAT.reference_element.ufc_cell("tetrahedron")

    return Mesh(fiat_cell, vertices, cells)


def build_unit_square_mesh(nx, ny):
    x = numpy.linspace(0, 1, nx + 1)
    y = numpy.linspace(0, 1, ny + 1)
    c = numpy.meshgrid(x, y, copy=False, indexing='ij')
    vertices = numpy.transpose(c).reshape((nx + 1)*(ny + 1), 2)
    cells = numpy.ndarray((2*nx*ny, 3), dtype=numpy.uintc)

    @numba.jit(nopython=True)
    def build_topology(nx, ny, cells):
        for iy in range(ny):
            for ix in range(nx):
                v0 = iy * (nx + 1) + ix
                v1 = v0 + 1
                v2 = v0 + (nx + 1)
                v3 = v1 + (nx + 1)

                c0 = 2*(iy*nx + ix)
                cells[c0+0, :] = [v0, v1, v2]
                cells[c0+1, :] = [v1, v2, v3]
    build_topology(nx, ny, cells)

    fiat_cell = FIAT.reference_element.ufc_cell("triangle")

    return Mesh(fiat_cell, vertices, cells)


def get_mesh_from_url(url):
    """ Read a 2D triangle mesh in XDMF XML format (not HDF5) from a URL"""
    try:
        import requests
        import xml.etree.ElementTree as ET
    except ImportError:
        raise("Missing required library")

    r = requests.get(url)
    if r.status_code != 200:
        raise IOError("Cannot read from URL")

    et = ET.fromstring(r.text)
    assert(et.tag == 'Xdmf')
    assert(et[0].tag == 'Domain')
    assert(et[0][0].tag == 'Grid')
    grid = et[0][0]

    # Get topology array
    topology = grid.find('Topology')
    assert(topology.attrib['TopologyType'] == 'Triangle')
    tdims = numpy.fromstring(topology[0].attrib['Dimensions'],
                             sep=' ', dtype='int')
    nptopo = numpy.fromstring(topology[0].text,
                              sep=' ', dtype='int').reshape(tdims)

    # Get geometry array
    geometry = grid.find('Geometry')
    assert(geometry.attrib['GeometryType'] == 'XY')
    gdims = numpy.fromstring(geometry[0].attrib['Dimensions'],
                             sep=' ', dtype='int')
    npgeom = numpy.fromstring(geometry[0].text,
                              sep=' ', dtype='float').reshape(gdims)

    # Find all attributes and put them in a list
    attrlist = grid.findall('Attribute')
    data_all = []
    for attr in attrlist:
        #        adims = numpy.fromstring(attr[0].attrib['Dimensions'],
        #                        sep=' ', dtype='int')
        npattr = attr.attrib
        npattr['value'] = numpy.fromstring(attr[0].text, sep=' ', dtype='int')
        data_all.append(npattr)

    fiat_cell = FIAT.reference_element.ufc_cell("triangle")
    mesh = Mesh(fiat_cell, npgeom, nptopo)

    # Attach any data
    mesh.data = data_all

    return mesh
