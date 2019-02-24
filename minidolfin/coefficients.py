# minidolfin
# Copyright (C) 2019 Chris Richardson and Jan Blechta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl
import numpy
import ffc.codegeneration.jit as ffc_jit

def attach_coefficient_values(coefficient, values):

    # Get compiled element for this coefficient
    compiled_elements, module = ffc_jit.compile_elements(
        [coefficient.ufl_element()], parameters={})
    elem = compiled_elements[0][0]

    if len(values.shape) != 2 or values.shape[1] != elem.space_dimension:
        raise ValueError("values must be 2D array with {} dofs per cell"
                         .format(elem.space_dimension))

    # Patch in values
    coefficient._values = values
