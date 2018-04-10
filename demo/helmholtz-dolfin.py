from dolfin import *
import matplotlib.pyplot as plt


parameters['form_compiler']['representation'] = 'tsfc'

class PlaneWave2D(UserExpression):
    def eval(self, values, x):
        u = cos(-1.5*x[0] + x[1])
        values[0] = u

# Prepare mesh and function space
mesh = UnitSquareMesh(256, 256)
V = FunctionSpace(mesh, "P", 3)

# Prepare exaxt plane wave solution and its wave number
u0 = PlaneWave2D(domain=mesh, element=V.ufl_element())
omega2 = 1.5**2 + 1.0**2

# Prepare variational problem
u, v = TrialFunction(V), TestFunction(V)
a = (inner(grad(u), grad(v)) - Constant(omega2)*inner(u, v))*dx
L = inner(Constant(0), v)*dx

# Solve
u = Function(V)
bc = DirichletBC(V, u0, "on_boundary")
solve(a == L, u, bc, solver_parameters={'linear_solver': 'mumps'})

list_timings(TimingClear.clear, [TimingType.wall])

plot(u)
plt.show()
