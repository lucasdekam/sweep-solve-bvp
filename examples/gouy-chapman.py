import numpy as np
from bvpsweep import sweep_solve_bvp


def dirichlet(ya, yb, y0):
    return np.array(
        [
            ya[0] - y0,
            yb[0],
        ]
    )


def ode_rhs(x, y):
    dy1 = y[1, :]
    dy2 = np.sinh(y[0, :])
    return np.vstack([dy1, dy2])


x = np.linspace(0, 1, 100)
y_initial = np.zeros((2, 100))
y_pzc = 0
y_electrode = np.linspace(-1, 1, 50)

sol = sweep_solve_bvp(
    fun=ode_rhs,
    bc=dirichlet,
    x0=x,
    y0=y_initial,
    sweep_par=y_electrode,
    sweep_par_start=y_pzc,
)

print(sol.shape)
