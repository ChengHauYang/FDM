import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from matplotlib import cm
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams.update({'font.size': 14, 'font.family': 'serif'})
matplotlib.rcParams.update({'text.usetex': 'true'})

TotalT = 0.01
l = 8
Nx = 2**l - 1
All_N = Nx + 2
N = All_N
dx = 1 / (All_N - 1)
dy = 1 / (All_N - 1)

x, y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

# IC, Exact solution
U = np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y)
U_star = np.copy(U)
Uexact = U * np.exp(-32 * np.pi**2 * TotalT)


# BC
U[:, [0, -1]] = 0
U[[0, -1], :] = 0
U_star[[0, -1], :] = 0
U_star[:, [0, -1]] = 0

dt = 0.01 / 2**l

ax = 1 * dt / (2 * dx**2)
ay = 1 * dt / (2 * dy**2)

CurrentTime = 0

# build A1 and A2
main_diag_A1 = (1 + 2 * ax) * np.ones(N)
sub_diag_A1 = -ax * np.ones(N - 1)
super_diag_A1 = -ax * np.ones(N - 1)
main_diag_A1[[0, 0]] = 1
super_diag_A1[[0, 0]] = 0
main_diag_A1[[N-1, N-1]] = 1
sub_diag_A1[[N-2, N-2]] = 0
A1 = diags([sub_diag_A1, main_diag_A1, super_diag_A1], [-1, 0, 1])

main_diag_A2 = (1 + 2 * ay) * np.ones(N)
sub_diag_A2 = -ay * np.ones(N - 1)
super_diag_A2 = -ay * np.ones(N - 1)
main_diag_A2[[0, 0]] = 1
super_diag_A2[[0, 0]] = 0
main_diag_A2[[N-1, N-1]] = 1
sub_diag_A2[[N-2, N-2]] = 0
A2 = diags([sub_diag_A2, main_diag_A2, super_diag_A2], [-1, 0, 1])

for t_ in range(int(TotalT / dt)):
    CurrentTime += dt
    for j in range(1, N - 1):  # Y

        # print(A1.todense())
        b1 = np.zeros(N)
        b1[0] = U[0, j]
        b1[N - 1] = U[N - 1, j]
        b1[1:N - 1] = ay * U[1:N - 1, j + 1] + \
            (1 - 2 * ay) * U[1:N - 1, j] + ay * U[1:N - 1, j - 1]
        # print(b1)

        U_star[:, j] = spsolve(A1, b1)

    for i in range(1, N - 1):  # X
        # print(A2.todense())
        b2 = np.zeros(N)
        b2[0] = U[i, 0]
        b2[N - 1] = U[i, N - 1]
        b2[1:N - 1] = ax * U_star[i + 1, 1:N - 1] + \
            (1 - 2 * ax) * U_star[i, 1:N - 1] + ax * U_star[i - 1, 1:N - 1]
        # print(b2)

        U[i, :] = spsolve(A2, b2)


Error = np.abs(Uexact - U)

plt.pcolormesh(x, y, U, cmap='rainbow')
plt.colorbar()

plt.title(r'pcolormesh of solution')
plt.xlabel(r'x')
plt.ylabel(r'y')
#plt.show()
plt.savefig('Pcolor_hw9.pdf',format='pdf',bbox_inches='tight')

# error calculation
UmunusUxactL2 = np.sqrt(dx * np.sum(Error**2))
UL2 = np.sqrt(dx * np.sum(Uexact**2))

Error = UmunusUxactL2/UL2

print("Error = %20.15f" % Error)

## log - log scale
# Define the range of mesh sizes
l_values = np.arange(5, 9)
dx_values = 1 / (2 ** l_values - 1)

errors = [0.040563282526510, 0.010024957668893,0.002499035255411, 0.000624309138540]

# Plot the error with respect to dx on a log-log scale
plt.figure()
plt.loglog(dx_values, errors, 'o-', markerfacecolor='none')
plt.xlabel(r'$h_x$')
plt.ylabel('Error')
plt.title('Error with respect to $h_x$')
plt.grid(True, which="both", ls="--")
#plt.show()
plt.savefig('MeshConvergence.pdf',format='pdf',bbox_inches='tight')
