import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from matplotlib import cm
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams.update({'font.size':14,'font.family':'serif'})
matplotlib.rcParams.update({'text.usetex':'true'})

# def TDMAsolver(a, b, c, d):
#     n = len(d)
#     c_prime = np.zeros(n)
#     d_prime = np.zeros(n)
#
#     # Forward elimination
#     c_prime[0] = c[0] / b[0]
#     d_prime[0] = d[0] / b[0]
#
#     for i in range(1, n-1):
#         denominator = b[i] - a[i - 1] * c_prime[i - 1]
#         c_prime[i] = c[i] / denominator
#         d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denominator
#
#     # Backward substitution
#     x = np.zeros(n)
#     x[-1] = d_prime[-1]
#
#     for i in range(n - 2, -1, -1):
#         x[i] = d_prime[i] - c_prime[i] * x[i + 1]
#
#     return x

TotalT = 0.01
l = 7
Nx = 2**l - 1
All_N = Nx + 2
N = All_N
dx = 1 / (All_N - 1)
dy = 1 / (All_N - 1)

x, y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
# p = np.column_stack((x.flatten(), y.flatten()))
# t = np.array([[1, 2, N + 2], [1, N + 2, N + 1]])
# t = np.kron(t, np.ones((N - 1, 1))) + np.kron(np.ones(t.shape), np.arange(0, N - 1)[:, None])
# t = np.kron(t, np.ones((N - 1, 1))) + np.kron(np.ones(t.shape), (np.arange(0, N - 1) * N)[:, None])

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
    # for j in range(1, N - 1):  # Y
    #     A1 = np.zeros((N, N))
    #     b1 = np.zeros(N)
    #     A1[0, 0] = 1
    #     A1[N-1, N-1] = 1
    #     b1[0] = U[0, j]
    #     b1[N-1] = U[N-1, j]
    #
    #     for i in range(1, N - 1):  # X
    #         A1[i, i - 1] = -ax
    #         A1[i, i] = 1 + 2 * ax
    #         A1[i, i + 1] = -ax
    #         b1[i] = ay * U[i, j + 1] + \
    #             (1 - 2 * ay) * U[i, j] + ay * U[i, j - 1]
    #
    #     #U_star[:, j] = np.linalg.solve(A1, b1)
    #     U_star[:, j] = TDMAsolver(A1.diagonal(-1), A1.diagonal(), A1.diagonal(1), b1)
    #
    #
    # for i in range(1, N - 1):  # X
    #     A2 = np.zeros((N, N))
    #     b2 = np.zeros(N)
    #     A2[0, 0] = 1
    #     A2[N-1, N-1] = 1
    #     b2[0] = U[i, 0]
    #     b2[N-1] = U[i, N-1]
    #
    #     for j in range(1, N - 1):  # Y
    #         A2[j, j - 1] = -ay
    #         A2[j, j] = 1 + 2 * ay
    #         A2[j, j + 1] = -ay
    #         b2[j] = ax * U_star[i + 1, j] + \
    #             (1 - 2 * ax) * U_star[i, j] + ax * U_star[i - 1, j]
    #
    #     #U[i, :] = np.linalg.solve(A2, b2)
    #     U[i, :] = TDMAsolver(A2.diagonal(-1), A2.diagonal(), A2.diagonal(1), b2)
    for j in range(1, N - 1):  # Y

        #print(A1.todense())
        b1 = np.zeros(N)
        b1[0] = U[0, j]
        b1[N - 1] = U[N - 1, j]
        b1[1:N - 1] = ay * U[1:N - 1, j + 1] + (1 - 2 * ay) * U[1:N - 1, j] + ay * U[1:N - 1, j - 1]
        #print(b1)

        U_star[:, j] = spsolve(A1, b1)

    for i in range(1, N - 1):  # X
        #print(A2.todense())
        b2 = np.zeros(N)
        b2[0] = U[i, 0]
        b2[N - 1] = U[i, N - 1]
        b2[1:N - 1] = ax * U_star[i + 1, 1:N - 1] + (1 - 2 * ax) * U_star[i, 1:N - 1] + ax * U_star[i - 1, 1:N -1]
        #print(b2)

        U[i, :] = spsolve(A2, b2)


Error = np.abs(Uexact - U)

# # Create a figure window
# fig = plt.figure(figsize=(18, 6))
#
# # Plot the numerical solution
# ax1 = fig.add_subplot(1, 3, 1, projection='3d')
# surf = ax1.plot_trisurf(p[:, 1], p[:, 0], U.T.flatten(), cmap=cm.jet, linewidth=0)
# ax1.view_init(elev=90, azim=-90)
# ax1.axis('equal')
# plt.colorbar(surf, ax=ax1, shrink=0.7)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Numerical Solution')
#
# # Plot the exact solution
# ax2 = fig.add_subplot(1, 3, 2, projection='3d')
# surf = ax2.plot_trisurf(p[:, 1], p[:, 0], Uexact.T.flatten(), cmap=cm.jet, linewidth=0)
# ax2.view_init(elev=90, azim=-90)
# ax2.axis('equal')
# plt.colorbar(surf, ax=ax2, shrink=0.7)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Exact Solution')
#
# # Plot the error
# ax3 = fig.add_subplot(1, 3, 3, projection='3d')
# surf = ax3.plot_trisurf(p[:, 1], p[:, 0], Error.T.flatten(), cmap=cm.jet, linewidth=0)
# ax3.view_init(elev=90, azim=-90)
# ax3.axis('equal')
# plt.colorbar(surf, ax=ax3, shrink=0.7)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Error')

plt.show()

# Contour plot
# fig2, ax = plt.subplots()
# contour = ax.contourf(x, y, U.T, cmap=cm.jet)
# plt.colorbar(contour)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

plt.pcolormesh(x,y,U,cmap='rainbow')
plt.colorbar()

plt.title(r'pcolormesh of solution')
plt.xlabel(r'x')
plt.ylabel(r'y')
plt.show()

# error calculation
UmunusUxactL2 = np.sqrt(dx * np.sum(Error**2))
UL2 = np.sqrt(dx * np.sum(Uexact**2))

Error = UmunusUxactL2/UL2

print("Error = %20.15f" %Error)


