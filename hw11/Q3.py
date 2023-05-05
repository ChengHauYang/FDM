import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')


def CreateDmat(N):
    Dmat = np.zeros((N, N))
    h = 2 * np.pi / N  # grid spacing (from class note)
    for j in range(N):
        for m in range(N):
            if (np.mod(j-m, N) != 0):
                Dmat[j][m] = (-1) ** (j-m) * (1 / np.tan((j-m) * h / 2)) / 2
    return Dmat


def CreateD2mat(N):
    D2mat = np.zeros((N, N))
    h = 2 * np.pi / N  # grid spacing (from class note)
    for j in range(N):
        for m in range(N):
            if (np.mod(j-m, N) != 0):
                D2mat[j][m] = -(-1) ** (j-m) * \
                    (1 / np.sin((j-m) * h / 2)) ** 2 / 2
            else:
                D2mat[j][m] = -np.pi ** 2 / (3 * h ** 2) - 1 / 6
    return D2mat


def PrintMatrix(mat):
    N = mat.shape[0]
    for j in range(N):  # row j
        for m in range(N):  # col m
            print("%5.4e" % mat[j][m], end=", ")
        print("\n")


def MatrixMultiply(mat1, mat2):
    N = mat1.shape[0]
    mat1Xmat2 = np.zeros((N, N))
    for j in range(N):  # row j of mat1
        for m in range(N):  # col m of mat2
            for k in range(N):
                mat1Xmat2[j][m] += mat1[j][k]*mat2[k][m]

    return mat1Xmat2


if __name__ == "__main__":

    # 3 b
    N = 4
    Dmat = CreateDmat(N)
    D2mat = CreateD2mat(N)

    DmatSquare = MatrixMultiply(Dmat, Dmat)
    print("D_N^2")
    PrintMatrix(DmatSquare)
    print("D2mat")
    PrintMatrix(D2mat)

    # 3 c
    N = 24
    h = 2 * np.pi / N
    x = h * np.arange(1, N + 1)

    Dmat = CreateDmat(N)
    D2mat = CreateD2mat(N)

    v = np.exp(np.sin(x))
    v_FirstDerivative = np.exp(np.sin(x)) * np.cos(x)
    v_SecondDerivative = np.exp(np.sin(x)) * \
        np.cos(x) ** 2 - np.exp(np.sin(x)) * np.sin(x)

    NumericalFirstDerivative = np.dot(Dmat, v)
    NumericalSecondDerivative = np.dot(D2mat, v)

    plt.figure(figsize=(8, 6), dpi=120)
    plt.plot(x, v)
    plt.title(r"$f(x)$ vs $x$", fontsize=18,
              fontname="Times", fontweight="bold")
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$f(x)$", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.grid(True)
    # plt.show()
    plt.savefig('Q3a.pdf', format='pdf', bbox_inches='tight')

    plt.figure(figsize=(8, 6), dpi=120)
    plt.plot(x, v_FirstDerivative, label="Exact first derivative", linewidth=3)
    plt.plot(x, NumericalFirstDerivative, "--",
             label="Numerical first derivative", linewidth=3)
    plt.legend(fontsize=12)
    plt.title(r"$f'(x)$ vs $x$", fontsize=18,
              fontname="Times", fontweight="bold")
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$f'(x)$", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.grid(True)
    # plt.show()
    plt.savefig('Q3b.pdf', format='pdf', bbox_inches='tight')

    plt.figure(figsize=(8, 6), dpi=120)
    plt.plot(x, v_SecondDerivative,
             label="Exact second derivative", linewidth=3)
    plt.plot(x, NumericalSecondDerivative, "--",
             label="Numerical second derivative", linewidth=3)
    plt.legend(fontsize=12)
    plt.title(r"$f''(x)$ vs $x$", fontsize=18,
              fontname="Times", fontweight="bold")
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$f''(x)$", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.grid(True)
    # plt.show()
    plt.savefig('Q3c.pdf', format='pdf', bbox_inches='tight')
