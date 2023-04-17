import numpy as np
import matplotlib.pyplot as plt

def Analysol(x):
    return 2 * np.exp(-200.*(x-0.5)**2) * np.cos(40*np.pi*x)

def update_indices(i, mx):
    ipp = i + 2
    ip = i + 1
    im = i - 1
    imm = i - 2

    if i == 1:
        imm = mx - 2
    elif i == 0:
        im = mx - 2
        imm = mx - 3
    elif i == mx - 1:
        ip = 1
        ipp = 2
    elif i == mx - 2:
        ipp = 1

    return ipp, ip, im, imm

Tfinal = 2
a = 1

# define range of mesh sizes to test
#meshSizes = np.array([49,99,199,399,799,899,999,1199,1399,1599])
#meshSizes = np.array([49,99,199])
meshSizes = np.array([49,99,199,399,799,1599,3199])

# initialize error vector
errors = np.zeros(meshSizes.shape)

for idxMesh in range(len(meshSizes)):
    Nx = meshSizes[idxMesh]
    mx = Nx+2

    h = 1 / (Nx + 1)
    k = h / 2
    time = np.arange(0, Tfinal+k, k)
    numTimeSteps = len(time)

    # include boundary points below
    x = np.linspace(0, 1, mx)
    U = np.zeros((mx, numTimeSteps))

    # for RK3
    Y1 = np.zeros(mx)
    U_star = np.zeros((mx,numTimeSteps))
    Y2 = np.zeros(mx)
    U_star_star = np.zeros((mx,numTimeSteps))
    Y3 = np.zeros(mx)


    AnalyticalSolution = np.zeros((mx, numTimeSteps))

    x_move = np.zeros_like(x)

    # To ensure stability and accuracy, the time step should be chosen such that the CFL number is less than 1.
    CFL = a*k/h

    # Initialize the solution at the initial time step
    U[:, 0] = Analysol(x)

    OutputFreq = int(numTimeSteps/5)
    num = 0
    CurrentTime = 0
    for t in range(numTimeSteps-1):
        CurrentTime = CurrentTime+k

        # step 1
        for i in range(mx):
            ipp, ip, im, imm = update_indices(i, mx)
            Y1[i]  = -a*(U[imm,t]-6*U[im,t]+3*U[i,t]+2*U[ip,t])/(6*h)

        # step 2
        for i in range(mx):
            ipp, ip, im, imm = update_indices(i, mx)
            U_star[i,t]= U[i,t] + k*Y1[i]/2

        # step 3
        for i in range(mx):
            ipp, ip, im, imm = update_indices(i, mx)
            Y2[i] = -a*(U_star[imm,t]-6*U_star[im,t]+3*U_star[i,t]+2*U_star[ip,t])/(6*h)

        # step 4
        for i in range(mx):
            ipp, ip, im, imm = update_indices(i, mx)
            U_star_star[i, t] = U[i, t] + 3 * k * Y2[i] / 4

        # step 5
        for i in range(mx):
            ipp, ip, im, imm = update_indices(i, mx)
            Y3[i] = -a*(U_star_star[imm,t]-6*U_star_star[im,t]+3*U_star_star[i,t]+2*U_star_star[ip,t])/(6*h)


        U[:,t+1] = U[:,t] + k*(2*Y1[:]+3*Y2[:]+4*Y3[:])/9

        for i in range(mx):
            x_move[i] = x[i] - a * time[t+1]
            while (x_move[i]>1):
                x_move[i]=x_move[i]-1
            while (x_move[i] < 0):
                x_move[i]=x_move[i]+1
            if t == numTimeSteps -2:
                AnalyticalSolution[i, t+1] = Analysol(x_move[i])

    # Compute the L2 error
    CurrentTime = time[-1]
    error = np.abs(AnalyticalSolution[:, -1] - U[:, -1])
    L2Error = np.sqrt(np.sum(error**2)*h)
    L2U = np.sqrt(np.sum(AnalyticalSolution[:, -1]**2)*h)
    errors[idxMesh] = L2Error/L2U

# plot convergence plot with slope=3 line
scale = 5*errors[-1]/(1./meshSizes[-1])**3
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')
plt.figure(figsize=(8, 6), facecolor='w')
plt.loglog(1./meshSizes, errors, '-o', linewidth=1.5, markersize=6)
plt.loglog(1./meshSizes, scale*(1./meshSizes)**3, '--', linewidth=1.5)
plt.xlabel(r'$\mathrm{Mesh\ size\ (h)}$', fontsize=14)
plt.ylabel(r'$\mathrm{L2\ error}$', fontsize=14)
plt.title(r'$\mathrm{Mesh\ Convergence\ Study}$', fontsize=16)
plt.legend(['Numerical error', 'Slope = 3'], fontsize=12, loc='upper left')
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12, width=1.5)
#plt.show()
plt.savefig('MeshConvergence_RK3.pdf',format='pdf',bbox_inches='tight')


