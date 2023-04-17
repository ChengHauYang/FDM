import numpy as np
import matplotlib.pyplot as plt

def Analysol(x):
    return 2 * np.exp(-200.*(x-0.5)**2) * np.cos(40*np.pi*x)

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
        for i in range(mx):
            ipp = i+2
            ip=i+1
            im=i-1
            imm=i-2
            if i==1:
                imm=mx-2
            if i ==0:
                im=mx-2
                imm=mx-3
            if i==mx-1:
                ip=1
                ipp=2
            if i==mx-2:
                ipp=1
            U[i,t+1] = U[i,t]-CFL*(U[imm,t]-6*U[im,t]+3*U[i,t]+2*U[ip,t])/6 + CFL**2*(U[ip,t]-2*U[i,t]+U[im,t])/2 - CFL**3*(-U[imm,t]+3*U[im,t]-3*U[i,t]+U[ip,t])/6

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
plt.savefig('MeshConvergence.pdf',format='pdf',bbox_inches='tight')
