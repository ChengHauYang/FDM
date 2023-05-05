import numpy as np
import matplotlib.pyplot as plt

def a(x):
    return 1/5 + np.sin(x-1)**2

def da(x):
    return 2*np.sin(x-1)*np.cos(x-1)

def d2a(x):
    return -2*np.sin(x-1)**2 + 2*np.cos(x-1)**2

def Analysol(x):
    return np.exp(-4*(x-np.pi)**2)

Tfinal = 10*np.pi/np.sqrt(6)

aMax = 1.2

DomainSize = 2*np.pi
meshSizes = np.array([2**i-1 for i in range(5, 13)])

errors = np.zeros(meshSizes.shape)

for idxMesh in range(len(meshSizes)):
    Nx = meshSizes[idxMesh]
    mx = Nx + 2

    h = 2*np.pi/(Nx + 1)

    k = Tfinal/round(Tfinal/(0.9*h/aMax))

    time = np.arange(0, Tfinal+k, k)
    numTimeSteps = time.size

    x = np.linspace(0, 2*np.pi, mx)
    U = np.zeros((mx, numTimeSteps))

    a_values = a(x)
    da_values = da(x)
    a_da_values = a_values * da_values
    app_values = d2a(x)

    U[:, 0] = Analysol(x)

    for t in range(numTimeSteps-1):
        for i in range(mx):
            CFL = a_values[i]*k/h

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

            # U[i, t+1] = U[i, t] - CFL*(U[ip, t]-U[im, t])/2 \
            #         + (k**2/2)*a_values[i]*da_values[i]*(U[ip, t]-U[im, t])/(2*h) \
            #         + CFL**2*(U[ip, t]-2*U[i, t]+U[im, t])/2 \
            #         - (k**3/6)*a_values[i]*da_values[i]**2*(U[ip, t]-U[im, t])/(2*h) \
            #         - (k**3/6)*a_values[i]*app_values[i]*(U[ip, t]-U[im, t])/(2*h) \
            #         - (k**3/2)*a_values[i]**2*da_values[i]*(U[ip, t]-2*U[i, t]+U[im, t])/(2*h) \
            #         - CFL**3*(-U[imm, t]+3*U[im, t]-3*U[i, t]+U[ip, t])/6
            U[i, t+1] = U[i, t] - CFL*(U[ip, t]-U[im, t])/2 \
                    + (k**2/2)*a_values[i]*da_values[i]*(U[ip, t]-U[im, t])/(2*h) \
                    + CFL**2*(U[ip, t]-2*U[i, t]+U[im, t])/2

    error = abs(np.exp(-4*(x-np.pi)**2)-U[:, -1])
    L2Error = np.sqrt(np.sum(error**2)*h)
    L2U = np.sqrt(np.sum(np.exp(-4*(x-np.pi)**2)**2)*h)
    errors[idxMesh] = L2Error/L2U
    print(errors[idxMesh])

    if idxMesh != 0:
        print(errors[idxMesh-1]/errors[idxMesh])

    scale = 5 * errors[-1] / (DomainSize / meshSizes[-1]) ** 2

plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')

plt.figure(figsize=(8, 6), dpi=80, facecolor='w')
plt.loglog(DomainSize/meshSizes, errors, '-o', linewidth=1.5, markersize=6)
plt.loglog(DomainSize/meshSizes, scale*(DomainSize/meshSizes)**2, '--', linewidth=1.5)
plt.xlabel(r'$\textrm{Mesh size (h)}$', fontsize=14)
plt.ylabel(r'$\textrm{L2 error}$', fontsize=14)
plt.title(r'$\textrm{Mesh Convergence Study}$', fontsize=16)
plt.legend(['Numerical error', 'Slope = 2'], fontsize=12, loc='upper left')
plt.grid()
plt.tick_params(axis='both', which='both', direction='in', labelsize=12, width=1.5)
#plt.show()
plt.savefig('MeshConvergence.pdf',format='pdf',bbox_inches='tight')

