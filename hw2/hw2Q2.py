

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib
import matplotlib.pylab as plt
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams.update({'font.size':14,'font.family':'serif'})
matplotlib.rcParams.update({'text.usetex':'true'})

if __name__ == "__main__":
    L = 1
    N = 800
    h = L/(N+1)
    e=np.ones(N+1)

    A=-(spdiags(16*e,-N,N+1,N+1)+ spdiags(-e,1-N,N+1,N+1)+ spdiags(-e,-2,N+1,N+1)+spdiags(16*e,-1,N+1,N+1)+spdiags((-30-12*h**2)*e,0,N+1,N+1)+spdiags(16*e,1,N+1,N+1)+spdiags(-e,2,N+1,N+1)+spdiags(-e,N-1,N+1,N+1)+spdiags(16*e,N,N+1,N+1))/(12*h**2);
    #A.toarray()  # if you want to print the matrix to screen

    F=np.ones(N+1)
    Uexact = np.ones(N+1)
    X = np.ones(N+2) # with boundary points

    X[0] = 0
    for i in range(N+1):
        x = (i+1)/(N+1)
        X[i+1] = x
        F[i] = np.sin(4*np.pi*x)
        Uexact[i] =np.sin(4*np.pi*x)/(16*np.pi**2+1)

    U=spsolve(A,F)

    UmunusUxact=np.ones(N+1)
    UmunusUxact = np.abs(U-Uexact)

    UmunusUxactL2 = 0
    UL2 = 0
    for i in range(N+1):
        UmunusUxactL2 = UmunusUxactL2 + UmunusUxact[i]**2
        UL2 = UL2 + U[i]**2
    UmunusUxactL2 = np.sqrt(h*UmunusUxactL2)
    UL2 = np.sqrt(h*UL2)

    Error = UmunusUxactL2/UL2

    print("Error = %20.15f" %Error)

    ##
    UwithBoundaryValue = np.ones(N+2)
    UexactwithBoundaryValue = np.ones(N+2)


    ## periodic boundary condition
    UwithBoundaryValue[0] = U[N]
    UexactwithBoundaryValue[0] = Uexact[N]

    for i in range(N+1):
        UwithBoundaryValue[i+1] = U[i]
        UexactwithBoundaryValue[i+1] = Uexact[i]

    # plot
    plt.figure(1)
    plt.clf()
    plt.grid()
    plt.plot(X,UwithBoundaryValue,'b-')
    plt.plot(X,UexactwithBoundaryValue,'r--')
    plt.title(r'Solution vs x with N = 50')
    plt.xlabel(r'x')
    plt.ylabel('Solution')

    plt.legend(['numerical solution','exact solution'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    #plt.show()


    plt.savefig('hw2Q2.pdf',format='pdf',bbox_inches='tight')

