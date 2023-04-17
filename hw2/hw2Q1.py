

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
    alpha = 21/20
    beta = 1 + np.exp(np.pi)/20
    e=np.ones(N)

    A=(spdiags(-e,-1,N,N)+spdiags((2+h**2*np.pi**2)*e,0,N,N)+spdiags(-e,1,N,N))/h**2
    #A.toarray()  # if you want to print the matrix to screen

    F=np.ones(N)
    Uexact = np.ones(N)
    X = np.ones(N+2) # with boundary points

    X[0] = 0
    X[N+1] = L
    for i in range(N):
        x = (i+1)/(N+1)
        X[i+1] = x
        F[i] = 5*np.pi**2*(np.cos(2*np.pi*x)+np.sin(2*np.pi*x))
        Uexact[i] =np.cos(2*np.pi*x)+np.sin(2*np.pi*x)+ np.exp(np.pi*x)/20

    F[0] = F[0] + alpha/h**2
    F[N-1] = F[N-1] + beta/h**2

    U=spsolve(A,F)

    UmunusUxact=np.ones(N)
    UmunusUxact = np.abs(U-Uexact)

    UmunusUxactL2 = 0
    UL2 = 0
    for i in range(N):
        UmunusUxactL2 = UmunusUxactL2 + UmunusUxact[i]**2
        UL2 = UL2 + U[i]**2
    UmunusUxactL2 = np.sqrt(h*UmunusUxactL2)
    UL2 = np.sqrt(h*UL2)

    Error = UmunusUxactL2/UL2

    print("Error = %20.15f" %Error)

    ##
    UwithBoundaryValue = np.ones(N+2)
    UexactwithBoundaryValue = np.ones(N+2)

    UwithBoundaryValue[0] = alpha
    UexactwithBoundaryValue[0] = alpha
    UwithBoundaryValue[N+1] = beta
    UexactwithBoundaryValue[N+1] = beta

    for i in range(N):
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
    plt.legend(['numerical solution','exact solution'])
    #plt.show()

    plt.savefig('hw2Q1.pdf',format='pdf',bbox_inches='tight')

