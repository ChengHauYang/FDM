import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye
from scipy.sparse import kron
import matplotlib
import matplotlib.pylab as plt
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams.update({'font.size':14,'font.family':'serif'})
matplotlib.rcParams.update({'text.usetex':'true'})

if __name__ == "__main__":

    L = 1
    h = 0.003125
    N = int(L/h -1)
    MatrixSize = N**2


    # change place 1
    diag=np.ones([N*N])
    diag2= spdiags([-4*diag,20*diag,-4*diag],[-1,0,1],N,N)
    T1 = spdiags([-1*diag,-4*diag,-1*diag],[-1,0,1],N,N)

    I= eye(N)
    I1= eye(N,k=-1)
    I2= eye(N,k=1)
    A = kron(I,diag2)+kron(I1,T1)+kron(I2,T1)

    AllNode = (N+2)**2
    m = N+2
    diag=np.ones([m*m])
    diag2= spdiags([-4*diag,20*diag,-4*diag],[-1,0,1],m,m)
    T1 = spdiags([-1*diag,-4*diag,-1*diag],[-1,0,1],m,m)
    I= eye(m)
    I1= eye(m,k=-1)
    I2= eye(m,k=1)
    Aall = kron(I,diag2)+kron(I1,T1)+kron(I2,T1)



    # places we need to set BC
    top=np.arange(m*m-m,m*m)
    bottom=np.arange(0,m)
    left=np.arange(m,m*(m-1)-m+1,m)
    right=np.arange(m*2-1,m*m-m,m)

    # calculating forcing vector
    Fall = np.zeros(AllNode)
    BC = np.zeros(AllNode)
    UwithBC = np.zeros(AllNode)

    counter = 0
    for iy in range(N+2):
        for ix in range(N+2):
            x=ix/(N+1)
            y=iy/(N+1)
            if (counter in top or counter in bottom or counter in left or counter in right):
                BC[counter] = np.exp(x+0.5*y)
            counter=counter+1

    # enforcing strong BC on the RHS
    Fall = Fall - Aall.dot(BC)

    # inner nodes
    I = np.setdiff1d(np.arange(0,AllNode),bottom)
    I = np.setdiff1d(I,top)
    I = np.setdiff1d(I,left)
    I = np.setdiff1d(I,right)

    # Obtaining F
    F = np.zeros(MatrixSize)

    counter = 0
    for iy in range(N):
        for ix in range(N):
            x=(ix+1)/(N+1)
            y=(iy+1)/(N+1)
            dh= 1/(N+1)
            # change place 2 -> multiply it by 6 + h**2*laplace(f)/12
            laplace9_fij = (-1.25*np.exp((x+dh)+0.5*(y+dh))-1.25*np.exp((x-dh)+0.5*(y+dh))-1.25*np.exp((x+dh)+0.5*(y-dh))-1.25*np.exp((x-dh)+0.5*(y-dh))
                            -4*1.25*np.exp((x+dh)+0.5*(y))-4*1.25*np.exp((x-dh)+0.5*(y))
                            -4*1.25*np.exp((x)+0.5*(y+dh))-4*1.25*np.exp((x)+0.5*(y-dh))
                            +20*1.25*np.exp((x)+0.5*(y))
                            )/(6*h**2)
            F[counter] = Fall[I[counter]] + (-1.25*np.exp(x+0.5*y)+h**2*laplace9_fij/12)*6*h**2
            counter = counter+1

    # solving
    U=spsolve(A,F)

    # UwithBC
    UwithBC = BC.copy()

    counter = 0
    for InnerNodesIndex in I:
        UwithBC[InnerNodesIndex] = U[counter]
        counter=counter+1

    # visualize
    Xmesh = np.zeros((m, m))
    Ymesh = np.zeros((m, m))
    Umesh = np.zeros((m, m))
    counter = 0
    for iy in range(m):
        for ix in range(m):
            x=ix/(N+1)
            y=iy/(N+1)
            Xmesh[ix][iy] = x
            Ymesh[ix][iy] = y
            Umesh[ix][iy] = UwithBC[counter]
            counter=counter+1

    plt.pcolormesh(Xmesh,Ymesh,Umesh,cmap='rainbow')
    plt.colorbar()

    plt.title(r'pcolormesh of solution')
    plt.xlabel(r'x')
    plt.ylabel('y')
    #plt.show()

    plt.savefig('hw5.pdf',format='pdf',bbox_inches='tight')

    # calculate Error of inner points (not including Boundary points)
    Uexact = np.zeros(MatrixSize)
    counter = 0
    for iy in range(N):
        for ix in range(N):
            x=(ix+1)/(N+1)
            y=(iy+1)/(N+1)
            Uexact[counter] = np.exp(x+0.5*y)
            counter=counter+1

    UmunusUxact=np.zeros(MatrixSize)
    UmunusUxact = np.abs(U-Uexact)

    UmunusUxactL2 = 0
    UL2 = 0
    for i in range(MatrixSize):
        UmunusUxactL2 = UmunusUxactL2 + UmunusUxact[i]**2
        UL2 = UL2 + Uexact[i]**2
    UmunusUxactL2 = np.sqrt(h*UmunusUxactL2)
    UL2 = np.sqrt(h*UL2)

    Error = UmunusUxactL2/UL2

    print("Error = %20.15f" %Error)





