import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye
from scipy.sparse import kron
import matplotlib
import matplotlib.pylab as plt
from matplotlib import rc
import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import math

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams.update({'font.size':14,'font.family':'serif'})
matplotlib.rcParams.update({'text.usetex':'true'})

if __name__ == "__main__":

    case = 2
    L = 1
    N = 39
    h = L/(N-1)
    InnerRowElementNumber = N -2
    InnerRowElementNumber = int(L/h -1)
    MatrixSize = InnerRowElementNumber**2

    # diag=np.ones([InnerRowElementNumber*InnerRowElementNumber])
    # mat= spdiags([-diag,2*diag,-diag],[-1,0,1],InnerRowElementNumber,InnerRowElementNumber)
    # I= eye(InnerRowElementNumber)
    # A = kron(I,mat)+kron(mat,I)

    m = InnerRowElementNumber+2
    AllNode = m**2
    diag=np.ones([m*m])
    mat= spdiags([-diag,2*diag,-diag],[-1,0,1],m,m)
    I= eye(m)
    Aall = kron(I,mat)+kron(mat,I)


    # places we need to set BC
    top=np.arange(m*m-m,m*m)
    bottom=np.arange(0,m)
    left=np.arange(m,m*(m-1)-m+1,m)
    right=np.arange(m*2-1,m*m-m,m)

    inner=np.arange(m*(m-1)/2,m*(m-1)/2+(m-1)/2+1)
    array_of_arrays = np.array([np.arange(m*(m-1)/2+i*m,m*(m-1)/2+(m-1)/2+1+i*m) for i in range(int(0.5*(m-1)))])


    # calculating forcing vector
    Fall = np.zeros(AllNode)
    BC = np.zeros(AllNode)
    UwithBC = np.zeros(AllNode)

    counter = 0
    for iy in range(N):
        for ix in range(N):
            x=ix/(InnerRowElementNumber+1)
            y=iy/(InnerRowElementNumber+1)

            if (case == 1):
                # case 1
                Fall[counter] = 1*h**2
            elif (case == 2):
                # case 2
                Fall[counter] = 2*np.exp(-(10*x-5)**2-(10*y-5)**2)*h**2
                #print((-(-10*x-5)**2-(10*y-5)**2))
                #print(2*np.exp(-(-10*x-5)**2-(10*y-5)**2)*h**2)

            #print(Fall[counter])
            if (counter in top or counter in bottom or counter in left or counter in right):
                Fall[counter] = 0 # Boundary condition
                Aall[counter,:] = 0
                Aall[counter,counter] = 1

            counter=counter+1

    for i in array_of_arrays:
        for j in i:
            Fall[int(j)] = 0 # Boundary condition
            Aall[int(j),:] = 0
            Aall[int(j),int(j)] = 1

    # showing matrix
    plt.spy(Aall, markersize=20)
    name = str(N)
    plt.title(f'spy plot of the matrix with N = {name}')
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")

    #plt.show()
    plt.savefig(f'hw6_spy_{N}_case{case}.pdf',format='pdf',bbox_inches='tight')
    plt.clf() # Clear the plot for the next iteration


    # solving
    UwithBC=spsolve(Aall,Fall)

    # inner nodes
    I = np.setdiff1d(np.arange(0,AllNode),bottom)
    I = np.setdiff1d(I,top)
    I = np.setdiff1d(I,left)
    I = np.setdiff1d(I,right)
    for i in array_of_arrays:
        I = np.setdiff1d(I,i)

    # Obtaining U
    U = np.zeros(MatrixSize)
    counter =0
    for InnerNodesIndex in I:
        U[counter] = UwithBC[InnerNodesIndex]
        counter=counter+1

    # visualize
    Xmesh = np.zeros((m, m))
    Ymesh = np.zeros((m, m))
    Umesh = np.zeros((m, m))
    counter = 0
    for iy in range(m):
        for ix in range(m):
            x=ix/(InnerRowElementNumber+1)
            y=iy/(InnerRowElementNumber+1)
            Xmesh[ix][iy] = x
            Ymesh[ix][iy] = y
            Umesh[ix][iy] = UwithBC[counter]
            counter=counter+1

    csfont = {'fontname':'Times New Roman'}
    plt.pcolormesh(Xmesh,Ymesh,Umesh,cmap='rainbow')
    plt.colorbar()

    plt.title(f'pcolormesh of solution (N={N})',**csfont)
    plt.xlabel('x',**csfont)
    plt.ylabel('y',**csfont)
    #plt.show()
    plt.savefig(f'hw6_pcolor_{N}_case{case}.pdf',format='pdf',bbox_inches='tight')

    # calculate Error of inner points (not including Boundary points)
    # Uexact = np.zeros(MatrixSize)
    # counter = 0
    # for iy in range(InnerRowElementNumber):
    #     for ix in range(InnerRowElementNumber):
    #         x=(ix+1)/(InnerRowElementNumber+1)
    #         y=(iy+1)/(InnerRowElementNumber+1)
    #         Uexact[counter] = np.exp(x+0.5*y)
    #         counter=counter+1
    #
    # UmunusUxact=np.zeros(MatrixSize)
    # UmunusUxact = np.abs(U-Uexact)
    #
    # UmunusUxactL2 = 0
    # UL2 = 0
    # for i in range(MatrixSize):
    #     UmunusUxactL2 = UmunusUxactL2 + UmunusUxact[i]**2
    #     UL2 = UL2 + Uexact[i]**2
    # UmunusUxactL2 = np.sqrt(h*UmunusUxactL2)
    # UL2 = np.sqrt(h*UL2)
    #
    # Error = UmunusUxactL2/UL2
    #
    # print("Error = %20.15f" %Error)

