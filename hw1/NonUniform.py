
def FD2(func,x,h1,h2,h3):
    return (2*(2*h2 + h3))*func(x-h1)/(h1*(h1 + h2)*(h1 + h2 + h3))\
           -(2*(2*h2 - h1 + h3))*func(x)/(h1*h2*(h2 + h3))\
           +(2*(h2 - h1 + h3))*func(x+h2)/(h2*h3*(h1 + h2))\
           +(2*(h1 - h2))*func(x+h2+h3)/(h3*(h2 + h3)*(h1 + h2 + h3))

if __name__ == "__main__":

    import numpy as np
    import matplotlib
    import matplotlib.pylab as plt
    import random as rand

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams.update({'font.size':14,'font.family':'serif'})
    matplotlib.rcParams.update({'text.usetex':'true'})

    # create my function
    def myfunc(x):
        return np.exp(x)

    # numerical experiment
    num_tests = 1001
    evec = np.zeros(num_tests)

    H = 10**np.linspace(-5,-1,num_tests)
    x = 1
    for n in range(0,num_tests):
        h1 = H[n]*rand.random()
        h2 = H[n]*rand.random()
        h3 = H[n]*rand.random()
        # exact second derivative
        df2_exact = np.exp(x)
        evec[n] = np.abs(df2_exact - FD2(myfunc,x,h1,h2,h3))

    # plot
    plt.figure(1)
    plt.clf()
    plt.grid()
    plt.loglog(H,evec,'b-')
    plt.title(r'Loglog error vs H for FD2 of $e^x$ at $x=1$')
    plt.xlabel('H')
    plt.ylabel('absolute error')
    #plt.show()

    plt.savefig('roundoff_NonUnifrom.pdf',format='pdf',bbox_inches='tight')

    # Least-square fitting
    A = np.array([[]])
    b = np.array([[]])
    for i in range(num_tests):
        np_array1 = np.array([[1,np.log(H[i])]])
        np_array2 = np.array([[np.log(evec[i])]])
        if (i == 0):
            A = np_array1
            b = np_array2
        else:
            A=np.append(A,np_array1,axis=0)
            b=np.append(b,np_array2,axis=0)

    StiffnesssMatrix =  np.dot(np.transpose(A),A)
    ForceVector = np.dot(np.transpose(A),b)

    Solution = np.linalg.solve(StiffnesssMatrix,ForceVector)

    # get C and p
    C = np.exp(Solution[0])
    p = Solution[1]

    print("C = %20.15f, p = %20.15f" % (C,p))
