import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from matplotlib import rc
from evaluate_population import evaluate_population
from scipy.spatial import distance
from non_domx import ndx

rc('font',**{'family':'serif','serif':['Helvetica']})
rc('text', usetex=True)



def plt_surface3d(surrogate_problem, filename):

    # Make data.
    x1 = np.arange(-1, 1, 0.01)
    x2 = np.arange(-1, 1, 0.01)

    #R = np.sqrt(x1**2 + x2**2)
    #y = np.sin(R)
    X = None
    for i in x1:
        for j in x2:
            if X is None:
                X = [i, j]
            else:
                X = np.vstack((X,[i,j]))
    x1, x2 = np.meshgrid(x1, x2)
    y = surrogate_problem.evaluate(X, use_surrogate=True)[0]
    y1 = y[:,0]
    y2 = y[:,1]
    #y3 = y[:,2]

    y1 = y1.reshape(np.shape(x1))
    # Plot the surface.
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, y1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(filename + '_3D_1.pdf', bbox_inches='tight')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()



    y2 = y2.reshape(np.shape(x1))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, y2, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(filename + '_3D_2.pdf', bbox_inches='tight')

    """
    y3 = y3.reshape(np.shape(x1))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, y3, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(filename + '_3.pdf', bbox_inches='tight')
    """
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()

def plt_surface2d(surrogate_problem, filename):

    # Make data.
    x1 = np.arange(-1, 1, 0.01)
    x2 = np.arange(-1, 1, 0.01)

    #R = np.sqrt(x1**2 + x2**2)
    #y = np.sin(R)
    X = None
    for i in x1:
        for j in x2:
            if X is None:
                X = [i, j]
            else:
                X = np.vstack((X,[i,j]))
    x1, x2 = np.meshgrid(x1, x2)
    y = surrogate_problem.evaluate(X, use_surrogate=True)[0]
    y1 = y[:,0]
    y2 = y[:,1]
    y3 = y[:,2]

    y1 = y1.reshape(np.shape(x1))
    y2 = y2.reshape(np.shape(x1))
    y3 = y3.reshape(np.shape(x1))
    # Plot the surface.
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.contour(x1,x2,y1,colors='red')
    plt.contour(x1,x2,y2,colors='blue')
    plt.contour(x1,x2,y3,colors='green')  
    plt.savefig(filename + '_contour.pdf', bbox_inches='tight')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()

def plt_surface_all(surrogate_problem,
                    filename,
                    init_folder,
                    problem_testbench, 
                    problem_name, 
                    nobjs, 
                    nvars, 
                    sampling, 
                    nsamples):
    # Make data.
    x1 = np.arange(-1, 1, 0.02)
    x2 = np.arange(-1, 1, 0.02)
    X = None
    for i in x1:
        for j in x2:
            if X is None:
                X = [i, j]
            else:
                X = np.vstack((X,[i,j]))
    x1, x2 = np.meshgrid(x1, x2)
    y = surrogate_problem.evaluate(X, use_surrogate=True)[0]
    y1 = y[:,0]
    y2 = y[:,1]
    y1 = y1.reshape(np.shape(x1))
    y2 = y2.reshape(np.shape(x1))

    y_underlying = evaluate_population(X,
                                        init_folder,
                                        problem_testbench, 
                                        problem_name, 
                                        nobjs, 
                                        nvars, 
                                        sampling, 
                                        nsamples)
    y_underlying1 = y_underlying[:,0].reshape(np.shape(x1))
    y_underlying2 = y_underlying[:,1].reshape(np.shape(x1))
    non_dom_front = ndx(y_underlying)
    y_underlying_nds = y_underlying[non_dom_front[0][0]]
    x_underlying_nds = X[non_dom_front[0][0]]
    rmse_mv_sols = np.zeros(np.shape(y)[0])
    for i in range(np.shape(y)[0]):
        rmse_mv_sols[i]= distance.euclidean(y[i,:],y_underlying[i,:])
    rmse_mv_sols = rmse_mv_sols.reshape(np.shape(x1))

    plt.contour(x1,x2,y1,30,linewidths=0.5,colors='red')
    plt.contour(x1,x2,y2,30,linewidths=0.5,colors='blue')
    plt.contourf(x1, x2, rmse_mv_sols, 30, cmap='Greens')
    plt.colorbar()
    plt.scatter(x_underlying_nds[:,0], x_underlying_nds[:,1])
    plt.savefig(filename + '_contour.pdf', bbox_inches='tight')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()