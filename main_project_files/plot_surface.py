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



def plt_surface3d(surrogate_problem, 
                    filename,
                    init_folder,
                    problem_testbench, 
                    problem_name, 
                    nobjs, 
                    nvars, 
                    sampling, 
                    nsamples):

    # Make data.
    x1 = np.arange(-1, 1, 0.01)
    x2 = np.arange(-1, 1, 0.01)
    #v_max=2*np.sqrt(2)
    v_max = 2
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
    m = ax.plot_surface(x1, x2, y1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, vmin=-0.1, vmax=v_max)
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter('{x:.02f}')
    m = plt.cm.ScalarMappable(cmap=cm.coolwarm)
    m.set_array( y1)
    m.set_clim(0., v_max)
    fig.colorbar(m, boundaries=np.linspace(0, v_max, 11),shrink=0.5, aspect=5)
    fig.savefig(filename + '_3D_1.pdf', bbox_inches='tight')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()



    y2 = y2.reshape(np.shape(x1))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    m = ax.plot_surface(x1, x2, y2, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, vmin=-0.1, vmax=v_max)
    m = plt.cm.ScalarMappable(cmap=cm.coolwarm)
    m.set_array( y2)
    m.set_clim(0., v_max)
    fig.colorbar(m, boundaries=np.linspace(0, v_max, 11),shrink=0.5, aspect=5)
    fig.savefig(filename + '_3D_2.pdf', bbox_inches='tight') 

    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()
    
    y_underlying = evaluate_population(X,
                                        init_folder,
                                        approaches = "abc",
                                        problem_testbench = problem_testbench, 
                                        problem_name = problem_name, 
                                        nobjs = nobjs, 
                                        nvars = nvars, 
                                        sampling = sampling, 
                                        nsamples = nsamples)
    y_underlying1 = y_underlying[:,0].reshape(np.shape(x1))
    y_underlying2 = y_underlying[:,1].reshape(np.shape(x1))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    m = ax.plot_surface(x1, x2, y_underlying1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, vmin=-0.1, vmax=v_max)
    m = plt.cm.ScalarMappable(cmap=cm.coolwarm)
    m.set_array( y_underlying1)
    m.set_clim(0., v_max)
    fig.colorbar(m, boundaries=np.linspace(0, v_max, 11),shrink=0.5, aspect=5)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(filename + '_3D_undelying_1.pdf', bbox_inches='tight')
    
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    m = ax.plot_surface(x1, x2, y_underlying2, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, vmin=-0.1, vmax=v_max)
    m = plt.cm.ScalarMappable(cmap=cm.coolwarm)
    m.set_array( y_underlying2)
    m.set_clim(0., v_max)
    fig.colorbar(m, boundaries=np.linspace(0, v_max, 11),shrink=0.5, aspect=5)
    fig.savefig(filename + '_3D_undelying_2.pdf', bbox_inches='tight') 


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
                                        approaches = "abc",
                                        problem_testbench = problem_testbench, 
                                        problem_name = problem_name, 
                                        nobjs = nobjs, 
                                        nvars = nvars, 
                                        sampling = sampling, 
                                        nsamples = nsamples)
    y_underlying1 = y_underlying[:,0].reshape(np.shape(x1))
    y_underlying2 = y_underlying[:,1].reshape(np.shape(x1))
    non_dom_front = ndx(y_underlying)

    y_underlying_nds = y_underlying[non_dom_front[0][0]]
    x_underlying_nds = X[non_dom_front[0][0]]
    y_nds = y[non_dom_front[0][0]]

    rmse_mv_sols = np.zeros(np.shape(y)[0])
    for i in range(np.shape(y)[0]):
        rmse_mv_sols[i]= distance.euclidean(y[i,:],y_underlying[i,:])
    rmse_mv_sols = rmse_mv_sols.reshape(np.shape(x1))

    rmse_mv_nds = np.zeros(np.shape(y_nds)[0])
    for i in range(np.shape(y_nds)[0]):
        rmse_mv_nds[i]= distance.euclidean(y_nds[i,:],y_underlying_nds[i,:])

    v_max = 0.2
    lev = 25
    cmap_reversed = cm.get_cmap('inferno_r')
    #plt.contour(x1,x2,y1,30,linewidths=0.5,colors='red')
    #plt.contour(x1,x2,y2,30,linewidths=0.5,colors='blue')
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$x_1$', fontsize=17)
    ax.set_ylabel(r'$x_2$', fontsize=17)
    CS = ax.contour(x1,x2,y1,30,linewidths=0.7,colors='black')
    CS = ax.contour(x1,x2,y2,30,linewidths=0.7,colors='black',linestyles='dashed')
    #ax.set_xlabel('x_1')
    #ax.set_ylabel('x_2')    
    

    print("Max RMSE:",np.max(rmse_mv_sols))
    #plt.contourf(x1, x2, rmse_mv_sols, 30, cmap='Greens', vmin = 0, vmax = 0.5)
    CS = plt.contourf(x1, x2, rmse_mv_sols, levels= lev, cmap=cmap_reversed, vmin = 0, vmax = v_max)
    m = plt.cm.ScalarMappable(cmap=cmap_reversed)
    m.set_array(rmse_mv_sols)
    m.set_clim(0., v_max)
    plt.colorbar(m, boundaries=np.linspace(0, v_max, lev+1))
    plt.scatter(x_underlying_nds[:,1], x_underlying_nds[:,0], marker='*',color='blue')
    plt.savefig(filename + '_contour.pdf', bbox_inches='tight')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()

    print("Max RMSE:",np.max(rmse_mv_nds))
    # Plot the non dominated solutions
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$x_1$', fontsize=17)
    ax.set_ylabel(r'$x_2$', fontsize=17)
    CS = ax.contour(x1,x2,y1,30,linewidths=0.7,colors='black')
    CS = ax.contour(x1,x2,y2,30,linewidths=0.7,colors='black',linestyles='dashed')
    ax.set_xlim(right=-0.4, left=-1)
    ax.set_ylim(bottom=-1, top=-0.4)
    CS = plt.contourf(x1, x2, rmse_mv_sols, levels= lev, cmap=cmap_reversed, vmin = 0, vmax = v_max)
    m = plt.cm.ScalarMappable(cmap=cmap_reversed)
    m.set_array(rmse_mv_sols)
    m.set_clim(0., v_max)
    plt.colorbar(m, boundaries=np.linspace(0, v_max, lev+1))
    #sc = plt.scatter(x_underlying_nds[:,1], x_underlying_nds[:,0], marker='*',c=rmse_mv_nds, cmap='winter', vmin = 0, vmax = 0.1)
    #sc = plt.scatter(x_underlying_nds[:,0], x_underlying_nds[:,1], marker='*',s=rmse_mv_nds*500, color='blue')
    plt.scatter(x_underlying_nds[:,1], x_underlying_nds[:,0], marker='*',color='blue')
    #plt.colorbar(sc)
    plt.savefig(filename + '_contour_zoomed.pdf', bbox_inches='tight')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()



    #plt.contour(x1,x2,y_underlying1,30,linewidths=0.5,colors='red')
    #plt.contour(x1,x2,y_underlying2,30,linewidths=0.5,colors='blue')
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$x_1$', fontsize=17)
    ax.set_ylabel(r'$x_2$', fontsize=17)
    CS = ax.contour(x1,x2,y_underlying1,30,linewidths=0.7,colors='black')
    CS = ax.contour(x1,x2,y_underlying2,30,linewidths=0.7,colors='black',linestyles='dashed')
    #CS = plt.contourf(x1, x2, np.zeros(np.shape(y)[0]).reshape(np.shape(x1)),  levels= lev, cmap=cmap_reversed, vmin = 0, vmax = v_max)
    m = plt.cm.ScalarMappable(cmap=cmap_reversed)
    m.set_array(np.zeros(np.shape(y)[0]).reshape(np.shape(x1)))
    m.set_clim(0., v_max)
    plt.colorbar(m, boundaries=np.linspace(0, v_max, lev+1))
    plt.scatter(x_underlying_nds[:,1], x_underlying_nds[:,0], marker='*',color='blue')
    plt.savefig(filename + '_contour_underlying.pdf', bbox_inches='tight')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()




def clippedcolorbar(CS, **kwargs):
    from matplotlib.cm import ScalarMappable
    from numpy import arange, floor, ceil
    fig = CS.ax.get_figure()
    vmin = CS.get_clim()[0]
    vmax = CS.get_clim()[1]
    m = ScalarMappable(cmap=CS.get_cmap())
    m.set_array(CS.get_array())
    m.set_clim(CS.get_clim())
    step = CS.levels[1] - CS.levels[0]
    cliplower = CS.zmin<vmin
    clipupper = CS.zmax>vmax
    noextend = 'extend' in kwargs.keys() and kwargs['extend']=='neither'
    # set the colorbar boundaries
    boundaries = arange((floor(vmin/step)-1+1*(cliplower and noextend))*step, (ceil(vmax/step)+1-1*(clipupper and noextend))*step, step)
    kwargs['boundaries'] = boundaries
    # if the z-values are outside the colorbar range, add extend marker(s)
    # This behavior can be disabled by providing extend='neither' to the function call
    if not('extend' in kwargs.keys()) or kwargs['extend'] in ['min','max']:
        extend_min = cliplower or ( 'extend' in kwargs.keys() and kwargs['extend']=='min' )
        extend_max = clipupper or ( 'extend' in kwargs.keys() and kwargs['extend']=='max' )
        if extend_min and extend_max:
            kwargs['extend'] = 'both'
        elif extend_min:
            kwargs['extend'] = 'min'
        elif extend_max:
            kwargs['extend'] = 'max'
    return fig.colorbar(m, **kwargs)