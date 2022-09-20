#from turtle import color
#from turtle import color
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

"""
def plot_partition(surrogate_problem, pop, iteration, delta, plot_folder):
    GP_leaves0 = surrogate_problem.objectives[0]._model.error_leaves
    GP_leaves1 = surrogate_problem.objectives[1]._model.error_leaves
    tree_model_0 = surrogate_problem.objectives[0]._model.regr
    tree_model_1 = surrogate_problem.objectives[1]._model.regr
    leaf_samples_0 = tree_model_0.tree_.n_node_samples
    leaf_samples_1 = tree_model_1.tree_.n_node_samples
    x1 = np.arange(-1, 1, 0.01)
    x2 = np.arange(-1, 1, 0.01)
    X = None
    for i in x1:
        for j in x2:
            if X is None:
                X = [i, j]
            else:
                X = np.vstack((X,[i,j]))
    #cmap = cm.get_cmap('prism', 30) 

    x1, x2 = np.meshgrid(x1, x2)

    #cmap = cm.get_cmap('nipy_spectral', 77)
    
    
    h_max = 200
    w_max = 200
    diff=1.01

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)    
       
    y0=tree_model_0.apply(X).reshape(np.shape(x1))
    y0_unique = np.unique(y0)
    y0rot=np.rot90(y0)
    for i in y0_unique:
        locs = np.where(y0rot==i)
        loc_y = locs[0]
        loc_x = locs[1]
        height = (2*(np.max(loc_y) - np.min(loc_y))/h_max)
        width = (2*(np.max(loc_x) - np.min(loc_x))/w_max)
        #xy=((2*(np.min(loc_x)/w_max)-1),(2*((h_max-np.max(loc_y))/h_max)-1))
        xy=((2*((h_max-np.max(loc_y))/h_max)-diff),(2*(np.min(loc_x)/w_max)-1))
        if i==GP_leaves0[-1] and delta!=0:    
            ax.add_patch(Rectangle(xy,  height, width, color='red'))
            #plt.text(xy[0],xy[1],leaf_samples_0[i],fontsize=15)
            plt.text(xy[0],xy[1],i,fontsize=15)
        elif np.isin(i,GP_leaves0):
            ax.add_patch(Rectangle(xy,  height, width, color='yellow'))
            #plt.text(xy[0],xy[1],leaf_samples_0[i],fontsize=15)
            plt.text(xy[0],xy[1],i,fontsize=15)
        else:
            ax.add_patch(Rectangle(xy,  height, width))
    plt.scatter(pop[:,0],pop[:,1],color='black',zorder=10) 

    #CS = ax.contourf(x1,x2,y1,77, cmap=cmap) #,linewidths=0.7,colors='black')
    #disp.ax_.scatter(X[:, 0], X[:, 1], edgecolor="k")
    ax.set_xlim(right=1, left=-1)
    ax.set_ylim(top=1, bottom=-1)
    plt.title(r'$f_1, I= {cnt}$'.format(cnt=str(iteration)))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    fig.tight_layout()
    plt.savefig(plot_folder+'/f1_partition_'+str(iteration)+'.pdf')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)        
    y1=tree_model_1.apply(X).reshape(np.shape(x1))
    y1_unique = np.unique(y1)
    y1rot=np.rot90(y1)
    for i in y1_unique:
        locs = np.where(y1rot==i)
        loc_y = locs[0]
        loc_x = locs[1]
        height = (2*(np.max(loc_y) - np.min(loc_y))/h_max)
        width = (2*(np.max(loc_x) - np.min(loc_x))/w_max)
        #xy=((2*(np.min(loc_x)/w_max)-1),(2*((h_max-np.max(loc_y))/h_max)-1))
        xy=((2*((h_max-np.max(loc_y))/h_max)-diff),(2*(np.min(loc_x)/w_max)-1))
        if i==GP_leaves1[-1] and delta!=0:    
            ax.add_patch(Rectangle(xy,  height, width, color='red'))
            #plt.text(xy[0],xy[1],leaf_samples_1[i],fontsize=15)
            plt.text(xy[0],xy[1],i,fontsize=15)
        elif np.isin(i,GP_leaves1):
            ax.add_patch(Rectangle(xy,  height, width, color='yellow'))
            #plt.text(xy[0],xy[1],leaf_samples_1[i],fontsize=15)
            plt.text(xy[0],xy[1],i,fontsize=15)
        else:
            ax.add_patch(Rectangle(xy,  height, width))

    ax.scatter(pop[:,0],pop[:,1],color='black', zorder=10)
    #CS = ax.contourf(x1,x2,y1,77, cmap=cmap) #,linewidths=0.7,colors='black')
    #disp.ax_.scatter(X[:, 0], X[:, 1], edgecolor="k")
    ax.set_xlim(right=1, left=-1)
    ax.set_ylim(top=1, bottom=-1)
    plt.title(r'$f_2, I= {cnt}$'.format(cnt=str(iteration)))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    fig.tight_layout()
    plt.savefig(plot_folder+'/f2_partition_'+str(iteration)+'.pdf')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    
"""
def plot_partition(surrogate_problem, pop, iteration, delta, plot_folder):
    GP_leaves0 = surrogate_problem.objectives[0]._model.error_leaves
    GP_leaves1 = surrogate_problem.objectives[1]._model.error_leaves
    tree_model_0 = surrogate_problem.objectives[0]._model.regr
    tree_model_1 = surrogate_problem.objectives[1]._model.regr
    leaf_samples_0 = tree_model_0.tree_.n_node_samples
    leaf_samples_1 = tree_model_1.tree_.n_node_samples
    x1 = np.arange(-1, 1, 0.01)
    x2 = np.arange(-1, 1, 0.01)
    X = None
    for i in x1:
        for j in x2:
            if X is None:
                X = [i, j]
            else:
                X = np.vstack((X,[i,j]))
    #cmap = cm.get_cmap('prism', 30) 

    x1, x2 = np.meshgrid(x1, x2)

    #cmap = cm.get_cmap('nipy_spectral', 77)
    
    
    h_max = 200
    w_max = 200
    diff=1.01

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)    
       
    y0=tree_model_0.apply(X).reshape(np.shape(x1))
    y0_unique = np.unique(y0)
    y0rot=np.rot90(y0)
    for i in y0_unique:
        locs = np.where(y0rot==i)
        loc_y = locs[0]
        loc_x = locs[1]
        height = (2*(np.max(loc_y) - np.min(loc_y))/h_max)
        width = (2*(np.max(loc_x) - np.min(loc_x))/w_max)
        xy=((2*(np.min(loc_x)/w_max)-1),(2*((h_max-np.max(loc_y))/h_max)-1))
        #xy=((2*((h_max-np.max(loc_y))/h_max)-diff),(2*(np.min(loc_x)/w_max)-1))
        if i==GP_leaves0[-1] and delta[0]!=0:    
            ax.add_patch(Rectangle(xy,  width,  height,color='red'))
            plt.text(xy[0],xy[1],leaf_samples_0[i],fontsize=15)
            #plt.text(xy[0],xy[1],i,fontsize=15)
        elif np.isin(i,GP_leaves0):
            ax.add_patch(Rectangle(xy,  width,  height,color='yellow'))
            plt.text(xy[0],xy[1],leaf_samples_0[i],fontsize=15)
            #plt.text(xy[0],xy[1],i,fontsize=15)
        else:
            ax.add_patch(Rectangle(xy, width,  height))
    plt.scatter(pop[:,0],pop[:,1],color='black',zorder=10) 

    #CS = ax.contourf(x1,x2,y1,77, cmap=cmap) #,linewidths=0.7,colors='black')
    #disp.ax_.scatter(X[:, 0], X[:, 1], edgecolor="k")
    ax.set_xlim(right=1, left=-1)
    ax.set_ylim(top=1, bottom=-1)
    plt.title(r'$f_1, I= {cnt}$'.format(cnt=str(iteration)))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    fig.tight_layout()
    plt.savefig(plot_folder+'/f1_partition_'+str(iteration)+'.pdf')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)        
    y1=tree_model_1.apply(X).reshape(np.shape(x1))
    y1_unique = np.unique(y1)
    y1rot=np.rot90(y1)
    for i in y1_unique:
        locs = np.where(y1rot==i)
        loc_y = locs[0]
        loc_x = locs[1]
        height = (2*(np.max(loc_y) - np.min(loc_y))/h_max)
        width = (2*(np.max(loc_x) - np.min(loc_x))/w_max)
        xy=((2*(np.min(loc_x)/w_max)-1),(2*((h_max-np.max(loc_y))/h_max)-diff))
        #xy=((2*((h_max-np.max(loc_y))/h_max)-diff),(2*(np.min(loc_x)/w_max)-1))
        if i==GP_leaves1[-1] and delta[1]!=0:    
            ax.add_patch(Rectangle(xy,  width, height, color='red'))
            plt.text(xy[0],xy[1],leaf_samples_1[i],fontsize=15)
            #plt.text(xy[0],xy[1],i,fontsize=15)
        elif np.isin(i,GP_leaves1):
            ax.add_patch(Rectangle(xy,  width,  height,color='yellow'))
            plt.text(xy[0],xy[1],leaf_samples_1[i],fontsize=15)
            #plt.text(xy[0],xy[1],i,fontsize=15)
        else:
            ax.add_patch(Rectangle(xy, width,   height))

    ax.scatter(pop[:,0],pop[:,1],color='black', zorder=10)
    #CS = ax.contourf(x1,x2,y1,77, cmap=cmap) #,linewidths=0.7,colors='black')
    #disp.ax_.scatter(X[:, 0], X[:, 1], edgecolor="k")
    ax.set_xlim(right=1, left=-1)
    ax.set_ylim(top=1, bottom=-1)
    plt.title(r'$f_2, I= {cnt}$'.format(cnt=str(iteration)))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    fig.tight_layout()
    plt.savefig(plot_folder+'/f2_partition_'+str(iteration)+'.pdf')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    