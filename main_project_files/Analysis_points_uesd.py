import sys
sys.path.insert(1, '/home/amrzr/Work/Codes/TreedGP_MOEA/')
import numpy as np
import pickle
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import csv
#from IGD_calc import igd, igd_plus
from non_domx import ndx
from pygmo import hypervolume as hv
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
import math
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib import rc
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from ranking_approaches import  calc_rank
import seaborn as sns; sns.set()
from brokenaxes import brokenaxes
#rc('font',**{'family':'serif','serif':['Helvetica']})
#rc('text', usetex=True)
#plt.rcParams.update({'font.size': 15})
rc('font',**{'family':'serif','serif':['Helvetica']})
rc('text', usetex=True)

#pareto_front_directory = 'True_Pareto_5000'

mod_p_val = True
metric = 'HV'
#metric = 'IGD'
save_fig = 'pdf'
dims = [5] #,8,10]
#dims = [2, 5, 7, 10]

sample_sizes = [2000]
#sample_sizes = [10000,50000]
#sample_sizes = [2000, 10000, 50000]
data_folder = '/home/amrzr/Work/Codes/data'
init_folder = data_folder + '/initial_samples'


#problem_testbench = 'DTLZ'
problem_testbench = 'DDMOPP'

#main_directory = 'Offline_Prob_DDMOPP3'
#main_directory = 'Tests_Probabilistic_Finalx_new'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'
#main_directory = 'Test_Gpy3'
#main_directory = 'Test_DR_4'
#main_directory = 'Test_DR_CSC_ARDMatern4'
main_directory = 'Test_DR_CSC_Finalx'

#objectives = [3,5,7]
objectives = [3]
#objectives = [2,3,5]
#objectives = [3,5,6,8,10]

#problems = ['DTLZ2']
#problems = ['DTLZ2','DTLZ4']
#problems = ['P4']
#problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['P1','P2','P3','P4']
#problems = ['P1','P2','P3']
problems = ['P3']
#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3

#modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2, 7 = Approach_Prob
#modes = [0,7,70,71]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
#modes = [0,1,7,8]
#modes = [0,1,7,8]
#modes = [1,2]
#mode_length = int(np.size(modes))
#sampling = ['BETA', 'MVNORM']
#sampling = ['LHS']
#sampling = ['BETA','OPTRAND','MVNORM']
#sampling = ['OPTRAND']
#sampling = ['MVNORM']
sampling = ['LHS']
#sampling = ['LHS', 'MVNORM']

emo_algorithm = ['RVEA']

approaches = ["htgp"]

mode_length = int(np.size(approaches))

c1=0.1
c2=0.95
nruns = 31
pool_size = 4
breakaxis = False
#breakaxis = True
plot_boxplot = True

l = [approaches]*nruns
labels = [list(i) for i in zip(*l)]
labels = [y for x in labels for y in x]

p_vals_all_hv = None
p_vals_all_rmse = None
p_vals_all_time = None
index_all = None



for sample_size in sample_sizes:
    for samp in sampling:
        for prob in problems:
            for obj in objectives:
                for n_vars in dims:
                    max_length = math.ceil(sample_size / (10*n_vars))
                    if (problem_testbench == 'DTLZ' and obj < n_vars) or problem_testbench == 'DDMOPP':
                        
                        #fig = plt.figure()
                        #fig.set_size_inches(15, 5)
                        points_sequence_df = []

                        for algo in emo_algorithm:
                            #igd_all = np.zeros([nruns, np.shape(modes)[0]])
                            igd_all = None
                            rmse_mv_all = None
                            solution_ratio_all = None
                            time_taken_all = None
                            for mode, mode_count in zip(approaches,range(np.shape(approaches)[0])):
                                path_to_file = data_folder + '/test_runs/' + main_directory \
                                            + '/Offline_Mode_' + mode + '_' + algo + \
                                            '/' + samp + '/' + str(sample_size) + '/' + problem_testbench  + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                                print(path_to_file)

                                def parallel_execute(run, path_to_file, prob, obj):
                                    path_to_file = path_to_file + '/Run_' + str(run)
                                    infile = open(path_to_file, 'rb')
                                    results_data=pickle.load(infile)
                                    infile.close()
                                    total_points_per_model_sequence = results_data["total_points_per_model_sequence"]
                                    total_points_per_model_sequence = np.asarray(total_points_per_model_sequence)
                                    length_points = np.shape(total_points_per_model_sequence)[0]
                                    print("data length:",length_points)
                                    if length_points < max_length:                                        
                                        last_value = total_points_per_model_sequence[length_points-1]
                                        #print("last value=",last_value)
                                        #print(np.shape(total_points_per_model_sequence))
                                        total_points_per_model_sequence= np.vstack((total_points_per_model_sequence,
                                                                                    np.tile(last_value,(max_length-length_points,1))))
                                    print(np.shape(total_points_per_model_sequence))

                                    return total_points_per_model_sequence


                                temp = Parallel(n_jobs=pool_size)(delayed(parallel_execute)(run, path_to_file, prob, obj) for run in range(nruns))
                                temp=np.asarray(temp)
                                print("shape temp:",np.shape(temp))
                                for i in range(obj):
                                    for j in range(nruns):
                                        for k in range(max_length):
                                            points_sequence_df.append([k+1, j, r'$f_'+str(i+1)+'$', temp[j, k, i]])
                                points_sequence_dfpd = pd.DataFrame(points_sequence_df, columns=['Iteration', 'Run', 'Objective', 'Number of points'])
                                color_map = plt.cm.get_cmap('viridis')
                                color_map = color_map(np.linspace(0, 1, obj+1))
                                
                                #bax = brokenaxes(xlims=((0, 50), (450, 500)), hspace=.05)
                                if breakaxis is True:
                                    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1,sharey=True, gridspec_kw={'width_ratios': [3, 1]})
                                    fig.set_size_inches(5, 4)
                                    ax1 = sns.lineplot(x="Iteration", y="Number of points",
                                                    hue="Objective", style="Objective", #err_style="bars",
                                                    markers=True, dashes=False, data=points_sequence_dfpd, ax=ax1, palette=color_map)
                                    ax2 = sns.lineplot(x="Iteration", y="Number of points",
                                                    hue="Objective", style="Objective", #err_style="bars",
                                                    markers=True, dashes=False, data=points_sequence_dfpd, ax=ax2, palette=color_map)
                                    ax1.set_xlim(0, c1*max_length)
                                    ax2.set_xlim(c2*max_length, max_length)   
                                    #ax1.figure.set_size_inches(5, 5)
                                    #ax2.figure.set_size_inches(3, 5)             
                                    #ax2.get_yaxis().set_visible(False)
                                    ax1.set_xlabel("")
                                    ax2.set_xlabel("")
                                    ax2.set_ylabel("")
                                    ax1.set_ylabel('Number of points')
                                    fig.text(0.45, 0, 'Iteration', va='center')
                                    ax1.get_legend().remove()
                                    ax2.get_legend().remove()
                                    #ax2.legend(loc=(1.025, 0.7), title='Objective')
                                    handles, labels = ax1.get_legend_handles_labels()
                                    # ax.legend(handles=handles, labels=labels)
                                    fig.legend(handles=handles[0:], labels=labels[0:], frameon=False, loc='upper center',
                                            bbox_to_anchor=(1, 0.9), ncol=1, title='Objective')
                                    #ax1.yaxis.tick_top()
                                    #ax2.yaxis.tick_bottom()
                                    #fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
                                    frac1=1/3
                                    frac2=1/frac1
                                    d = .02  # how big to make the diagonal lines in axes coordinates
                                    # arguments to pass to plot, just so we don't keep repeating them
                                    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
                                    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)        # top-left diagonal
                                    ax2.plot((1-d, 1+ d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

                                    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
                                    ax2.plot((-frac2*d, +frac2*d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
                                    
                                    ax2.plot((-frac2*d, +frac2*d), (-d, +d), **kwargs)  # top-right diagonal
                                    if samp == 'MVNORM':
                                        sampx='MVNS'
                                    else:
                                        sampx=samp
                                    if samp == 'MVNORM':
                                        sampx='MVNS'
                                    else:
                                        sampx=samp
                                    fig.text(0.2, 0.9, str(sample_size)+', '+sampx+', '+prob+', '+str(obj)+', '+str(n_vars), va='center')
                                    fig.show()
                                    filename_fig =  data_folder + '/test_runs/'+ main_directory + '/Plots/BRK_Points_consumed_progress_' + str(sample_size) + '_' + samp + '_' + algo + '_' + prob + '_' + str(
                                        obj) + '_' + str(n_vars)
                                    if save_fig == 'png':
                                        fig.savefig(filename_fig + '.png', bbox_inches='tight')
                                    else:
                                        fig.savefig(filename_fig + '.pdf', bbox_inches='tight')
                                    ax1.clear()
                                    ax2.clear()
                                else:
                                    fig = plt.figure()
                                    fig.set_size_inches(5, 4)
                                    ax = sns.lineplot(x="Iteration", y="Number of points",
                                                    hue="Objective", style="Objective", #err_style="bars",
                                                    markers=True, dashes=False, data=points_sequence_dfpd, palette=color_map)
                                    if sample_size != 2000:
                                        ax.set(xscale="log")
                                    ax.get_legend().remove()
                                    handles, labels = ax.get_legend_handles_labels()
                                    # ax.legend(handles=handles, labels=labels)
                                    fig.legend(handles=handles[0:], labels=labels[0:], frameon=False, loc='upper center',
                                            bbox_to_anchor=(1, 0.9), ncol=1, title='Objective')
                                    if samp == 'MVNORM':
                                        sampx='MVNS'
                                    else:
                                        sampx=samp
                                    fig.text(0.2, 0.9, str(sample_size)+', '+sampx+', '+prob+', '+str(obj)+', '+str(n_vars), va='center')
                                    fig = ax.get_figure()
                                    fig.show()
                                
                                    filename_fig =  data_folder + '/test_runs/'+ main_directory + '/Plots/Points_consumed_progress_' + str(sample_size) + '_' + samp + '_' + algo + '_' + prob + '_' + str(
                                        obj) + '_' + str(n_vars)
                                    if save_fig == 'png':
                                        fig.savefig(filename_fig + '.png', bbox_inches='tight')
                                    else:
                                        fig.savefig(filename_fig + '.pdf', bbox_inches='tight')
                                    
                                    ax.clear()



                                



