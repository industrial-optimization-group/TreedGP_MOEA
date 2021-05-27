import sys
sys.path.insert(1, '/home/amrzr/Work/Codes/TreedGP_MOEA/')
import copy
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
import seaborn as sns
import pandas as pd

################## Grouped boxplots #####################

#rc('font',**{'family':'serif','serif':['Helvetica']})
#rc('text', usetex=True)
#plt.rcParams.update({'font.size': 15})


#pareto_front_directory = 'True_Pareto_5000'
is_plot = False
mod_p_val = True
perform_bonferonni = True
metric = 'HV'
#metric = 'IGD'
save_fig = 'pdf'
#dims = [5,8,10] #,8,10]
dims = [2, 5, 7, 10]
#dims = [10]
sample_sizes = [2000]
#sample_sizes = [2000, 10000]#, 50000]
#sample_sizes = [10000, 50000]

data_folder = '/home/amrzr/Work/Codes/data'
init_folder = data_folder + '/initial_samples'



#main_directory = 'Offline_Prob_DDMOPP3'
#main_directory = 'Tests_Probabilistic_Finalx_new'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'
#main_directory = 'Test_Gpy3'
#main_directory = 'Test_DR_4'
#main_directory = 'Test_DR_CSC_Final_1'
main_directory = 'Test_DR_CSC_ARDMatern4'

objectives = [3,5,7]
#objectives = [5]
#objectives = [2,3,5]
#objectives = [3,5,6,8,10]


problem_testbench = 'DTLZ'
#problem_testbench = 'DDMOPP'

#problems = ['DTLZ7']
#problems = ['DTLZ2','DTLZ4']
#problems = ['P1']
problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['P1','P2','P3','P4']
#problems = ['P4']
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
sampling = ['LHS', 'MVNORM']

#emo_algorithm = ['RVEA','IBEA']
emo_algorithm = ['RVEA']
#emo_algorithm = ['IBEA']
#emo_algorithm = ['NSGAIII']
#emo_algorithm = ['MODEL_CV']




#approaches = ['Generic', 'Approach 1', 'Approach 2', 'Approach 3']
#approaches = ['Generic', 'Approach 1', 'Approach 2', 'Approach X']
#approaches = ['Initial sampling','Generic', 'Approach Prob','Approach Hybrid']
#approaches = ['Initial sampling','Generic_RVEA','Generic_IBEA']
#approaches = ['Initial sampling','Prob Old', 'Prob constant 1','Prob FE/FEmax']
#approaches = ['Generic', 'Approach Prob','Approach Hybrid']
#approaches = ['Generic', 'Probabilistic','Hybrid']
#approaches = ["generic_fullgp","generic_sparsegp","strategy_2","strategy_3","rf_ne10","rf", "htgp0"]
#approaches = ["generic_fullgp0","generic_fullgp","generic_sparsegp0","generic_sparsegp", "htgp0", "htgp1" , "htgp"]
#approaches = ["generic_fullgp","generic_sparsegp","htgp_1","htgp"]
#approaches = ["generic_fullgp","generic_sparsegp_50","generic_sparsegp","htgp_mse"]
approaches = ["generic_fullgp","generic_sparsegp","htgp"]
#approaches = ["generic_sparsegp","htgp"]
#approaches_nice = ['Sparse GP', 'Treed GP']
approaches_nice = ['Full GP','Sparse GP', 'Treed GP']

mode_length = int(np.size(approaches))
#approaches = ['7', '9', '11']
#"DTLZ6": {"2": [10, 10], "3": [10, 10, 10], "5": [10, 10, 10, 10, 10]},
#"DTLZ4": {"2": [4, 4], "3": [4, 4, 4], "5": [4, 4, 4, 4, 4]},
#hv_ref = {"DTLZ2": {"2": [3, 3], "3": [3, 3, 3], "5": [3, 3, 3, 3, 3],  "7": [3, 3, 3, 3, 3, 3, 3]},
#          "DTLZ4": {"2": [3, 3.1], "3": [3, 3, 3], "5": [3, 3, 3, 3, 3] ,  "7": [3, 3, 3, 3, 3, 3, 3]},
#          "DTLZ5": {"2": [2.5, 3], "3": [2.5, 3, 3], "5": [2, 2, 2, 2, 2] ,  "7": [3, 3, 3, 3, 3, 3, 3]},
#          "DTLZ6": {"2": [10, 10], "3": [10, 10, 10], "5": [7, 7, 7, 7, 7] ,  "7": [10, 10, 10, 10, 10, 10, 10]},
#          "DTLZ7": {"2": [1, 20], "3": [1, 1, 30], "5": [1, 1, 1, 1, 50] ,  "7": [1, 1, 1, 1, 1, 1, 70]}}

hv_ref = {"DTLZ2": {"2": [3, 3], "3": [6, 6, 6], "5": [6, 6, 6, 6, 6],  "7": [6, 6, 6, 6, 6, 6, 6]},
          "DTLZ4": {"2": [3, 3.1], "3": [6, 6, 6], "5": [6, 6, 6, 6, 6] ,  "7": [6, 6, 6, 6, 6, 6, 6]},
          "DTLZ5": {"2": [2.5, 3], "3": [6, 6, 6], "5": [6, 6, 6, 6, 6] ,  "7": [6, 6, 6, 6, 6, 6, 6]},
          "DTLZ6": {"2": [10, 10], "3": [20, 20, 20], "5": [20, 20, 20, 20, 20] ,  "7": [20, 20, 20, 20, 20, 20, 20]},
          "DTLZ7": {"2": [1, 20], "3": [1, 1, 40], "5": [1, 1, 1, 1, 50] ,  "7": [1, 1, 1, 1, 1, 1, 70]}}


nruns = 11
pool_size = 1

plot_boxplot = True

l = [approaches]*nruns
labels = [list(i) for i in zip(*l)]
labels = [y for x in labels for y in x]

p_vals_all_hv = None
p_vals_all_rmse = None
p_vals_all_time = None
index_all = None

df_all = None

for sample_size in sample_sizes:
    for samp in sampling:
        for prob in problems:
            for obj in objectives:
                for n_vars in dims:
                    if (problem_testbench == 'DTLZ' and obj < n_vars) or problem_testbench == 'DDMOPP':
                        #fig = plt.figure(1, figsize=(10, 10))
                        #fig = plt.figure()
                        #ax = fig.add_subplot(111)
                        #fig.set_size_inches(5, 5)
                        
                        fig = plt.figure()
                        # ax = fig.add_subplot(111, projection='3d')
                        #ax = fig.add_subplot(111)
                        fig.set_size_inches(15, 5)
                        #plt.xlim(0, 1)
                        #plt.ylim(0, 1)

                        #if save_fig == 'pdf':
                        #    plt.rcParams["text.usetex"] = True
                        #with open(pareto_front_directory + '/True_5000_' + prob + '_' + obj + '.txt') as csv_file:
                        #    csv_reader = csv.reader(csv_file, delimiter=',')
                        
                        #############
                        #if problem_testbench is 'DTLZ':
                        #    pareto_front = np.genfromtxt(pareto_front_directory + '/True_5000_' + prob + '_' + str(obj) + '.txt'
                        #                             , delimiter=',')
                        ##############

                        #path_to_file = pareto_front_directory + '/' + 'Pareto_Weld'
                        #infile = open(path_to_file, 'rb')
                        #pareto_front = pickle.load(infile)
                        #infile.close()
                        #problem_weld = WeldedBeam()
                        #pareto_front = problem_weld.pareto_front()

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
                                    rmse_mv_sols = 0
                                    path_to_file = path_to_file + '/Run_' + str(run)
                                    infile = open(path_to_file, 'rb')
                                    results_data=pickle.load(infile)
                                    infile.close()
                                    surrogate_objectives = results_data["obj_solutions"]
                                    time_taken=results_data["time_taken"]
                                    infile = open(path_to_file+'_evaluated', 'rb')
                                    results_data=pickle.load(infile)
                                    infile.close()
                                    underlying_objectives = results_data['obj_solutions_evaluated']

                                    if problem_testbench is 'DDMOPP':
                                        ref = [obj*np.sqrt(2)]*obj
                                    else:
                                        ref = hv_ref[prob][str(obj)]

                                    if np.shape(underlying_objectives)[0] > 1:
                                        non_dom_front = ndx(underlying_objectives)
                                        underlying_objectives_nds = underlying_objectives[non_dom_front[0][0]]
                                    else:
                                        underlying_objectives_nds = underlying_objectives.reshape(1, obj)
                                    solution_ratio = 0
                                    hyp = hv(underlying_objectives_nds)
                                    hv_x = hyp.compute(ref)
                                    #print(np.amax(underlying_objectives_nds,axis=0))
                                    print(np.shape(underlying_objectives_nds))

                                    for i in range(np.shape(surrogate_objectives)[0]):
                                        rmse_mv_sols += distance.euclidean(surrogate_objectives[i,:],underlying_objectives[i,:])
                                    rmse_mv_sols = rmse_mv_sols/np.shape(surrogate_objectives)[0]

                                    return [hv_x, rmse_mv_sols, time_taken, solution_ratio]


                                temp = Parallel(n_jobs=pool_size)(delayed(parallel_execute)(run, path_to_file, prob, obj) for run in range(nruns))
                                temp=np.asarray(temp)
                                igd_temp = np.transpose(temp[:, 0])
                                print("HV:", igd_temp)
                                solution_ratio_temp = np.transpose(temp[:, 3])
                                rmse_mv_sols_temp = np.transpose(temp[:, 1])
                                time_taken_temp = np.transpose(temp[:, 2])
                                
                                outfile_median = open(path_to_file + '/median_index', 'wb')
                                median_index = np.argsort(igd_temp)[len(igd_temp)//2]
                                pickle.dump(median_index, outfile_median)
                                outfile_median.close()


                                if plot_boxplot is True:
                                    if igd_all is None:
                                        igd_all = igd_temp
                                        rmse_mv_all = rmse_mv_sols_temp
                                        solution_ratio_all = solution_ratio_temp
                                        time_taken_all = time_taken_temp
                                    else:
                                        igd_all = np.vstack((igd_all, igd_temp))
                                        rmse_mv_all = np.vstack((rmse_mv_all,rmse_mv_sols_temp))                            
                                        solution_ratio_all = np.vstack((solution_ratio_all,solution_ratio_temp))
                                        time_taken_all = np.vstack((time_taken_all,time_taken_temp))

                            igd_all = np.transpose(igd_all)
                            rmse_mv_all = np.transpose(rmse_mv_all)
                            solution_ratio_all = np.transpose(solution_ratio_all)
                            time_taken_all = np.transpose(time_taken_all)

                        
                            lenx = np.zeros(int(math.factorial(mode_length)/((math.factorial(mode_length-2))*2)))
                            p_value_rmse =  copy.deepcopy(lenx)
                            p_value_hv = copy.deepcopy(lenx)
                            p_value_time = copy.deepcopy(lenx)
                            p_cor_temp_hv =  copy.deepcopy(lenx)
                            p_cor_temp_rmse = copy.deepcopy(lenx)
                            p_cor_temp_time = copy.deepcopy(lenx)
                            count = 0
                            count_prev = 0
                            for i in range(mode_length-1):
                                for j in range(i+1,mode_length):
                                    w, p1 = wilcoxon(x=igd_all[:, i], y=igd_all[:, j])
                                    p_value_hv[count] = p1
                                    w, p2 = wilcoxon(x=rmse_mv_all[:, i], y=rmse_mv_all[:, j])
                                    p_value_rmse[count] = p2
                                    w, p3 = wilcoxon(x=time_taken_all[:, i], y=time_taken_all[:, j])
                                    p_value_time[count] = p3
                                    count +=1
                                if perform_bonferonni is True:
                                    if mod_p_val is True:
                                        r, p_cor_temp_hv[count_prev:count], alps, alpb = multipletests(p_value_hv[count_prev:count], alpha=0.05, method='bonferroni',
                                                                            is_sorted=False, returnsorted=False)
                                        r, p_cor_temp_rmse[count_prev:count], alps, alpb = multipletests(p_value_rmse[count_prev:count], alpha=0.05, method='bonferroni',
                                                                            is_sorted=False, returnsorted=False)
                                        r, p_cor_temp_time[count_prev:count], alps, alpb = multipletests(p_value_time[count_prev:count], alpha=0.05, method='bonferroni',
                                                                            is_sorted=False, returnsorted=False)
                                        count_prev = count
                            if perform_bonferonni is True:
                                p_cor_hv = p_cor_temp_hv
                                p_cor_rmse = p_cor_temp_rmse
                                p_cor_time = p_cor_temp_time
                            else:
                                p_cor_hv = p_value_hv
                                p_cor_rmse = p_value_rmse
                                p_cor_time = p_value_time

                            if mod_p_val is False:
                                r, p_cor_hv, alps, alpb = multipletests(p_value_hv, alpha=0.05, method='bonferroni', is_sorted=False,
                                                                    returnsorted=False)
                                r, p_cor_rmse, alps, alpb = multipletests(p_value_rmse, alpha=0.05, method='bonferroni', is_sorted=False,
                                                                    returnsorted=False)
                                r, p_cor_time, alps, alpb = multipletests(p_value_time, alpha=0.05, method='bonferroni', is_sorted=False,
                                                                    returnsorted=False)
                            current_index = [sample_size, samp, prob, obj, n_vars]
                            ranking_hv = calc_rank(p_cor_hv, np.median(igd_all, axis=0),mode_length)
                            ranking_rmse = calc_rank(p_cor_rmse, np.median(rmse_mv_all, axis=0)*-1,mode_length)
                            ranking_time = calc_rank(p_cor_time, np.median(time_taken_all, axis=0)*-1,mode_length)
                            #adding other indicators mean, median, std dev
                            #p_cor = (np.asarray([p_cor, np.mean(igd_all, axis=0),
                            #                             np.median(igd_all, axis=0),
                            #                             np.std(igd_all, axis=0)])).flatten()
                            p_cor_hv = np.hstack((p_cor_hv, np.mean(igd_all, axis=0),
                                                        np.median(igd_all, axis=0),
                                                        np.std(igd_all, axis=0), ranking_hv))
                            p_cor_hv = np.hstack((current_index,p_cor_hv))

                            p_cor_rmse = np.hstack((p_cor_rmse, np.mean(rmse_mv_all, axis=0),
                                                        np.median(rmse_mv_all, axis=0),
                                                        np.std(rmse_mv_all, axis=0), ranking_rmse))
                            p_cor_rmse = np.hstack((current_index,p_cor_rmse))

                            p_cor_time = np.hstack((p_cor_time, np.mean(time_taken_all, axis=0),
                                                        np.median(time_taken_all, axis=0),
                                                        np.std(time_taken_all, axis=0), ranking_time))
                            p_cor_time = np.hstack((current_index,p_cor_time))
                            
                            if p_vals_all_hv is None:
                                p_vals_all_hv = p_cor_hv
                                p_vals_all_rmse = p_cor_rmse
                                p_vals_all_time = p_cor_time
                            else:
                                p_vals_all_hv = np.vstack((p_vals_all_hv, p_cor_hv))
                                p_vals_all_rmse = np.vstack((p_vals_all_rmse, p_cor_rmse))
                                p_vals_all_time = np.vstack((p_vals_all_time, p_cor_time))
                            #l = []
                            #l.append('xyz')
                            #inst=l*11
                            df_temp = None
                            #instance = samp + '_' + str(sample_size) + '_' + 'DBMOPP'  + '_' + prob + '_' + str(obj) + '_' + str(n_vars)
                            instance = str(sample_size) + '_' + 'DBMOPP'  + '_' + prob + '_' + str(obj)
                            for approach_name, approach_count in zip(approaches_nice,range(np.shape(approaches)[0])):
                                for i in range(nruns):                                    
                                    if df_temp is None:
                                        data_temp = {'Approaches':[approach_name], 
                                                    'HV':[igd_all[i,approach_count]],
                                                    'RMSE': [rmse_mv_all[i,approach_count]], 
                                                    'Time (s)':[time_taken_all[i,approach_count]], 
                                                    'Instances': [instance], 
                                                    'Sampling':[samp], 
                                                    'Objectives':[obj],
                                                    'n':[n_vars],
                                                    'Sample Size':[sample_size]}
                                        df_temp = pd.DataFrame(data_temp)
                                    else:
                                        data_temp = {'Approaches':approach_name, 
                                                    'HV':igd_all[i,approach_count], 
                                                    'RMSE': rmse_mv_all[i,approach_count], 
                                                    'Time (s)':time_taken_all[i,approach_count], 
                                                    'Instances': instance, 
                                                    'Sampling':samp, 
                                                    'Objectives':obj,
                                                    'n':n_vars,
                                                    'Sample Size':sample_size}
                                        df_temp = df_temp.append(data_temp, ignore_index=True)
                            
                        if df_all is None:
                            df_all = df_temp
                        else:
                            df_all = df_all.append(df_temp, ignore_index=True)

    sns.set(rc={'figure.figsize':(20,15)})
    sns.set_theme(style="ticks", palette="pastel")
    hv_plt = sns.catplot(x="Sampling", y="HV",
                hue="Approaches",
                data=df_all, row="n",col="Instances",kind="box",sharey=False)
    plt.tight_layout()
    plt.savefig('hv_grouped_'+problem_testbench+'_'+str(sample_size)+'.pdf')
    plt.clf()

    sns.set_theme(style="ticks", palette="pastel")
    rmse_plt = sns.catplot(x="Sampling", y="RMSE",
                hue="Approaches",
                data=df_all, row="n",col="Instances",kind="box",sharey=False)
    plt.tight_layout()
    plt.savefig('rmse_grouped_'+problem_testbench+'_'+str(sample_size)+'.pdf')
    plt.clf()

    sns.set_theme(style="ticks", palette="pastel")
    time_plt = sns.catplot(x="Sampling", y="Time (s)",
                hue="Approaches",
                data=df_all, row="n",col="Instances",kind="box",sharey=False)
    plt.tight_layout()
    plt.savefig('time_grouped_'+problem_testbench+'_'+str(sample_size)+'.pdf')
    plt.clf()
"""
sns.set(rc={'figure.figsize':(20,15)})
sns.set_theme(style="ticks", palette="pastel")
hv_plt = sns.boxplot(x="Instances", y="HV",
            hue="Approaches",
            data=df_all)
#hv_plt = sns.despine(trim=True)
hv_plt.set_xticklabels(hv_plt.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.savefig('hv_grouped.pdf')
plt.clf()


rmse_plt = sns.boxplot(x="Instances", y="RMSE",
            hue="Approaches",
            data=df_all)
#rmse_plt = sns.despine(trim=True)
rmse_plt.set_xticklabels(rmse_plt.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.savefig('rmse_grouped.pdf')
plt.clf()

time_plt = sns.boxplot(x="Instances", y="Time (s)",
            hue="Approaches",
            data=df_all)
#time_plt = sns.despine(trim=True)
time_plt.set_xticklabels(time_plt.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.savefig('time_grouped.pdf')
plt.clf()
"""




