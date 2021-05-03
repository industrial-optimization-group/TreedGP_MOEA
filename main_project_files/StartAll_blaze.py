#import sys
#sys.path.insert(1, '/scratch/project_2003769/HTGP_MOEA_CSC')
#sys.path.insert(1, '/home/amrzr/Work/Codes/AmzNew/')
#import Main_Execute_Prob as mexeprob
import Main_Execute_SS as mexe
#import Main_Execute_interactive as mexe_int
import pickle
import pickle_to_mat_converter as pickmat
import os
from joblib import Parallel, delayed
import datetime
import os.path
from os import path
from evaluate_population import evaluate_run

data_folder = '/home/amrzr/Work/Codes/data'
init_folder = data_folder + '/initial_samples'

#evaluate_data = False
evaluate_data = True
is_plot = True
#is_plot = False
file_exists_check = False
#file_exists_check = True

#convert_to_mat = True
convert_to_mat = False
#import Telegram_bot.telegram_bot_messenger as tgm
#dims = [5,8,10] #,8]
#dims = [2, 5, 7, 10]
dims = [2]
#dims = [10]

sample_sizes = [2000]
#sample_sizes = [10000, 50000]
#dims = 4
############################################
#folder_data = 'AM_Samples_109_Final'
#folder_data = 'AM_Samples_1000'


"""
objs(1) = max_NOISE;
objs(2) = max_WEMP;
objs(3) = max_DOC;
objs(4) = max_ROUGH;
objs(5) = max_WFUEL;
objs(6) = max_PURCH;
objs(7) = -min_RANGE;
objs(8) = -min_LDMAX;
objs(9) = -min_VCMAX;
objs(10) = PFPF;
"""

#main_directory = 'Offline_Prob_DDMOPP3'
#main_directory = 'Tests_Probabilistic_Finalx_new'
#main_directory = 'Tests_additional_obj1'
#main_directory = 'Tests_Gpy_1'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'
#main_directory = 'Test_Gpy3'
#main_directory = 'Test_DR_4'  #DR = Datatset Reduction
#main_directory = 'Test_DR_Scratch'
main_directory = 'Test_DR_CSC_ARDMatern4'
#main_directory = 'Test_DR_CSC_1'
#main_directory = 'Test_RF'
#main_directory = 'Test_DR_CSC_Final_1'


objectives = [3]
#objectives = [3,5,7]
#objectives = [3, 5, 7]
#objectives = [3,5,7]
#objectives = [2,3,5]
#objectives = [2,3,4,5,6,8,10]
#objectives = [3,5,6,8,10]
#objectives = [3,5,6,8,10]

#problem_testbench = 'DTLZ'
problem_testbench = 'DDMOPP'
#problem_testbench = 'GAA'

#problems = ['DTLZ2']
#problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']

problems = ['P1']
#problems = ['P1','P2','P3','P4']
#problems = ['P1','P3','P4']


#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3
#problems = ['GAA']

#modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2, 7 = Approach_Prob
#modes = [7]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
#modes = [1, 7, 8]
#approaches = ["generic_fullgp","generic_sparsegp"]
#approaches = ["generic_fullgp","generic_sparsegp","strategy_1"]
#approaches = ["generic_fullgp"]
#approaches = ["generic_sparsegp"]
#approaches = ["strategy_1"]
#approaches = ["strategy_2"]
#approaches = ["strategy_3"]
#approaches = ["rf"]
#approaches = ["htgp"]
#approaches = ["generic_sparsegp"]
#approaches = ["generic_fullgp","htgp"]
#approaches = ["generic_fullgp","generic_sparsegp","htgp"]
#approaches = ["generic_sparsegp","htgp"]
approaches = ["htgp"]
#approaches = ["generic_fullgp","generic_sparsegp"]


#sampling = ['BETA', 'MVNORM']
sampling = ['LHS']
#sampling = ['BETA','OPTRAND','MVNORM']
#sampling = ['OPTRAND']
#sampling = ['MVNORM']
#sampling = ['LHS', 'MVNORM']

#emo_algorithm = ['RVEA','IBEA']
emo_algorithm = ['RVEA']
#emo_algorithm = ['IBEA']
#emo_algorithm = ['NSGAIII']
#emo_algorithm = ['MODEL_CV']

interactive = False

#############################################


nruns = 1
parallel_jobs = 1
log_time = str(datetime.datetime.now())


def parallel_execute(run, approach, algo, prob, n_vars, obj, samp, sample_size):
    path_to_file = data_folder + '/test_runs/'+ main_directory \
                + '/Offline_Mode_' + approach + '_' + algo + \
                '/' + samp + '/' + str(sample_size) + '/' + problem_testbench  + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
    print(path_to_file)
    with open(data_folder + '/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
        text_file.write("\n"+path_to_file+"___"+str(run)+"___Started___"+str(datetime.datetime.now()))
    if not os.path.exists(path_to_file):
        os.makedirs(path_to_file)
        print("Creating Directory...")
    if convert_to_mat is False and evaluate_data is False:
        if (problem_testbench == 'DTLZ' and obj < n_vars) or problem_testbench == 'DDMOPP':
            path_to_file = path_to_file + '/Run_' + str(run)
            if path.exists(path_to_file) is False or file_exists_check is False:
                results_dict = mexe.run_optimizer(problem_testbench=problem_testbench, 
                                                    problem_name=prob, 
                                                    nobjs=obj, 
                                                    nvars=n_vars, 
                                                    sampling=samp, 
                                                    nsamples=sample_size, 
                                                    is_data=True, 
                                                    surrogate_type=approach,
                                                    run=run)
                
                outfile = open(path_to_file, 'wb')
                pickle.dump(results_dict, outfile)
                outfile.close()
                with open(data_folder + '/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
                    text_file.write("\n"+path_to_file+"___"+str(run)+"___Ended___"+str(datetime.datetime.now()))
            else:
                with open(data_folder + '/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
                    text_file.write("\n"+path_to_file+"___"+str(run)+"___File already exists!___"+str(datetime.datetime.now()))
                #print('File already exists!')
        
    elif convert_to_mat is True and evaluate_data is False:
        if (problem_testbench == 'DTLZ' and obj < n_vars) or problem_testbench == 'DDMOPP':
            path_to_file = path_to_file + '/Run_' + str(run)
            pickmat.convert(path_to_file, path_to_file+'.mat')
    elif evaluate_data is True:
        if is_plot is True:
            data_median_index = pickle.load(open(path_to_file + '/median_index', "rb"))
            run = data_median_index
            print("Run:",run)  
        if (problem_testbench == 'DTLZ' and obj < n_vars) or problem_testbench == 'DDMOPP':
            path_to_file = path_to_file + '/Run_' + str(run)
            if path.exists(path_to_file) is False:
                print(path_to_file,"-does not exist!")
            else:
                if path.exists(path_to_file+'_evaluated') is False or file_exists_check is False:  
             
                    evaluate_run(init_folder,
                                path_to_file,
                                problem_testbench, 
                                prob, 
                                obj, 
                                n_vars, 
                                samp, 
                                sample_size, 
                                approach,
                                run)
                #else:
                #    print(path_to_file,"-already evaluated")




#try:
temp = Parallel(n_jobs=parallel_jobs)(
    delayed(parallel_execute)(run, approach, algo, prob, n_vars, obj, samp, sample_size)        
    for run in range(nruns)
    for approach in approaches
    for algo in emo_algorithm
    for prob in problems
    for n_vars in dims
    for obj in objectives
    for samp in sampling
    for sample_size in sample_sizes)
#    for run in range(nruns):
#        parallel_execute(run, path_to_file)
#except Exception as e:
#    print(e)
#    with open(data_folder + '/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
#        text_file.write("\n"+ str(e) + "______" + str(datetime.datetime.now()))      


"""
                                outfile_median = open(path_to_file + '_median_index', 'wb')
                                median_index = np.argsort(igd_temp)[len(igd_temp)//2]
                                pickle.dump(, outfile_median)
                                outfile_median.close()
                                """