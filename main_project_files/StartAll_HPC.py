import sys
#sys.path.insert(1, '/scratch/project_2003769/HTGP_MOEA_CSC')
sys.path.insert(1, '/home/amrzr/Work/Codes/TreedGP_MOEA/')

from main_project_files.evaluate_population import evaluate_run_archive
import Main_Execute_SS as mexe
#import Main_Execute_interactive as mexe_int
import pickle
import pickle_to_mat_converter as pickmat
import os
from joblib import Parallel, delayed
import datetime
import os.path
from os import path
import traceback
from evaluate_population import evaluate_run

data_folder = '/home/amrzr/Work/Codes/data'
init_folder = data_folder + '/initial_samples'

evaluate_data = False
#evaluate_data = True
evaluate_data_archive = False
#evaluate_data_archive = True
is_plot = False
#is_plot = True
file_exists_check = False
#file_exists_check = True
convert_to_mat = False
#convert_to_mat = True
interactive = False

main_directory = 'Test_DR_CSC_Finalx'

nruns = 1
parallel_jobs = 1

dims = [2] #, 5, 7, 10]
sampling = ['LHS'] #, 'MVNORM']
sample_sizes = [2000]#,10000, 50000]

#objectives = [3,5,7]
objectives = [2]

#problem_testbench = 'DTLZ'
problem_testbench = 'DDMOPP'

#problems = ['DTLZ2'] #,'DTLZ4','DTLZ5','DTLZ6','DTLZ7']
problems = ['P1'] #,'P2','P3','P4']

#approaches = ["generic_fullgp","generic_sparsegp","htgp"]
#approaches = ["generic_sparsegp","htgp"]
approaches = ["htgp"]
emo_algorithm = ['RVEA']

log_time = str(datetime.datetime.now())

def parallel_execute(run, approach, algo, prob, n_vars, obj, samp, sample_size):
    run=run+1
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
                print('Starting Run!')
                try:
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
                except Exception as e:
                    print(e)
                    with open(data_folder + '/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
                        text_file.write("\n"+path_to_file+"___"+str(run)+"______"+str(e) + "______" + traceback.format_exc()+"___Error___"+str(datetime.datetime.now()))
            else:
                with open(data_folder + '/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
                    text_file.write("\n"+path_to_file+"___"+str(run)+"___File already exists!___"+str(datetime.datetime.now()))

        
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
                    if evaluate_data_archive is True:
                        evaluate_run_archive(init_folder,
                                            path_to_file,
                                            problem_testbench, 
                                            prob, 
                                            obj, 
                                            n_vars, 
                                            samp, 
                                            sample_size, 
                                            approach,
                                            run)
                    else:
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
