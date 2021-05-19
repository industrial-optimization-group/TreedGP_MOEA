import matlab.engine
import pickle
import numpy as np
from desdeo_problem.testproblems.TestProblems import test_problem_builder

eng = matlab.engine.start_matlab()
s = eng.genpath('./matlab_files')
eng.addpath(s, nargout=0)
is_plot = 1
plot_init = 0

data_folder = '/home/amrzr/Work/Codes/data'

def evaluate_population(population,
                        init_folder,
                        approaches,
                        problem_testbench, 
                        problem_name, 
                        nobjs, 
                        nvars, 
                        sampling, 
                        nsamples,
                        run=0):
    s = eng.genpath(init_folder)
    eng.addpath(s, nargout=0)
    size_pop = np.shape(population)[0]
    if problem_testbench == 'DTLZ':
        prob = test_problem_builder(
                    name=problem_name, n_of_objectives=nobjs, n_of_variables=nvars
                )
        np_a = prob.evaluate(population)[0]
    elif problem_testbench == 'DDMOPP':
        population = matlab.double(population.tolist())
        objs_evaluated = eng.evaluate_python(population, init_folder, approaches, sampling, problem_name, nobjs, nvars, nsamples, run, is_plot, plot_init)
        #print(objs_evaluated)
        np_a = np.array(objs_evaluated._data.tolist())
        np_a = np_a.reshape((nobjs,size_pop))
        np_a = np.transpose(np_a)
    return np_a



def evaluate_run(init_folder,
                path_to_file,
                problem_testbench, 
                problem_name, 
                nobjs, 
                nvars, 
                sampling, 
                nsamples, 
                approaches,
                run):
    #path_to_file = path_to_file + '/Run_' + str(run)
    run = run.tolist()
    data = pickle.load(open(path_to_file, "rb"))
    population = data['individuals_solutions']
    data_evaluted = {}
    data_evaluted['obj_solutions_evaluated']=evaluate_population(population,
                                                                init_folder,
                                                                approaches,
                                                                problem_testbench, 
                                                                problem_name, 
                                                                nobjs, 
                                                                nvars, 
                                                                sampling, 
                                                                nsamples,
                                                                run)
    #print(data_evaluted)
    outfile = open(path_to_file + '_evaluated', 'wb')
    pickle.dump(data_evaluted, outfile)
    outfile.close()