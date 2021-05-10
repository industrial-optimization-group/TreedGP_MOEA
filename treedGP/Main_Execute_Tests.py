import sys
sys.path.insert(1, '/home/amrzr/Work/Codes/TreedGP_MOEA/')

from desdeo_emo.EAs.RVEA import RVEA
from desdeo_problem.Problem import DataProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs
import numpy as np
import pandas as pd
from desdeo_problem.surrogatemodels.surrogate_fullGP import FullGPRegressor as fgp
from desdeo_problem.surrogatemodels.surrogate_sparseGP import SparseGPRegressor as sgp
#from desdeo_problem.surrogatemodels.treedGP import treedGP as treedGP2
from treedGP_framework import run_treed_GP as treedGP
import time
import scipy.io

init_folder = '/home/amrzr/Work/Codes/data/initial_samples/'
plot_folder = '/home/amrzr/Work/Codes/data/plots_htgp/'
plotting = False
plotting_sols = False

def build_surrogates(problem_testbench, problem_name, nobjs, nvars, nsamples, sampling, is_data, x_data, y_data, surrogate_type, Z=None, z_samples=None):
    total_points_per_model = None
    total_points_per_model_sequence = None
    x_names = [f'x{i}' for i in range(1,nvars+1)]
    y_names = [f'f{i}' for i in range(1,nobjs+1)]
    row_names = ['lower_bound','upper_bound']
    if is_data is False:
        prob = test_problem_builder(problem_name, nvars, nobjs)
        x = lhs(nvars, nsamples)
        y = prob.evaluate(x)
        data = pd.DataFrame(np.hstack((x,y.objectives)), columns=x_names+y_names)
    else:
        data = pd.DataFrame(np.hstack((x_data,y_data)), columns=x_names+y_names)
    if problem_testbench == 'DDMOPP':
        x_low = np.ones(nvars)*-1
        x_high = np.ones(nvars)
    elif problem_testbench == 'DTLZ':
        x_low = np.ones(nvars)*0
        x_high = np.ones(nvars)
    bounds = pd.DataFrame(np.vstack((x_low,x_high)), columns=x_names, index=row_names)
    problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names,bounds=bounds)
    start = time.time()
    if surrogate_type == "generic_fullgp":
        problem.train(fgp)
    elif surrogate_type == "generic_sparsegp":
        problem.train(sgp,  model_parameters=z_samples)
    elif surrogate_type == "htgp":
        problem, total_points_per_model, total_points_per_model_sequence = treedGP(x_data, y_data, x_low, x_high)

    end = time.time()
    time_taken = end - start
    return problem, time_taken, total_points_per_model, total_points_per_model_sequence

def read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run):
    mat = scipy.io.loadmat(init_folder + 'Initial_Population_' + problem_testbench + '_' + sampling +
                        '_AM_' + str(nvars) + '_'+str(nsamples)+'.mat')
    x = ((mat['Initial_Population_'+problem_testbench])[0][run])[0]
    if problem_testbench == 'DDMOPP':
        mat = scipy.io.loadmat(init_folder + 'Obj_vals_DDMOPP_'+sampling+'_AM_'+problem_name+'_'
                                + str(nobjs) + '_' + str(nvars)
                                + '_'+str(nsamples)+'.mat')
        y = ((mat['Obj_vals_DDMOPP'])[0][run])[0]
    elif problem_testbench == 'DTLZ':
        prob = test_problem_builder(
                    name=problem_name, n_of_objectives=nobjs, n_of_variables=nvars
                )
        y = prob.evaluate(x)[0]
    return x, y

def optimize_surrogates(problem, max_iters = 10):
    print("Optimizing...")
    evolver_opt = RVEA(problem, use_surrogates=True, n_iterations=max_iters)
    while evolver_opt.continue_evolution():
        evolver_opt.iterate()
        print("Population size:",np.shape(evolver_opt.population.objectives)[0])
    return evolver_opt.population

def run_optimizer(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, is_data, surrogate_type, run):
    print("Reading Data...")
    if is_data is True:
        x, y = read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run)
    print("Building surrogates...")
    surrogate_problem, time_taken, total_points_per_model, total_points_per_model_sequence = build_surrogates(problem_testbench,problem_name, nobjs, nvars, nsamples, sampling, is_data, x, y, surrogate_type,z_samples={'z_samples':10*nvars})
    print("Time taken to build:",time_taken)
    print("Optimizing surrogates...")
    population = optimize_surrogates(surrogate_problem)
    if surrogate_type == "htgp":
        results_dict = {
                        'individual_archive': population.individuals_archive,
                        'objectives_archive': population.objectives_archive,
                        'uncertainty_archive': population.uncertainty_archive,
                        'individuals_solutions': population.individuals,
                        'obj_solutions': population.objectives,
                        'uncertainty_solutions': population.uncertainity,
                        'time_taken': time_taken,
                        'total_points':  total_points_per_model,
                        'total_points_per_model_sequence': total_points_per_model_sequence
                        }
    else:
        results_dict = {
                'individual_archive': population.individuals_archive,
                'objectives_archive': population.objectives_archive,
                'uncertainty_archive': population.uncertainty_archive,
                'individuals_solutions': population.individuals,
                'obj_solutions': population.objectives,
                'uncertainty_solutions': population.uncertainity,
                'time_taken': time_taken
            }
    return results_dict
