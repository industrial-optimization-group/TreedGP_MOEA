# TGP-MO: Treed Gaussian Processes for Solving Offline Data-Driven Multiobjective Optimization Problems

TGP-MO are tailored surrogate models for solving offline data-driven multiobjective optimization problems. These surrogates are capable of handling large datasets and are computationally inexpensive to build.  

### Data and Results
* Detailed results of the experiments can be found in the `Results` folder or the link provided below.
* The dataset used for the tests are provided in https://drive.google.com/drive/folders/1JzcKdpc9_RhfAqd2ld88RczBUVGTebDc?usp=sharing
### Clone Repository
`gh repo clone industrial-optimization-group/TreedGP_MOEA`

### Requirements:
* Python 3.7 or up                                          

### Installation Process:
* Create a new virtual enviroment for the project
* Run the following to initialize the virtual environment: 
`pip3 install virtualenv`
`virtualenv venv`
`source venv/bin/activate`
* Install poetry: `pip3 install poetry` and `poetry init`
* Add packages: `poetry add  numpy pandas matplotlib plotly sklearn scipy pygmo optproblems tqdm diversipy pyDOE joblib jupyter statsmodels GPy graphviz paramz plotly_express pymoo seaborn`

### Example
An example notebook can be found in the example folder solving an offline bi-objective DTLZ2 problem, 10 decision variables with 2000 samples (LHS).

## Replicating the Experiments

### Offline Dataset
* For generating the dataset we used the matlab codes in `matlab_files/Initial_sampling.m` and `matlab_files/generate_dataset.m`. The dataset is in `.mat` format and necessary changes should be made before using other format files. The initial dataset with 2000 samples can be found in `data_template/initial_samples` folder.
The dataset used for the tests consisting of 2000, 10000 and 50000 samples are provided in https://drive.google.com/drive/folders/1JzcKdpc9_RhfAqd2ld88RczBUVGTebDc?usp=sharing.
* The DBMOPP problems are implemented in Matlab and are available in `matlab_files` folder. 
### Running the experiments
* For replicating the experiments in the paper run the `main_project_files/StartAll_HPC.py`
* The test results will be saved as pickle files in the `data_folder` provided. Change the folder locations and datset for different problem instances.
* Change the parameters `sample_sizes`, `sampling` (sampling strategy), `objectives` (number of objectives), `dims` (number of decision variables), `problem_testbench`, `problems`. For running optimization using full GP, sparse GP or TGP-MO surrogates, set `approaches = ["generic_fullgp","generic_sparsegp","htgp"]` respectively. Kindly note that TGP-MO is referred as `htgp` in the codes.
* Change the `nruns` parameter for the number of runs for each instance you desire to run. If you want to run parallel processes set `parallel_jobs` for the pool size.


### Evaluting the solutions with underlying objectives
* Set `evaluate_data=True` in the `main_project_files/StartAll_HPC.py` file. The DBMOPP problems are evaluted using a matlab bridge.
* Provide the data location and other parameter settings are same as described above.
* The evaluted data are stored in the `data_folder` in `.csv` format.

### Analyzing the Results
* Run the `main_project_files/Analysis_HV_RMSE_T.py` file with respective data location and other parameter settings are same as described above. You will be provided with tables in `.csv` format comparing different approaches. The tables consisit of the pairwise p-values, scores and other details about the approaches being compared.
* Run the `main_project_files/Analysis_points_used.py` for generating the plots of the number of samples used for building the leaf node GPs in the TGP-MO surrogates.
* Run the `main_project_files/Analysis_all_grouped_boxplt.py` to generate the grouped boxplots  comparing the differnt approaches. 

