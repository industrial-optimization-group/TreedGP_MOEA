# TreedGP for Solving Offline MOPs

### Clone repository
`gh repo clone industrial-optimization-group/TreedGP_MOEA`

### Requirements:
* Python 3.7 or up                                          

### Installation process for normal users:
* Create a new virtual enviroment for the project
* Run the following to initialize the virtual environment: 
`pip3 install virtualenv`
`virtualenv venv`
`source venv/bin/activate`
* Install poetry: `pip3 install poetry` and `poetry init`
* Add packages: `poetry add  numpy pandas matplotlib plotly sklearn scipy pygmo optproblems tqdm diversipy pyDOE joblib jupyter statsmodels GPy graphviz paramz plotly_express pymoo seaborn`

### Example
An example notebook can be found in the example folder solving an offline bi-objective DTLZ2 problem, 10 decision variables with 2000 samples (LHS).

### Running the experiments
* For replicating the experiments in the paper run the `treedGP/StartAll_HPC.py`
* The initial dataset in `.mat` Matlab format in `data_template/initial_samples` folder. The test results are saved as pickle files in  `data_template/test_runs`. Change the folder locations and datset for different problem instances.
* The DBMOPP problems are implemented in Matlab and are available in `matlab_files` folder.

