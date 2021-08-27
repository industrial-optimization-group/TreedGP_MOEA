# TGP-MO : Treed Gaussian processes for multiobjective problems

TGP-MO are tailored surrogate models for solving offline data-driven multiobjective optimization problems. These surrogates are capable of handling large datasets and are computationally inexpensive to build.  

Detailed results of the experiments can be found in
https://drive.google.com/drive/folders/1JzcKdpc9_RhfAqd2ld88RczBUVGTebDc?usp=sharing

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

## Replicating the experiments

* For generating the dataset we used the matlab codes in `matlab_files/Initial_sampling.m` and `matlab_files/generate_dataset.m`. The dataset is in `.mat` format and necessary changes should be made before using other format files. The initial dataset with 2000 samples can be found in `data_template/initial_samples` folder.
* The DBMOPP problems are implemented in Matlab and are available in `matlab_files` folder.
### Running the experiments
* For replicating the experiments in the paper run the `main_project_files/StartAll_HPC.py`
* The test results are saved as pickle files in  `data_template/test_runs`. Change the folder locations and datset for different problem instances.


### Evaluting the solutions with underlying objectives
* Set `evaluate_data=True` in the `main_project_files/StartAll_HPC.py` file. The DBMOPP problems are evaluted using a matlab bridge. 

