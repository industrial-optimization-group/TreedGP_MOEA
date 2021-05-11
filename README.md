# TreedGP_MOEA

### Clone repository
`gh repo clone industrial-optimization-group/TreedGP_MOEA`

### Requirements:
* Python 3.7 or up
* [Poetry dependency manager](https://github.com/sdispater/poetry): Only for developers

### Installation process for normal users:
* Create a new virtual enviroment for the project
* Run the following to initialize the virtual environment: 
`pip3 install virtualenv`
`virtualenv venv`
`source venv/bin/activate`
* Install poetry: `pip3 install poetry` and `poetry init`
* Add packages: `poetry add  numpy pandas matplotlib plotly sklearn scipy pygmo optproblems tqdm diversipy pyDOE joblib jupyter statsmodels GPy graphviz paramz plotly_express pymoo seaborn`

### Running the experiments
* For replicating the experiments in the paper run the treedGP/Start
