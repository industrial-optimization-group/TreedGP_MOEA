import numpy as np
import pandas as pd
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError
from sklearn import tree
import GPy
import graphviz
import datetime
class treeGP(BaseRegressor):
    def __init__(self, min_samples_leaf):
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.regr = None
        self.dict_gps = {}
        self.model_htgp = None
        self.error_leaves = None
        self.total_point = 0
        self.total_point_gps = 0
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = y
        self.X = X
        self.y = y
        self.regr = tree.DecisionTreeRegressor(max_depth=100, min_samples_leaf=self.min_samples_leaf)
        self.regr = self.regr.fit(X, y)
        n_nodes = self.regr.tree_.node_count
        children_left = self.regr.tree_.children_left
        children_right = self.regr.tree_.children_right
        feature = self.regr.tree_.feature
        threshold = self.regr.tree_.threshold
        rmse = self.regr.tree_.impurity
        rmse_threshold = 1
        samples_leaf_nodes = self.regr.apply(X)
        self.samples_leaf_nodes = samples_leaf_nodes
        error_leaves = None


    # add GPs to leaf nodes with max MSE
    def addGPs(self, X_solutions):
        Y_solution_leaf = self.regr.apply(X_solutions)  
        unique_solutions, count_solutions = np.unique(Y_solution_leaf, return_counts=True)
        unique_solutions = np.setdiff1d(unique_solutions,self.error_leaves)
        self.total_point_gps = unique_solutions.size

        if unique_solutions.size > 0:            
            # Taking max MSE
            mse_solutions = self.regr.tree_.impurity[unique_solutions]
            arg_max_mse = np.argmax(mse_solutions)
            if self.error_leaves is None:
                self.error_leaves = [unique_solutions[arg_max_mse]]
            else:
                self.error_leaves = np.append(self.error_leaves,unique_solutions[arg_max_mse])
            loc_leaf = np.where(self.samples_leaf_nodes==unique_solutions[arg_max_mse])[0]
            X_leaf = self.X[loc_leaf]
            Y_leaf = self.y[loc_leaf]
            self.total_point += np.shape(X_leaf)[0]
            kernel = GPy.kern.Matern52(np.shape(X_leaf)[1],ARD=True)
            m = GPy.models.GPRegression(X_leaf,Y_leaf.reshape(-1, 1),kernel=kernel)
            m.optimize('bfgs')
            self.dict_gps[str(unique_solutions[arg_max_mse])] = m

    def predict(self, X):
        Y_predict = self.regr.predict(X=X)
        Y_test_leaf = self.regr.apply(X)
        unique_solutions, count_solutions = np.unique(Y_test_leaf, return_counts=True)
        Y_predict_mod = Y_predict
        count=0
        if self.error_leaves is not None:            
            for i in range(np.shape(X)[0]):
                if Y_test_leaf[i] in self.error_leaves:                   
                    Y_predict_mod[i] = self.dict_gps[str(Y_test_leaf[i])].predict(X[i].reshape(1,-1))[0][0]
                    count += 1  
        y_mean = Y_predict_mod
        y_stdev = None
        return (y_mean, y_stdev)


