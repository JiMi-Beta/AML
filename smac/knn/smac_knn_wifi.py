"""
An example for the usage of SMAC within Python.
We optimize a simple SVM on the IRIS-benchmark.

Note: SMAC-documentation uses linenumbers to generate docs from this file.
"""

import logging
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier as knn

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition, AndConjunction

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

from sklearn.linear_model import SGDClassifier
import time, os


print("start time:%s"%(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        X = []
        y = []
        for line in f.readlines():
            line = line.strip('\n')
            it = line.split('\t')
            data = it[:-1]
            label = it[-1]
            X.append(data)
            y.append(label)
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='float32')
    return X, y

# X, y = load_data('../../dataset/wifi_localization-2000.txt')
X, y = load_data('/data/s2434792/amlData/hyperopt_dataset/wifi_localization-2000.txt')

def kNN_from_cfg(params):

    clf = knn(**params)
    
    return 1 - cross_val_score(clf, X, y, cv=5).mean()

#logger = logging.getLogger("SVMExample")
logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()

# We define a few possible types of SVM-kernels and add them as "kernel" to our cs

n_neighbors = UniformIntegerHyperparameter("n_neighbors", 1, 50, default_value=5)

weights = CategoricalHyperparameter("weights", ["uniform", "distance"], default_value="uniform")

p = UniformIntegerHyperparameter("p", 1, 5, default_value=2)

cs.add_hyperparameters([n_neighbors, weights, p])

# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 500,   # max. number of function evaluations; for this example set to a low number
                     "cs": cs,               # configuration space
                     "deterministic": "true"
                     })

# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_value = kNN_from_cfg(cs.get_default_configuration())
print("Default Value: %.2f" % (def_value))

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
        tae_runner=kNN_from_cfg)

a_time = time.process_time()
incumbent = smac.optimize()
b_time = time.process_time()

print("+++++++++++++++++++++++")
print("Optimization finished. CPU time consumed: %s"%(a_time - b_time))
print("+++++++++++++++++++++++")

inc_value = kNN_from_cfg(incumbent)

print("Optimized Value: %.6f" % (inc_value))

print("before validate time:%s"%(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# We can also validate our results (though this makes a lot more sense with instances)
smac.validate(config_mode='inc',      # We can choose which configurations to evaluate
              #instance_mode='train+test',  # Defines what instances to validate
              repetitions=100,        # Ignored, unless you set "deterministic" to "false" in line 95
              n_jobs=1)               # How many cores to use in parallel for optimization

print("end time:%s"%(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
