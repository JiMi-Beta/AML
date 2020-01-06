"""
An example for the usage of SMAC within Python.
We optimize a simple SVM on the IRIS-benchmark.

Note: SMAC-documentation uses linenumbers to generate docs from this file.
"""

import logging
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

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
            it = line.split(',')
            data = it[1:]
            label = it[0]
            X.append(data)
            y.append(label)
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='float32')
    return X, y

# X, y = load_data('../../dataset/CNAE-9.data.txt')
X, y = load_data('/data/s2434792/amlData/hyperopt_dataset/CNAE-9.data.txt')

def SGD_from_cfg(cfg):
    """ Creates a SVM based on a configuration and evaluates it on the
    iris-dataset using cross-validation.

    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!

    Returns:
    --------
    A crossvalidated mean score for the svm on the loaded data-set.
    """
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    #cfg = {k : cfg[k] for k in cfg if cfg[k]}
    # We translate boolean values:
    #cfg["warm_start"] = True if cfg["warm_start"] == "True" else False

    clf = SGDClassifier(**cfg, random_state=42)

    scores = cross_val_score(clf, X, y, cv=5)

    return 1-np.mean(scores)  # Minimize!

#logger = logging.getLogger("SVMExample")
logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()

# We define a few possible types of SVM-kernels and add them as "kernel" to our cs
penalty = CategoricalHyperparameter("penalty", ["l1", "l2", "elasticnet", "none"], default_value="l2")

alpha = UniformFloatHyperparameter("alpha", 0.00001, 0.1, default_value=0.0001)

learning_rate = CategoricalHyperparameter("learning_rate", ["constant", "optimal", "invscaling", "adaptive"], default_value="optimal")

eta0 = UniformFloatHyperparameter("eta0", 0.00001, 0.1, default_value=0.0001)

power_t = UniformFloatHyperparameter("power_t", 0.3, 0.7, default_value=0.5)

warm_start = CategoricalHyperparameter("warm_start", [True, False], default_value=False)

l1_ratio = UniformFloatHyperparameter("l1_ratio", 0.0, 1.0, default_value=0.15)

fit_intercept = CategoricalHyperparameter("fit_intercept", [True, False], default_value=False)

max_iter = UniformIntegerHyperparameter("max_iter", 10, 1000, default_value=500)

tol = UniformFloatHyperparameter("tol", 0.00001, 0.1, default_value=0.0001)

early_stopping = CategoricalHyperparameter("early_stopping", [True, False], default_value=False)

validation_fraction = UniformFloatHyperparameter("validation_fraction", 0.1, 0.9, default_value=0.2)

cs.add_hyperparameters([penalty,
                        alpha,
                        learning_rate,
                        eta0,
                        power_t,
                        warm_start,
                        l1_ratio,
                        fit_intercept,
                        max_iter,
                        tol,
                        early_stopping,
                        validation_fraction])

# Others are kernel-specific, so we can add conditions to limit the searchspace

use_eta0 = InCondition(child=eta0, parent=learning_rate, values=["constant", "invscaling", "adaptive"])

use_power_t = InCondition(child=power_t, parent=learning_rate, values=["invscaling"])

use_l1_ratio = InCondition(child=l1_ratio, parent=penalty, values=["elasticnet"])

cs.add_conditions([use_eta0, use_power_t, use_l1_ratio])




# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 500,   # max. number of function evaluations; for this example set to a low number
                     "cs": cs,               # configuration space
                     "deterministic": "true"
                     })

# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_value = SGD_from_cfg(cs.get_default_configuration())
print("Default Value: %.2f" % (def_value))

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
        tae_runner=SGD_from_cfg)

a_time = time.process_time()
incumbent = smac.optimize()
b_time = time.process_time()

print("+++++++++++++++++++++++")
print("Optimization finished. CPU time consumed: %s"%(a_time - b_time))
print("+++++++++++++++++++++++")

inc_value = SGD_from_cfg(incumbent)

print("Optimized Value: %.6f" % (inc_value))

print("before validate time:%s"%(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

# We can also validate our results (though this makes a lot more sense with instances)
smac.validate(config_mode='inc',      # We can choose which configurations to evaluate
              #instance_mode='train+test',  # Defines what instances to validate
              repetitions=100,        # Ignored, unless you set "deterministic" to "false" in line 95
              n_jobs=1)               # How many cores to use in parallel for optimization

print("end time:%s"%(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
