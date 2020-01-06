from hyperopt import hp, fmin, tpe, STATUS_OK, Trials 
from sklearn import datasets
from sklearn.preprocessing import normalize, scale
from sklearn.linear_model import SGDClassifier as sgd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        X = []
        y = []
        for line in f.readlines():
            line = line.strip('\n')
            it = line.split(',')
            if it[0] == 'B':
                it[0] = 0
            elif it[0] == 'R':
                it[0] = 1
            else:
                it[0] = 2
            data = it[1:]
            label = it[0]
            X.append(data)
            y.append(label)
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='float32')
    return X, y

# X, y = load_data('../../dataset/balance-scale/balance-scale.data.txt')
X, y = load_data('/data/s2434792/amlData/hyperopt_dataset/balance-scale.data.txt')

def hyperopt_train_test(params):
    X_ = X[:]
    """
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
            del params['normalize']
        else:
            del params['normalize']
    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
            del params['scale']
        else:
            del params['scale']
    """
    clf = sgd(**params)
    return cross_val_score(clf, X_, y, cv=5).mean()

space4sgd = {
    'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet', 'none']),
    'alpha': hp.uniform('alpha', 0.00001, 0.1),
    'learning_rate': hp.choice('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
    'eta0': hp.uniform('eta0', 0.00001, 0.1),
    'power_t': hp.uniform('power_t', 0.3, 0.7),
    'warm_start': hp.choice('warm_start', [True, False]),
    'l1_ratio': hp.uniform('l1_ratio', 0, 1),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    'max_iter': hp.uniform('max_iter', 10, 1000),
    'tol': hp.uniform('tol', 0.00001, 0.1),
    'early_stopping': hp.choice('early_stopping', [True, False]),
    'validation_fraction': hp.uniform('validation_fraction', 0.1, 0.9)
    # 'scale': hp.choice('scale', [0, 1]),
    # 'normalize': hp.choice('normalize', [0, 1])
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

t1 = time.process_time()
trials = Trials()
best = fmin(f, space4sgd, algo=tpe.suggest, max_evals=500, trials=trials)
t2 = time.process_time()
print('BEST:')
sgdbest = best

def convert_dic(sgdbest):
    if("learning_rate" in sgdbest):
        assert sgdbest["learning_rate"] < 4
        if(sgdbest["learning_rate"]==0):
            sgdbest["learning_rate"]="constant"
        elif(sgdbest["learning_rate"]==1):
            sgdbest["learning_rate"]="optimal"
        elif(sgdbest["learning_rate"]==2):
            sgdbest["learning_rate"]="invscaling"
        elif(sgdbest["learning_rate"]==3):
            sgdbest["learning_rate"]="adaptive"
    if("penalty" in sgdbest):
        assert sgdbest["penalty"] < 4
        if(sgdbest["penalty"]==0):
            sgdbest["penalty"]="l1"
        elif(sgdbest["penalty"]==1):
            sgdbest["penalty"]="l2"
        elif (sgdbest["penalty"] == 2):
            sgdbest["penalty"] = "elasticnet"
        elif (sgdbest["penalty"] == 3):
            sgdbest["penalty"] = "none"
    if("warm_start" in sgdbest):
        if(sgdbest["warm_start"]>0):
            sgdbest["warm_start"] = False
        else:
            sgdbest["warm_start"] = True
    if ("fit_intercept" in sgdbest):
        if (sgdbest["fit_intercept"] > 0):
            sgdbest["fit_intercept"] = False
        else:
            sgdbest["fit_intercept"] = True
    if ("early_stopping" in sgdbest):
        if (sgdbest["early_stopping"] > 0):
            sgdbest["early_stopping"] = False
        else:
            sgdbest["early_stopping"] = True
    return sgdbest

sgdbest = convert_dic(sgdbest)
print(sgdbest)
# Print best CV score
scores = [-trial['result']['loss'] for trial in trials.trials]
print("BEST CV SCORE: " + str(np.max(scores)))
print('TIME COST: %s'%(t2-t1))

