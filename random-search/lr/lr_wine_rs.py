from hyperopt import hp, fmin, rand, STATUS_OK, Trials 
from sklearn import datasets
from sklearn.preprocessing import normalize, scale
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import string
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
            it = line.split(';')
            data = it[:-1]
            label = it[-1]
            X.append(data)
            y.append(label)
    X.pop(0)
    y.pop(0)
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='float32')
    return X, y

# X, y = load_data('../../dataset/winequality-white.csv')
X, y = load_data('/data/s2434792/amlData/hyperopt_dataset/winequality-white.csv')

def hyperopt_train_test(params):
    X_ = X[:]
    clf = lr(**params)
    if params['penalty'] == 'l2':
        if params['dual'] is True:
            if params['solver'] == 'liblinear':
                if params['multi_class'] == 'multinomial':
                    return 0.001
                else:
                    return cross_val_score(clf, X_, y, cv=5).mean()
            else:
                return 0.001
        else:
            if params['solver'] == 'liblinear' and params['multi_class'] == 'multinomial':
                return 0.001
            else:
                return cross_val_score(clf, X_, y, cv=5).mean()
    elif params['penalty'] == 'l1':
        if params['dual'] is True:
            return 0.001
        else:
            if params['solver'] == 'liblinear':
                if params['multi_class'] == 'multinomial':
                    return 0.001
                else:
                    return cross_val_score(clf, X_, y, cv=5).mean()
            elif params['solver'] == 'saga':
                return cross_val_score(clf, X_, y, cv=5).mean()
            else:
                return 0.001
    elif params['penalty'] == 'elasticnet':
        if params['dual'] is True:
            return 0.001
        else:
            if params['solver'] == 'saga':
                return cross_val_score(clf, X_, y, cv=5).mean()
            else:
                return 0.001
    elif params['penalty'] == 'none':
        if params['dual'] is True:
            return 0.001
        else:
            if params['solver'] == 'liblinear':
                return 0.001
            else:
                return cross_val_score(clf, X_, y, cv=5).mean()
    else:
        return cross_val_score(clf, X_, y, cv=5).mean()

space4lr = {
    'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet', 'none']),
    'dual': hp.choice('dual', [True, False]),
    'tol': hp.uniform('tol', 0.00001, 0.1),
    'C': hp.uniform('C', 0, 5),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    'intercept_scaling': hp.uniform('intercept_scaling', 0, 5),
    'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
    'max_iter': hp.uniform('max_iter', 10, 1000),
    'multi_class': hp.choice('multi_class', ['ovr', 'multinomial', 'auto']),
    'warm_start': hp.choice('warm_start', [True, False]),
    'l1_ratio': hp.uniform('l1_ratio', 0, 1)

    # 'scale': hp.choice('scale', [0, 1]),
    # 'normalize': hp.choice('normalize', [0, 1])
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

t1 = time.process_time()
trials = Trials()
best = fmin(f, space4lr, algo=rand.suggest, max_evals=500, trials=trials)
t2 = time.process_time()
print('BEST:')
lrbest = best

def convert_dic(lrbest):
    if("penalty" in lrbest):
        assert lrbest["penalty"] < 4
        if(lrbest["penalty"]==0):
            lrbest["penalty"]="l1"
        elif(lrbest["penalty"]==1):
            lrbest["penalty"]="l2"
        elif (lrbest["penalty"] == 2):
            lrbest["penalty"] = "elasticnet"
        elif (lrbest["penalty"] == 3):
            lrbest["penalty"] = "none"
    if("warm_start" in lrbest):
        if(lrbest["warm_start"]>0):
            lrbest["warm_start"] = False
        else:
            lrbest["warm_start"] = True
    if ("dual" in lrbest):
        if (lrbest["dual"] > 0):
            lrbest["dual"] = False
        else:
            lrbest["dual"] = True
    if ("fit_intercept" in lrbest):
        if (lrbest["fit_intercept"] > 0):
            lrbest["fit_intercept"] = False
        else:
            lrbest["fit_intercept"] = True
    if("solver" in lrbest):
        assert lrbest["solver"] < 5
        if(lrbest["solver"] == 0):
            lrbest["solver"] = "newton-cg"
        elif(lrbest["solver"] == 1):
            lrbest["solver"] = "lbfgs"
        elif (lrbest["solver"] == 2):
            lrbest["solver"] = "liblinear"
        elif (lrbest["solver"] == 3):
            lrbest["solver"] = "sag"
        elif (lrbest["solver"] == 4):
            lrbest["solver"] = "saga"
    if("multi_class" in lrbest):
        assert lrbest["multi_class"] < 4
        if(lrbest["multi_class"] == 0):
            lrbest["multi_class"] = "ovr"
        elif(lrbest["multi_class"] == 1):
            lrbest["multi_class"] = "multinomial"
        elif (lrbest["multi_class"] == 2):
            lrbest["multi_class"] = "auto"
    return lrbest

lrbest = convert_dic(lrbest)
print(lrbest)
# Print best CV score
scores = [-trial['result']['loss'] for trial in trials.trials]
print("BEST CV SCORE: " + str(np.max(scores)))
print('TIME COST: %s'%(t2-t1))
