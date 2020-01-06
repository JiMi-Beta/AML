from hyperopt import hp, fmin, tpe, STATUS_OK, Trials 
from sklearn import datasets
from sklearn.preprocessing import normalize, scale
from sklearn.naive_bayes import MultinomialNB as nb
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
            data = it[1:]
            label = it[0]
            X.append(data)
            y.append(label)
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='float32')
    return X, y

# X, y = load_data('../../dataset/CNAE-9.data.txt')
X, y = load_data('/data/s2434792/amlData/hyperopt_dataset/CNAE-9.data.txt')

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
    clf = nb(**params)
    return cross_val_score(clf, X_, y, cv=5).mean()

space4nb = {
    'alpha': hp.uniform('alpha', 0, 5),
    'fit_prior': hp.choice('fit_prior', [True, False])
    # 'scale': hp.choice('scale', [0, 1]),
    # 'normalize': hp.choice('normalize', [0, 1])
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

t1 = time.process_time()
trials = Trials()
best = fmin(f, space4nb, algo=tpe.suggest, max_evals=500, trials=trials)
t2 = time.process_time()
print('BEST:')
nbbest = best

def convert_dic(nbbest):
    if("fit_prior" in nbbest):
        if(nbbest["fit_prior"]>0):
            nbbest["fit_prior"] = False
        else:
            nbbest["fit_prior"] = True
    return nbbest

nbbest = convert_dic(nbbest)
print(nbbest)
# Print best CV score
scores = [-trial['result']['loss'] for trial in trials.trials]
print("BEST CV SCORE: " + str(np.max(scores)))
print('TIME COST: %s'%(t2-t1))

