from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, rand
from sklearn import datasets
from sklearn.preprocessing import normalize, scale
from sklearn.neighbors import KNeighborsClassifier as knn
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
            it = line.split('\t')
            data = it[0:-1]
            label = it[-1]
            X.append(data)
            y.append(label)
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='float32')
    return X, y

# X, y = load_data('../../dataset/skin/skin.txt')
X, y = load_data('/data/s2434792/amlData/hyperopt_dataset/skin.txt')

def hyperopt_train_test(params):
    X_ = X[:]
    clf = knn(**params)
    return cross_val_score(clf, X_, y, cv=5).mean()

space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(1, 50)),
    'weights': hp.choice('weights', ['uniform', 'distance']),
    'p': hp.uniform('p', 1, 5)
    # 'scale': hp.choice('scale', [0, 1]),
    # 'normalize': hp.choice('normalize', [0, 1])
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

t1 = time.process_time()
trials = Trials()
best = fmin(f, space4knn, algo=rand.suggest, max_evals=500, trials=trials)
t2 = time.process_time()
print('BEST:')
knnbest = best

def convert_dic(knnbest):
    if("weights" in knnbest):
        assert knnbest["weights"] < 2
        if(knnbest["weights"] == 0):
            knnbest["weights"] = "uniform"
        elif(knnbest["weights"] == 1):
            knnbest["weights"] = "distance"
    return knnbest

knnbest = convert_dic(knnbest)
print(knnbest)
# Print best CV score
scores = [-trial['result']['loss'] for trial in trials.trials]
print("BEST CV SCORE: " + str(np.max(scores)))
print('TIME COST: %s'%(t2-t1))