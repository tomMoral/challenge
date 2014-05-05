import numpy as np
#import pandas as pd
import csv


def parse(val):
    if val == '':
        return np.nan
    if val == '(MISSING)':
        return np.nan
    try:
        return float(val)
    except ValueError:
        return val


def remove_col(tab, k):
    n, n_f = tab.shape
    if k < 0:
        k = n_f + k
    ind = np.arange(n_f)
    ind = ind[ind != k]
    return tab[:, ind]


def load(N, tst=None):
    fname = '../data/brevets_train.csv'
    if tst is not None:
        fname = '../data/brevets_test.csv'
    with open(fname) as f:
        reader = csv.reader(f, delimiter=';')
        i = 0
        rows = []
        for row in reader:
            if i != 0:
                p_row = [parse(c) for c in row]
                rows.append(p_row)
            i += 1
            if i > N and N > 0:
                break
        rows = np.array(rows)

        if tst is None:
            y = (rows[:, -5] == 'GRANTED').astype(int)
            X = remove_col(rows, -5)
        else:
            y = []
            X = rows

        return X, y


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--CV', action='store_true',
                   help='Perform a cross validation on the model')
    parser.add_argument('--val', action='store_true',
                   help='Keep a validation set')
    args = parser.parse_args()


    if args.val:
	N_trn = 150000
    	N_val = 10000
    else:
        N_trn = -1
        N_val = 0
    
    N_tst = -1

    print 'Load training test'
    X, y = load(N_trn+N_val)
    if N_trn == -1 or X.shape[0] < N_trn+N_val:
	N_trn = X.shape[0]

    print 'Load test set'
    X_tst, _ = load(N_tst, tst=True)
    N_tst = X_tst.shape[0]

    print 'Preprocess the data'
    from Preprocess import Preprocess
    prep_model = Preprocess(strategy_mv='median')
    X = prep_model.fit_transform(np.concatenate([X, X_tst]))

    # Get back the training, validation and test set
    X_trn = X[:N_trn]
    y_trn = y[:N_trn]

    X_val = X[N_trn:N_trn+N_val]
    y_val = y[N_trn:]
    X_tst = X[-N_tst:]

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_jobs=6)

    if args.CV:
        parameters = {'n_estimators': [150, 175, 200],
                      'oob_score': [True, False]}

        from sklearn import grid_search
        clf = grid_search.GridSearchCV(model, parameters,
                                       cv=4, verbose=10,
                                       n_jobs=1)
        print 'Grid Search for the model'
        clf.fit(X_trn, y_trn)
        print clf.best_params_

        model.n_estimators = clf.best_params_['n_estimators']
        model.oob_score = clf.best_estimator_['oob_score']

    else:
        model.n_estimators = 1000
        model.oob_score = False
	model.max_depth = 20
	model.n_jobs = 15

    print 'Fit the model'
    model.fit(X_trn, y_trn)

    if args.val:
    	print 'Evale the model'
    	p_val = model.predict_proba(X_val)
    	print 'Precision: ', 1-sum(abs(p_val[:, 1] - y_val))*1./N_val
    	from sklearn.metrics import roc_auc_score
    	print 'ROC: ', roc_auc_score(y_val, p_val[:, 1])

    print 'Predict the test set proba'
    p = model.predict_proba(X_tst)
    with open('out.txt', 'w') as f:
        for pr in p:
            f.write(str(pr[1])+'\n')

