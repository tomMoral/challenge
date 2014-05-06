"""TP Ranking MDI343

Implementation of pairwise ranking using scikit-learn

Authors: Fabian Pedregosa <fabian@fseoane.net>
         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

License : BSD 3
"""

import itertools
import numpy as np

from sklearn import cross_validation
from sklearn.linear_model import SGDClassifier


def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking

    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.

    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.

    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    diff: array
        target difference for each considered samples
    """
    X_new = []
    y_new = []
    diff = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    k = 0
    for (i, j) in comb:
        #if np.abs(y[i, 0] - y[j, 0]) <= 1. or y[i, 1] != y[j, 1]:
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        diff.append(y[i, 0] - y[j, 0])
        y_new.append(np.sign(diff[-1]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
            diff[-1] = - diff[-1]
        k += 1
    return np.asarray(X_new), np.asarray(y_new).ravel(), np.array(diff).ravel()


#class RankSVM(svm.SVC):
class RankSGD(SGDClassifier):
    """Performs pairwise ranking with an underlying LinearSVC model

    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.

    See object :ref:`svm.SVC` for a full description of parameters.
    """
    def __init__(self, C=1.0):
        super(RankSGD, self).__init__(loss='hinge', penalty='l2', shuffle=True,
                                      n_jobs=20, n_iter=100)

    def fit(self, X, y):
        """Fit a pairwise ranking model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        sample_weights : boolean
           whether to consider sample weights in the ranking problem
           (= weighted loss function)

        Returns
        -------
        self
        """
        max_iter = 10
        mbs = 10000
        n_trn = X.shape[0]
        n_batch = n_trn/mbs
        for n_iter in range(max_iter):
            for i in range(n_batch):
                X_trans, y_trans, diff = transform_pairwise(X[i*mbs:(i+1)*mbs],
                                                            y[i*mbs:(i+1)*mbs])
                super(RankSGD, self).partial_fit(X_trans, y_trans)
        return self

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        ord : array, shape (n_samples,)
            Returns an array of integers representing the relative order of
            the rows in X.
        """
        return np.argsort(np.dot(X, self.coef_.T))

    def score(self, X, y):
        """
        Because we transformed into a balanced pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans, diff = transform_pairwise(X, y)
        return np.mean(super(RankSGD, self).predict(X_trans) == y_trans)


if __name__ == '__main__':
    # TODO : Read the data in winequality-red.csv
    import pandas as pd
    df = pd.read_csv('dataset/winequality-red.csv', delimiter=';')
    data = np.array(df)

    # Select a subset of data
    #data = data[:200]  # take a smaller dataset

    # Fit and evaluate the performance on left out data
    X = data[:, :-1]
    y = data[:, -1]

    rank_svm = RankSGD()
    cv_s = cross_validation.cross_val_score(rank_svm, X, y, cv=3, n_jobs=4)
    print 'CV_score : ', np.mean(cv_s)

    rank_svm.fit(X, y)

    df = pd.read_csv('dataset/winequality-white.csv', delimiter=';')
    dataW = np.array(df)
    dataW = dataW[:200]
    Xw = dataW[:, :-1]
    yw = dataW[:, -1]
    print 'Score white : ', rank_svm.score(Xw, yw)
