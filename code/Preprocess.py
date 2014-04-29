import numpy as np

from sklearn import preprocessing as prep


class Preprocess(object):
    """Class for preprocessing the challenge data
    """
    def __init__(self, strategy_mv='mean'):
        self.strategy_mv = strategy_mv

    def fit_transform(self, X):
        import datetime
        import time
        import sys

        encs = {}
        feat = []

        n = X.shape[-1]
        sys.stdout.write('Preprocessing: {:7.2%}'.format(0))
        for i in range(n):
            sys.stdout.write('\b'*7 + '{:7.2%}'.format(i*1./n))
            sys.stdout.flush()
            try:
                c = X[:, i].astype(float).reshape((-1, 1))
            except ValueError:
                try:
                    c = [time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple())
                         for s in X[:, i]]
                except ValueError:
                    if len(np.unique(X[:, i])) > 100:
                        X[:, i] = [w[:3] for w in X[:, i]]
                    lab_enc = prep.LabelEncoder()
                    lab_bin = prep.LabelBinarizer()
                    c = lab_enc.fit_transform(X[:, i])
                    c = lab_bin.fit_transform(c)
                    encs[i] = (lab_enc, lab_bin)
            feat.append(c)
        sys.stdout.write('\b'*6+'100%   \n')

        self.encs = encs

        feat = np.concatenate(feat, axis=1)
        imp = prep.Imputer(missing_values='NaN', strategy=self.strategy_mv, axis=0)
        self.feat = imp.fit_transform(feat)

        return self.feat
