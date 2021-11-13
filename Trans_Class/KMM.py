#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: KMM.py
@time: 2021/10/31 10:46
"""

import numpy as np
import sklearn.metrics
from cvxopt import matrix, solvers


def kernel(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K


class KMM:
    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        """

        :param kernel_type: 'linear' | 'rbf'
        :param gamma:
        :param B:
        :param eps:
        """

        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        """

        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt/Ps) value vector (Beta in this paper)
        """
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps is None:
            self.eps = self.B / np.sqrt(ns)

        K = kernel(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(kernel(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1)

        K = matrix(K.astype(np.double))
        kappa = matrix(kappa.astype(np.double))

        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1+self.eps), ns*(self.eps-1), self.B*np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])

        return beta



if __name__ == "__main__":
    pass
