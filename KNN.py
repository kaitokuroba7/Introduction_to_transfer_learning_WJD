#!/usr/bin/env python
# encoding: utf-8

"""
@author: J.Zhang
@contact: 1027380683@qq.com
@site: https://github.com/kaitokuroba7
@software: PyCharm
@file: KNN.py
@time: 2021/10/30 11:01
"""
from scipy import io
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy import stats

from Trans_Class.KMM import KMM


def load_data(folder, domain):
    data = io.loadmat(os.path.join(folder, domain + '_fc6.mat'))
    return data['fts'], data["labels"]


def knn_classify(Xs, Ys, Xt, Yt, k=1):
    model = KNeighborsClassifier(n_neighbors=k)
    Ys = Ys.ravel()  # 将数组拉伸
    Yt = Yt.ravel()
    model.fit(Xs, Ys)
    Yt_pred = model.predict(Xt)
    acc = accuracy_score(Yt, Yt_pred)
    print("Accuracy using kNN: {:.2f}%".format(acc * 100))


if __name__ == "__main__":
    this_folder = r'.\\office31-decaf'
    src_domain = 'amazon'
    tar_domain = 'webcam'
    KMM_ins = KMM()

    Xs, Ys = load_data(folder=this_folder, domain=src_domain)

    Xt, Yt = load_data(folder=this_folder, domain=tar_domain)

    beta = KMM_ins.fit(Xs=Xs, Xt=Xt)

    Xs = beta*Xs

    # Xs = stats.zscore(Xs)
    # Xt = stats.zscore(Xt)
    print("Source:", src_domain, Xs.shape, Ys.shape)
    print("Target:", tar_domain, Xt.shape, Yt.shape)

    knn_classify(Xs, Ys, Xt, Yt)
