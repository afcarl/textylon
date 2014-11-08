
'''
Created on Nov 13, 2013 by Afshin Rahimi 
based on http://scikit-learn.org/stable/auto_examples/document_classification_20newsgroups.html

This program is a sample text classifier which is used for the purpose of sentiment analysis.
The input is a directory structure which contains a train and a test folder called the same respectively.
The content of train and test folders are folders themselves. The name of these folders corresponds to
the names of categories. In Sentiment Analysis case these categories are positive, negative and optionally neutral.

This program uses a few implemented classifiers in scikit-learn package and at the end benchmarks the training time,
the test time and the F-Score.

In order to use it with your own data, partition your corpus into train and test folders. put classes/categories
in folders with the same name.
for example my directory structure is like this:
        movie_revivew
            train
                pos
                neg
            test
                pos
                neg


This program supports multiple classes along with two pos/neg classes described here. you should only obey the required
directory structure.

this program is completely based on http://scikit-learn.org/stable/auto_examples/document_classification_20newsgroups.html
The only change is that it has been changed to support your own data and also has been a little simplified for
beginners like myself. The original implementation is authored by 

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause


Installation Notes:
You should install psikit-learn, matplotlib, numpy and setuptools.

@author: Afshin Rahimi
'''
import matplotlib
import os
from sklearn import cross_validation
from sklearn import metrics
from  sklearn.datasets import load_files
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from sklearn.neighbors import NearestNeighbors
from time import time
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pylab as pl


# matplotlib.use('Agg')
# DATA_FOLDER = "/home/af/Downloads/20news-18828"
DATA_FOLDER = "/home/af/Downloads/review_polarity/txt_sentoken"
K_FOLD = 10
data_target = load_files(DATA_FOLDER, encoding='latin1')
DO_PCA = True
Reduction_D = 2
DO_SVD = False
categories = data_target.target_names
DO_NMF = False

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

data_size_mb = size_mb(data_target.data)


print("%d documents - %0.3fMB (all data set)" % (
    len(data_target.data), data_size_mb))

print("%d categories" % len(categories))
print()

# split a training set and a test set
target = data_target.target



print("Extracting features from all the dataset using a sparse vectorizer")
t0 = time()
vectorizer = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, min_df=2, max_df=1.0, ngram_range=(1, 1))

# vectorizer = CountVectorizer(min_df=2, max_df=1.0, ngram_range=(1, 4))
# the output of the fit_transform (x_train) is a sparse csc matrix.
data = vectorizer.fit_transform(data_target.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_size_mb / duration))
print("n_samples: %d, n_features: %d" % data.shape)
print()
'''
print("Extracting best features by a chi-squared test")
ch2NumFeatures = 2000 
ch2 = SelectKBest(chi2, k=ch2NumFeatures)
#print vectorizer.get_stop_words()
data = ch2.fit_transform(data, target)
#print data
'''
if DO_PCA:
    print("dimension reduction pca with d=%d" % Reduction_D)
    pca = PCA(n_components=Reduction_D, copy=True, whiten=False)
    print type(data)
    data = pca.fit_transform(data.toarray())
if DO_SVD:
    print("dimension reduction svd with d=%d" % Reduction_D)
    svd = TruncatedSVD(n_components=Reduction_D, algorithm="randomized", n_iterations=5, random_state=None, tol=0)
    data = svd.fit_transform(data)
if DO_NMF:
    print("dimension reduction nmf with d=%d" % Reduction_D)
    nmf = NMF(n_components=Reduction_D)
    data = nmf.fit_transform(data)



# data = data.todense()
# print type(data)
x = data[:, 0]
y = data[:, 1]
z = np.array(data_target.target)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.set_title("Positive vs Negative", fontsize=14)
ax.set_xlabel("XAVG", fontsize=12)
ax.set_ylabel("YAVG", fontsize=12)
ax.grid(True, linestyle='-', color='0.75')
# scatter with colormap mapping to z value
ax.scatter(x, y, s=60, c=z, marker='o', cmap=cm.get_cmap("jet"));

plt.show()

