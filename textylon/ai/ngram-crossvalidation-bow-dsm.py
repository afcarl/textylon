
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
from scipy.sparse import hstack
from sklearn import cross_validation
from sklearn import metrics
from  sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
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
from time import time

import numpy as np
import pylab as pl
from stavicta.util.dsm import DSM, DSMDocument
from stavicta.util.externalvectors import loadExternalRepresentation
from stavicta.vectorizer.dsmvectorizer import DSMVectorizer


matplotlib.use('Agg')



# DATA_FOLDER = "/mnt/data/afshin/ngram-sentence-svd/data"
DATA_FOLDER = "/home/af/Downloads/review_polarity/data"
K_FOLD = 10
data_target = load_files(DATA_FOLDER, encoding='latin1')
DSM_MODEL = "wd"
# DSM_MODEL = "ww"
DO_PCA = False
Reduction_D = 500
DO_SVD = False
DO_NMF = False
ADD_BOW = True
DSMFile = 'allsents.txt'
# DSMFile = '/home/af/Downloads/rt-polaritydata/rt-polaritydata/allsents.txt'
categories = data_target.target_names

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

data_size_mb = size_mb(data_target.data)


print("%d documents - %0.3fMB (all data set)" % (
    len(data_target.data), data_size_mb))

print("%d categories" % len(categories))
print()

# split a training set and a test set
target = data_target.target


print("extracting dsm model")
if DSM_MODEL == "ww":
    dsm = DSM()
    
    # infile = '/home/af/Downloads/rt-polaritydata/rt-polaritydata/allsents.txt'
    dsm.count_freqs(DSMFile)
    dsm.dsm(DSMFile)
    print "PPMI weighting...."
    dsm.CoocMat.ppmiTransform()
    print "PPMI weighting finished."
elif DSM_MODEL == "wd":
    dsm = DSMDocument()
    dsm.count_freqs(DSMFile)
    dsm.dsm(DSMFile)
    print "TFIDF weighting...."
    dsm.tfidf()
    print "TFIDF weighting finished."    


# transformer = TfidfTransformer()
# doc_centroids = transformer.fit_transform(doc_centroids)

print("building dsm centroid for each document")
dsmvectorer = DSMVectorizer()
t0 = time()
doc_centroids = dsmvectorer.fit_transform(data_target.data, dsm.CoocMat, dsm.IndVoc)
print "type of centroids is " + str(type(doc_centroids))
print "shape of centroids is " + str(doc_centroids.shape)

# print("Extracting features from all the dataset using a sparse vectorizer")
# t0 = time()
if ADD_BOW:
    vectorizer = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, min_df=2, max_df=1.0, ngram_range=(1, 4))
    data = vectorizer.fit_transform(data_target.data)
    print "type of data is " + str(type(data))
    print "shape of data is " + str(data.shape)
    print "hstacking bag of words features with dsm centroid features..."
    data = hstack([data, doc_centroids])
    print "hstacking bag of words with dsm centroid features finished."
    print type(data)
    print data.shape
else:
    data = doc_centroids

# vectorizer = CountVectorizer(min_df=2, max_df=1.0, ngram_range=(1, 4))
# the output of the fit_transform (x_train) is a sparse csc matrix.






duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_size_mb / duration))
print()

if DO_PCA:
    print("dimension reduction pca with d=%d" % Reduction_D)
    pca = PCA(n_components=Reduction_D, copy=True, whiten=False)
    print type(data)
    data = pca.fit_transform(data.toarray())
elif DO_SVD:
    print("dimension reduction svd with d=%d" % Reduction_D)
    svd = TruncatedSVD(n_components=Reduction_D, algorithm="randomized", n_iterations=5, random_state=None, tol=0)
    data = svd.fit_transform(data)
elif DO_NMF:
    print("dimension reduction nmf with d=%d" % Reduction_D)
    nmf = NMF(n_components=Reduction_D)
    data = nmf.fit_transform(data)
    
print ("%d fold cross validation on data..." % K_FOLD)

'''
print("Extracting %d best features by a chi-squared test" %0)
t0 = time()
ch2NumFeatures = 400000
ch2 = SelectKBest(chi2, k=ch2NumFeatures)
#print vectorizer.get_stop_words()
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)
print("done in %fs" % (time() - t0))
print()
feature_names = np.asarray(vectorizer.get_feature_names())
'''



def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    scores = cross_validation.cross_val_score(clf, data, target, cv=K_FOLD, scoring='f1', verbose=0)
    score = scores.mean()
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    # print scores
    print("f1-score:   %0.3f" % score)


      

    
  

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, 0


results = []
for clf, name in (
       # (RidgeClassifier(tol=1e-2, solver="auto", normalize=True, max_iter=None), "Ridge Classifier"),
       # (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))


class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

print('=' * 80)
print("LinearSVC with L1-based feature selection")
results.append(benchmark(L1LinearSVC()))


# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)


pl.figure(figsize=(12, 8))
pl.title("Score")
pl.barh(indices, score, .2, label="score", color='r')
pl.barh(indices + .3, training_time, .2, label="training time", color='g')
# pl.barh(indices + .6, test_time, .2, label="test time", color='b')
pl.yticks(())
pl.legend(loc='best')
pl.subplots_adjust(left=.25)
pl.subplots_adjust(top=.95)
pl.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    pl.text(-.3, i, c)
pl.savefig('ngramoptimize.png')
pl.show()
