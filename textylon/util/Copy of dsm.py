#################################
# Distributional Semantics
#
# July 2013
#
# Revisions:
# ----------
# October 2013: added class structure
# November 2013: use dictionaries instead of arrays for index and context vocabularies, use Counter for frequencies
#
# Copyright (C) Magnus Sahlgren
#################################

from collections import Counter
import logging
import math
import operator
import os
from sklearn import metrics
from  sklearn.datasets import load_files
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
from time import gmtime, strftime
from time import time

from com.textylon.util.dsm import *
from com.textylon.util.dsm_classes import Matrix, Vocabulary
from com.textylon.vectorizer.dsmvectorizer import DSMVectorizer
from dsm_classes import Vocabulary, Matrix
import numpy as np
import numpy.random as nprnd
import pylab as pl
import scipy.spatial as sp


logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
hdlr = logging.FileHandler('log.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

# collect vocabulary and count frequencies
def count_freqs(infile):
    global Words
    Words = Counter()
    print "Started: " + strftime("%H:%M:%S", gmtime())
    inp = open(infile, "r")
    for line in inp.readlines():
        for wrd in line.split():
            Words[wrd] += 1
    inp.close()
    print "Finished: " + strftime("%H:%M:%S", gmtime())
    print "Token count: " + str(sum(Words.values()))

# count cooccurrence frequencies from infile within win
# frequency threhsolds for both index words and context words
# dsm = distributional semantic model
def dsm(infile, win, index_minf, index_maxf, ctx_minf, ctx_maxf):
    global IndVoc
    IndVoc = Vocabulary()
    global CtxVoc
    CtxVoc = Vocabulary()
    global CoocMat
    CoocMat = Matrix()
    print "Started: " + strftime("%H:%M:%S", gmtime())
    inp = open(infile, "r")
    update_vocabulary(index_minf, index_maxf, ctx_minf, ctx_maxf)
    print IndVoc.getlength(), CtxVoc.getlength()
    CoocMat.makematrix(IndVoc.getlength(), CtxVoc.getlength())
    line_nr = 0
    for line in inp.readlines():
        cnt = 0
        wrdlst = line.split()
        for wrd in wrdlst:
            if IndVoc.lookup(wrd):
            # count co-occurrences to the left
                ctx = 1
                while ctx <= win:
                    if (cnt - ctx) >= 0:
                        c = wrdlst[cnt - ctx]
                        update_counts(c, wrd, ctx_minf, ctx_maxf)
                        ctx += 1
                    else:
                        ctx = win + 1
           # count co-occurrences to the right
                ctx = 1
                while ctx <= win:
                    if (cnt + ctx) < len(wrdlst):
                        c = wrdlst[cnt + ctx]
                        update_counts(c, wrd, ctx_minf, ctx_maxf)
                        ctx += 1
                    else:
                        ctx = win + 1
            cnt += 1
        line_nr += 1
    inp.close()
    print "Finished: " + strftime("%H:%M:%S", gmtime())

# check if the word should be indexed and used as ctx word
def update_vocabulary(index_minf, index_maxf, ctx_minf, ctx_maxf):
    i_cnt = 0
    c_cnt = 0
    for w in Words.most_common():
        q = w[1]
        if (q > index_minf) and (q < index_maxf):
            IndVoc.set(w[0], i_cnt)
            i_cnt += 1
        if (q > ctx_minf) and (q < ctx_maxf):
            CtxVoc.set(w[0], c_cnt)
            c_cnt += 1

# update cooccurrence counts
def update_counts(w, wrd, minf, maxf):
    if CtxVoc.lookup(w):
        CoocMat.update(IndVoc.getindex(wrd), CtxVoc.getindex(w), 1)

# TODO: implement direction-sensitive dsm (aka HAL)

######
# Misc
######

# clean up vocabularies and co-occurrence matrix
def clear_ctx():

    IndVoc.delete()
    CtxVoc.delete()
    CoocMat.delete()

# clean up frequency counters
def clear_freq():
    Words.clear()

############
# Evaluation
############

# toefl test
def toefl(testfile):
    inp = open(testfile, "r")
    corr = 0
    tot = 0
    unknown_target = []
    unknown_answer = []
    incorrect = []
    for line in inp.readlines():
        flag = False
        target, correct, alt2, alt3, alt4 = line.replace("(", "").split()
        if IndVoc.lookup(target):
            targetvec = CoocMat.matrix.getrow(IndVoc.getindex(target)).todense()
            tot += 1
            if IndVoc.lookup(correct):
                correctvec = CoocMat.matrix.getrow(IndVoc.getindex(correct)).todense()
                sim = 1 - sp.distance.cosine(targetvec, correctvec)
                if sim > 0.0:
                    flag = True
                else:
                    incorrect.append(target)
                for i in (alt2, alt3, alt4):
                    if IndVoc.lookup(i):
                        i_vec = CoocMat.matrix.getrow(IndVoc.getindex(i)).todense()
                        i_sim = 1 - sp.distance.cosine(targetvec, i_vec)
                        if i_sim > sim:
                            if not target in incorrect:
                                incorrect.append(target)
                            flag = False
                if flag:
                    corr += 1
            else:
                unknown_answer.append(correct)
        else:
            unknown_target.append(target)
    inp.close()
    print "TOEFL synonym score: " + str(float(corr) / float(tot)) + " (" + str(corr) + "/" + str(tot) + ")"
    print "Incorrect: " + str(incorrect)
    print "Unknown targets: " + str(unknown_target)
    print "Unknown answers: " + str(unknown_answer)
    
    logger.info("TOEFL synonym score: " + str(float(corr) / float(tot)) + " (" + str(corr) + "/" + str(tot) + ")")
    logger.info("Incorrect: " + str(incorrect))
    logger.info("Unknown targets: " + str(unknown_target))
    logger.info("Unknown answers: " + str(unknown_answer))

# toefl test
# A is a numpy matrix
def toefl_mat(testfile, A):
    inp = open(testfile, "r")
    flag = False
    corr = 0
    tot = 0
    unknown_target = []
    unknown_answer = []
    incorrect = []
    for line in inp.readlines():
        target, correct, alt2, alt3, alt4 = line.replace("(", "").split()
        if IndVoc.lookup(target):
            targetvec = A[IndVoc.getindex(target), :]
            tot += 1
            if IndVoc.lookup(correct):
                correctvec = A[IndVoc.getindex(correct), :]
                sim = 1 - sp.distance.cosine(targetvec, correctvec)
                if sim > 0.0:
                    flag = True
                for i in (alt2, alt3, alt4):
                    if IndVoc.lookup(i):
                        i_vec = A[IndVoc.getindex(i), :]
                        i_sim = 1 - sp.distance.cosine(targetvec, i_vec)
                        if i_sim > sim:
                            if not target in incorrect:
                                incorrect.append(target)
                            flag = False
                if flag:
                    corr += 1
            else:
                unknown_answer.append(correct)
        else:
            unknown_target.append(target)
    inp.close()
    print "TOEFL synonym score: " + str(float(corr) / float(tot))
    print "Incorrect: " + str(incorrect)
    print "Unknown targets: " + str(unknown_target)
    print "Unknown answers: " + str(unknown_answer)
    logger.info("TOEFL synonym score: " + str(float(corr) / float(tot)) + " (" + str(corr) + "/" + str(tot) + ")")
    logger.info("Incorrect: " + str(incorrect))
    logger.info("Unknown targets: " + str(unknown_target))
    logger.info("Unknown answers: " + str(unknown_answer))

# find the nr nearest neighbors to word using cosine similarity
# TODO: optimization
def nns(word, nr):
    res = {}
    if IndVoc.lookup(word):
        w_vec = CoocMat.matrix.getrow(IndVoc.getindex(word)).todense()
        for k in IndVoc.hsh:
            k_vec = CoocMat.matrix.getrow(IndVoc.getindex(k)).todense()
            sim = 1 - sp.distance.cosine(w_vec, k_vec)
            if (not math.isnan(sim)) and (not math.isinf(sim)):
                res[k] = sim
    sorted_res = sorted(res.iteritems(), key=lambda(k, v): v, reverse=True)
    print word, sorted_res[0:nr]
    # for r in sorted_res[1:nr]: # 1 to avoid including word
        # print r[0] + ' ' + str(r[1][0][0])
DATA_FOLDER = "/home/af/Downloads/review_polarity/txt_sentoken"
TRAIN_FOLDER = os.path.join(DATA_FOLDER, "train")
TEST_FOLDER = os.path.join(DATA_FOLDER, "test")
infile = os.path.join(DATA_FOLDER, "all.txt")
data_train = load_files(TRAIN_FOLDER, encoding='latin1')
data_test = load_files(TEST_FOLDER, encoding='latin1')

categories = data_train.target_names

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

print("%d documents - %0.3fMB (training set)" % (
    len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test.data), data_test_size_mb))
print("%d categories" % len(categories))
print()

# split a training set and a test set
y_train = data_train.target
y_test = data_test.target
count_freqs(infile)
print "building dsm model..."
dsm(infile, 2, 0, 1000000, 0, 1000000)
print "finished dsm model."
print "starting ppmi transformation"
t0 = time()
CoocMat.ppmiTransform()
duration = time() - t0
print("ppmi transformation finished in %fs." % duration)


print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
vectorizer = DSMVectorizer(None)
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
# the output of the fit_transform (x_train) is a sparse csc matrix.
X_train = vectorizer.fit_transform(data_train.data, CoocMat, IndVoc)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test dataset using the same vectorizer")
t0 = time()
X_test = vectorizer.fit_transform(data_test.data, CoocMat, IndVoc)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

print("Extracting %d best features by a chi-squared test" % 0)
t0 = time()
ch2 = SelectKBest(chi2, k=0)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)
print("done in %fs" % (time() - t0))
print()


# feature_names = np.asarray(vectorizer.get_feature_names())

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
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(y_test, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

      

    
    print("classification report:")
    print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
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
test_time = np.array(test_time) / np.max(test_time)

pl.figure(figsize=(12, 8))
pl.title("Score")
pl.barh(indices, score, .2, label="score", color='r')
pl.barh(indices + .3, training_time, .2, label="training time", color='g')
pl.barh(indices + .6, test_time, .2, label="test time", color='b')
pl.yticks(())
pl.legend(loc='best')
pl.subplots_adjust(left=.25)
pl.subplots_adjust(top=.95)
pl.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    pl.text(-.3, i, c)

pl.show()
