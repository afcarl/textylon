'''
Created on Jan 28, 2014

@author: af
'''
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
from time import time

import numpy as np
import pylab as pl
from stavicta.util.dsm import *
from stavicta.util.dsm_classes import Matrix, Vocabulary
from stavicta.vectorizer.dsmvectorizer import DSMVectorizer


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
dsm(infile, 2, 0, 1000000, 0, 1000000)

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
vectorizer = DSMVectorizer(None)
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
# the output of the fit_transform (x_train) is a sparse csc matrix.
X_train = vectorizer.fit_transform(data_train.data, CoocMat, IndVoc)
