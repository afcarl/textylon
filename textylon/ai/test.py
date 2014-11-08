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

def median(mylist):
    sorts = sorted(mylist)
    length = len(sorts)
    print length
    if not length % 2:
        return (sorts[length / 2] + sorts[length / 2 - 1]) / 2.0
    return sorts[length / 2]

a = [1, 1, 1, 2, 2, 2, 3, 2.5, 1.5, 1.5, 1.5, 1.5]
print median(a)
