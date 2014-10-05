'''
Created on 16 Sep 2014

@author: af
'''

#GPy.models_modules.gp_multioutput_regression

import pylab as pb
pb.ion()
from GPy.core.gp import GP
import csv
from GPy.examples import regression


import numpy as np
import GPy
from GPy import kern, likelihoods
from GPy.models_modules.gp_regression import GPRegression
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer

def loadGeolocationDataset():
    records = readGeoTextRecords()
    print records
    
    vectorizer = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, min_df=5, max_df=50000, ngram_range=(1, 1))
    tvecs = vectorizer.fit_transform(texts)
    print("n_samples: %d, n_features: %d" % tvecs.shape)
    print len(lats)

def localize(max_iters=100, kernel=None, optimize=True, plot=True):
    """Predict the location of a robot given wirelss signal strength readings."""
    data = GPy.util.datasets.robot_wireless()
    #print data
    # create simple GP Model
    m = GPy.models.GPRegression(data['Y'], data['X'], kernel=kernel)

    # optimize
    if optimize:
        m.optimize(messages=False, max_iters=max_iters)

    Xpredict = m.predict(data['Ytest'])[0]
    if plot:
        pb.plot(data['Xtest'][:, 0], data['Xtest'][:, 1], 'r-')
        pb.plot(Xpredict[:, 0], Xpredict[:, 1], 'b-')
        pb.axis('equal')
        pb.title('WiFi Localization with Gaussian Processes')
        pb.legend(('True Location', 'Predicted Location'))
        

    sse = ((data['Xtest'] - Xpredict)**2).sum()

    print m
    print('Sum of squares error on test data: ' + str(sse))
    fig = pb.figure(None)
    pb.title('')
    raw_input()
    return m

#loatGeolocationDataset()
localize()
