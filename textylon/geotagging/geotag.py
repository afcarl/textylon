'''
Created on 4 Sep 2014

@author: af
'''
import codecs
import matplotlib as mpl
mpl.use('Agg')
import shutil
import os
import re
import itertools
from scipy import linalg
from scipy.stats import threshold
from sklearn import mixture
import copy
from scipy.io import mmwrite
from math import radians, cos, sin, asin, sqrt
from sklearn import cross_validation
from sklearn import metrics
from  sklearn.datasets import load_files
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, TruncatedSVD, NMF, SparsePCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import linear_model
from sklearn.linear_model import Perceptron
from sklearn.datasets import dump_svmlight_file
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC, SVC
from IPython.core.debugger import Tracer
from sklearn.utils.extmath import density
import scipy.sparse as sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import SelectKBest
import group_lasso 
# from extract import get_tokens
# from time import time
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from os import path
import math
# from datetime import datetime
import glob
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import  matplotlib.collections as collections
import matplotlib.ticker as ticker
import pylab as pb
from theano.tensor.basic import dmatrix
from sklearn.linear_model.coordinate_descent import MultiTaskLasso, ElasticNet
from scipy.spatial.distance import pdist
pb.ion()
from GPy.core.gp import GP
import csv
from GPy.examples import regression
from sklearn.linear_model.sgd_fast import Log
import numpy as np
import GPy
from GPy import kern, likelihoods
# from GPy.models_modules.gp_regression import GPRegression
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from numpy import float16, float32, float64
from scipy.sparse import csr_matrix
import sys
from scipy import mean
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time
import pickle
import numpy

import theano
import theano.tensor as T
DATASET_NUMBER = 2
DATA_HOME = '/home/arahimi/datasets'
DATASETS = ['cmu', 'na', 'world']
ENCODINGS = ['latin1', 'utf-8', 'utf-8']
GEOTEXT_HOME = path.join(DATA_HOME, DATASETS[DATASET_NUMBER-1])
data_encoding = ENCODINGS[DATASET_NUMBER-1]
# GEOTEXT_HOME = '/home/arahimi/Roller Dataset NA'
users_home = path.join(GEOTEXT_HOME, 'processed_data')
testfile = path.join(users_home, 'user_info.test')
devfile = path.join(users_home, 'user_info.dev')
trainfile = path.join(users_home, 'user_info.train')


lngs = []
ltts = []
pointText = {}
keys = []
userFirstTime = {}
userLocation = {}
locationUser = {}
userlat = {}
userlon = {}
testUsers = {}
trainUsers = {}
devUsers = {}
classLatMedian = {}
classLonMedian = {}
classLatMean = {}
classLonMean = {}
devClasses = {}
testClasses = {}
categories = []
mentions = []
testText = {}
devText = {}
trainText = {}

costMatrix = None
trainCostMatrix = None
testCostMarix = None
devCostMatrix = None

X_train = None
X_dev = None
X_test = None
Y_train = None
Y_dev = None
Y_test = None
U_train = None
U_dev = None
U_test = None



        
def median(mylist):
    sorts = sorted(mylist)
    length = len(sorts)
    if not length % 2:
        return (sorts[length / 2] + sorts[length / 2 - 1]) / 2.0
    return sorts[length / 2]

def distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a)) 

    # 6367 km is the radius of the Earth
    km = 6367 * c
    return km 

def users(file, type='train'):
    global testUsers
    global trainUsers
    global devUsers
    global userLocation
    global testText
    global devText
    global trainText
    global locationUser
    
    with codecs.open(file, 'r', encoding=data_encoding) as inf:
        for line in inf:
            # print line
            fields = line.split('\t')
            if len(fields)!=4:
                print fields
            user = fields[0].strip()
            lat = str(float(fields[1])).strip()
            lon = str(float(fields[2])).strip()
            text = fields[3].strip()
            locStr = lat + ',' + lon
            userLocation[user] = locStr
            if type == 'train':
                trainUsers[user] = locStr
                trainText[user] = text
                if locStr in locationUser:
                    # print "Warning: we have multiple users with exactly the same location!"
                    locationUser[locStr] = locationUser[locStr] + " " + user
                else:
                    locationUser[locStr] = user
            elif type == 'test':
                testUsers[user] = locStr
                testText[user] = text
            elif type == 'dev':
                devUsers[user] = locStr
                devText[user] = text
    
    


  


        
def plot_points():
    
    # N = 50
    # x = np.random.rand(N)
    # y = np.random.rand(N)
    x = []
    y = []
    for user in userLocation:
        if userlon[user] > -70:
            continue
        y.append(userlat[user])
        x.append(userlon[user])
    colors = np.random.rand(15)
    
    area = 1  # 0 to 15 point radiuses
    
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.show()

            



def partitionLocView(granularity, partitionMethod):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    filename = users_home + '/' + str(granularity).strip() + '_' + partitionMethod + '_clustered.train'
    allpoints = []
    allpointsMinLat = []
    allpointsMaxLat = []
    allpointsMinLon = []
    allpointsMaxLon = []
    with codecs.open(filename, 'r', encoding=data_encoding) as inf:
        for line in inf:
            points = []
            minlat = 1000
            maxlat = -1000
            minlon = 1000
            maxlon = -1000
            fields = line.split('\t')
            for field in fields:
                dims = field.split(',')
                lat = float(dims[0])
                if lat > maxlat:
                    maxlat = lat
                if lat < minlat:
                    minlat = lat
                lon = float(dims[1])
                if lon > maxlon:
                    maxlon = lon
                if lon < minlon:
                    minlon = lon                
                point = [lat, lon]
                points.append(point)
            allpointsMinLat.append(minlat)
            allpointsMaxLat.append(maxlat)
            allpointsMaxLon.append(maxlon)
            allpointsMinLon.append(minlon)
            allpoints.append(points)
    x = []
    y = []
    for i in range(0, len(allpointsMaxLat)):
        y.append(allpointsMinLat[i])
        y.append(allpointsMinLat[i])
        y.append(allpointsMaxLat[i])
        y.append(allpointsMaxLat[i])
         
        x.append(allpointsMinLon[i])
        x.append(allpointsMaxLon[i])
        x.append(allpointsMinLon[i])
        x.append(allpointsMaxLon[i])
        
        rect = mpatches.Rectangle((allpointsMinLon[i], allpointsMinLat[i]), allpointsMaxLon[i] - allpointsMinLon[i], allpointsMaxLat[i] - allpointsMinLat[i], facecolor='white')
        ax.add_artist(rect)
        ax.set_xlim([-125, -60])  # pylab.xlim([-400, 400])
        ax.set_ylim([25, 50])
         
    # colors = np.random.rand(15)
    
    # area = 1 # 0 to 15 point radiuses
    
    # plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    # ax.set_xlim([-400, -380]) #pylab.xlim([-400, 400])
    # ax.set_ylim([-400, -380]) #pylab.ylim([-400, 400])
    # patches = []
    # polygon = plt.Rectangle((-400, -400), 10, 10, color='yellow') #Rectangle((-400, -400), 10, 10, color='y')
    # patches.append(polygon)
    
    # pol2 = plt.Rectangle((-390, -390), 10, 10, facecolor='yellow', edgecolor='violet', linewidth=2.0)
    # ax.add_artist(pol2)
    
    
    # p = collections.PatchCollection(patches) #, cmap=matplotlib.cm.jet)
    # ax.add_collection(p)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(20)) # (MultipleLocator(20)) 
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(20)) # (MultipleLocator(20)) 
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('US Map of Twitter Users partitioned by ' + partitionMethod + ' method: ' + str(granularity).strip() + ' person per cluster')
    plt.savefig(filename + '.jpg')
    plt.close()
    # plt.show()  # pylab.show()            







def createTrainDir(granularity, partitionMethod, create_dir=False):
    # readlocationclusters
    global classLatMean
    global classLatMedian
    global classLonMean
    global classLonMedian
    global testClasses
    global devClasses
    filename = path.join(GEOTEXT_HOME, 'processed_data/' + str(granularity).strip() + '_' + partitionMethod + '_clustered.train')
    allpoints = []
    allpointsMinLat = []
    allpointsMaxLat = []
    allpointsMinLon = []
    allpointsMaxLon = []
    with codecs.open(filename, 'r', encoding=data_encoding) as inf:
        for line in inf:
            points = []
            minlat = 1000
            maxlat = -1000
            minlon = 1000
            maxlon = -1000
            fields = line.split('\t')
            for field in fields:
                dims = field.split(',')
                lat = float(dims[0].strip())
                if lat > maxlat:
                    maxlat = lat
                if lat < minlat:
                    minlat = lat
                lon = float(dims[1].strip())
                if lon > maxlon:
                    maxlon = lon
                if lon < minlon:
                    minlon = lon                
                point = [dims[0].strip(), dims[1].strip()]
                points.append(point)
            allpointsMinLat.append(minlat)
            allpointsMaxLat.append(maxlat)
            allpointsMaxLon.append(maxlon)
            allpointsMinLon.append(minlon)
            allpoints.append(points)
    if create_dir:
        trainhome = path.join(GEOTEXT_HOME, 'processed_data/train')
        shutil.rmtree(trainhome, ignore_errors=True)
        os.mkdir(trainhome)
    i = -1
    for cluster in allpoints:
        # create a directory
        i += 1
        lats = [float(location[0]) for location in cluster]
        longs = [float(location[1]) for location in cluster]
        medianlat = median(lats)
        medianlon = median(longs)
        meanlat = mean(lats)
        meanlon = mean(longs)
        
        classLatMedian[str(i).strip()] = medianlat
        classLonMedian[str(i).strip()] = medianlon
        classLatMean[str(i).strip()] = meanlat
        classLonMean[str(i).strip()] = meanlon    
        
        if create_dir:
            class_dir = path.join(trainhome, str(i).strip())
            os.mkdir(class_dir)
        
        for location in cluster:
            # find user(s) in that collection
            locationStr = location[0] + ',' + location[1]
            userstr = locationUser[locationStr]
            if not userstr:
                print "fatal error: something is wrong, no user for this location: " + locationStr 
            locusers = []
            if " " in userstr:
                # multiple users separated by space in this location
                locusers = userstr.split()
            else:
                # just one single user in this location
                locusers.append(userstr.strip())
            # for each user in this location find the text
            # groupbyusersText should be true for this to work
            # print "writing user texts in their corresponding geographical class in: " + class_dir
            if create_dir:
                for user in locusers:
                    if user in trainUsers:
                        with codecs.open(path.join(class_dir, user), 'w', encoding=data_encoding) as inf:
                            inf.write(trainText[user])
    print "train directories created and class median and mean lat,lon computed. trainfile: " + filename
    devDistances = []
    for user in devUsers:
        locationStr = devUsers[user]
        latlon = locationStr.split(',')
        latitude = float(latlon[0])
        longitude = float(latlon[1])
        classIndex, dist = assignClass(latitude, longitude)
        devDistances.append(dist)
        devClasses[user] = classIndex
    
    testDistances = []
    for user in testUsers:
        locationStr = testUsers[user]
        latlon = locationStr.split(',')
        latitude = float(latlon[0])
        longitude = float(latlon[1])
        classIndex, dist = assignClass(latitude, longitude)
        testDistances.append(dist)
        testClasses[user] = classIndex

            


        
    
    print "Ideal mean dev distance is " + str(mean(devDistances))
    print "Ideal median dev distance is " + str(median(devDistances))
    
    print "Ideal mean test distance is " + str(mean(testDistances))
    print "Ideal median test distance is " + str(median(testDistances))
        
# createTrainDir()
def print_class_coordinates():
    for c in classLatMedian:
        print str(c) + '\t' + str(classLatMedian[c]) + '\t' + str(classLonMedian[c])
def assignClass(latitude, longitude):
    '''
    Given a coordinate find the class whose median is the closest point. Then return the index of that class.
    This function can be used for parameter tuning with validation data and evaluation with test data.
    '''
    minDistance = 1000000
    classIndex = -1
    for i in classLatMedian:
        lat = classLatMedian[str(i).strip()]
        lon = classLonMedian[str(i).strip()]
        dist = distance(latitude, longitude, lat, lon)
        if dist < minDistance:
            minDistance = dist
            classIndex = i
    return classIndex, minDistance
        
def createTestDevDir(type='test'):
    print 'creating ' + type + ' collection.'
    t_home = path.join(users_home, type + '/')
    shutil.rmtree(t_home, ignore_errors=True)
    os.mkdir(t_home)
    userCollection = {}
    if type == 'test':
        userCollection = testUsers
        userClasses = testClasses
        userText = testText
    elif type == 'dev':
        userCollection = devUsers
        userClasses = devClasses
        userText = devText
    else:
        print "fatal error in createTestDevDir type:" + type
        return;
    
    for user in userCollection:
        userClass = userClasses[user]
        text = userText[user]
        classDir = path.join(t_home, str(userClass).strip())
        if not os.path.exists(classDir):
            os.makedirs(classDir)
        with codecs.open(path.join(classDir, user), 'w', encoding=data_encoding) as inf:
            inf.write(text)
# createTestDevDir('test')
# createTestDevDir('dev')
def create_directories(granularity, partitionMethod, write=False):
    createTrainDir(granularity, partitionMethod, write)
    if write:
        createTestDevDir(type='dev')
        createTestDevDir(type='test')
def size_mb(docs):
    return sum(len(s.encode(encoding=data_encoding)) for s in docs) / 1e6

def evaluate(preds, U_test, categories, scores):
    print U_test
    sumMeanDistance = 0
    sumMedianDistance = 0
    distances = []
    confidences = []
    randomConfidences = []
    gmm = False
    for i in range(0, len(preds)):
        user = U_test[i]
        location = userLocation[user].split(',')
        lat = float(location[0])
        lon = float(location[1])
        # gaussian mixture model
        if gmm:
            sumMedianLat = 0
            sumMedianLon = 0
            sumMeanLat = 0
            sumMeanLon = 0
            numClasses = len(categories)
            sortedScores = sorted(scores[i], reverse=True)
            top1Score = sortedScores[0]
            top2Score = sortedScores[1]
            print top1Score
            print top2Score
            for c in range(0, numClasses):
                score = scores[i][c]
                category = categories[c]
                medianlat = classLatMedian[category]  
                medianlon = classLonMedian[category]  
                meanlat = classLatMean[category] 
                meanlon = classLonMean[category]
                sumMedianLat += score * medianlat
                sumMedianLon += score * medianlon
                sumMeanLat += score * meanlat
                sumMeanLon += score * meanlon
            distances.append(distance(lat, lon, sumMedianLat, sumMedianLon)) 
            
        else:
            prediction = categories[preds[i]]
            if scores != None:
                confidence = scores[i][preds[i]] 
                confidences.append(confidence)
            medianlat = classLatMedian[prediction]  
            medianlon = classLonMedian[prediction]  
            meanlat = classLatMean[prediction] 
            meanlon = classLonMean[prediction]      
            distances.append(distance(lat, lon, medianlat, medianlon))
            sumMedianDistance = sumMedianDistance + distance(lat, lon, medianlat, medianlon)
            sumMeanDistance = sumMeanDistance + distance(lat, lon, meanlat, meanlon)
    # averageMeanDistance = sumMeanDistance / float(len(preds))
    # averageMedianDistance = sumMedianDistance / float(len(preds))
    # print "Average mean distance is " + str(averageMeanDistance)
    # print "Average median distance is " + str(averageMedianDistance)
    print "Mean distance is " + str(mean(distances))
    print "Median distance is " + str(median(distances))
def dataSpaceExpansion(X_train, Y_train, k=1):
    print "started data space expansion with factor: " + str(k)
    n_samples = X_train.shape[0]
    n_classes = len(categories)
    neighbors = trainCostMatrix.argsort()[:, 0:k]

    expanded_X_train = sparse.vstack([X_train, X_train])
    expanded_Y_train = np.hstack((Y_train, Y_train))
    for i in range(0, n_samples):
        user = U_train[i]
        locationStr = trainUsers[user]
        lat, lon = locationStr2Float(locationStr)
        sample_neighbors = neighbors[i].tolist()
        expanded_Y_train[i] = sample_neighbors[0]
            
    return expanded_X_train, expanded_Y_train
def dataSpaceModification(Y_train, U_train):
    for i in range(0, Y_train.shape[0]):
        lat1, lon1 = locationStr2Float(trainUsers[U_train[i]])
        dmin = 100000
        cmin = -1
        for c in range(0, len(categories)):
            lat2 = classLatMedian[categories[c]]
            lon2 = classLonMedian[categories[c]]
            d = distance(lat1, lon1, lat2, lon2)
            if d < dmin:
                cmin = c
                dmin = d
        Y_train[i] = cmin
    return Y_train
            
        
def loss(preds, U_test, loss='median'):
    if len(preds) != len(U_test): 
        print "The number of test sample predictions is: " + str(len(preds))
        print "The number of test samples is: " + str(len(U_test))
        print "fatal error!"
        sys.exit()
    sumMeanDistance = 0
    sumMedianDistance = 0
    distances = []
    for i in range(0, len(preds)):
        user = U_test[i]
        location = userLocation[user].split(',')
        lat = float(location[0])
        lon = float(location[1])
        prediction = categories[preds[i]]
        medianlat = classLatMedian[prediction]  
        medianlon = classLonMedian[prediction]  
        meanlat = classLatMean[prediction] 
        meanlon = classLonMean[prediction]
        predictionCoordinate = 'median'
        if predictionCoordinate == 'median':      
            distances.append(distance(lat, lon, medianlat, medianlon))
        elif predictionCoordinate == 'mean':
            distances.append(distance(lat, lon, meanlat, meanlon))
        
    averageMeanDistance = sumMeanDistance / float(len(preds))
    averageMedianDistance = sumMedianDistance / float(len(preds))
    medianDistance = median(distances)
    # print "Average distance from class mean is " + str(averageMeanDistance)
    # print "Average distance from class median is " + str(averageMedianDistance)
    print "Mean distance is " + str(mean(distances))
    print "Median distance is " + str(median(distances))
    
    return mean(distances), median(distances)

def lossbycoordinates(coordinates):
    if len(coordinates) != len(testUsers): 
        print "The number of test sample predictions is: " + str(len(coordinates))
        print "The number of test samples is: " + str(len(testUsers))
        print "fatal error!"
        sys.exit()
    sumMeanDistance = 0
    sumMedianDistance = 0
    distances = []
    U = testUsers.keys()
    for i in range(0, len(coordinates)):
        user = U[i]
        location = userLocation[user].split(',')
        lat = float(location[0])
        lon = float(location[1])
        distances.append(distance(lat, lon, coordinates[i][0], coordinates[i][1]))
        
    # print "Average distance from class mean is " + str(averageMeanDistance)
    # print "Average distance from class median is " + str(averageMedianDistance)
    print "Mean distance is " + str(mean(distances))
    print "Median distance is " + str(median(distances))
    
    if loss == 'median':
        return median(distances)
    elif loss == 'mean':
        return mean(distances) 
def ppmiTransform(matrix):
    '''
    transform matrix to  ppmi weighting as in:
    Turney, Peter D., and Patrick Pantel. 
    From frequency to meaning: Vector space models of semantics. 
    Journal of artificial intelligence research 37.1 (2010): 141-188.
    '''
    print "PPMI transform started..." 
    # compute the sum of each row and save it in rowSums matrix [number_of_rows, 1]
    rowSums = matrix.sum(1)
    # compute the sum of each column and save it in colSums matrix [1, number_of_columns]
    colSums = matrix.sum(0)
    # compute the sum of all values in the matrix by summing over all rowSums
    # we could also sum over colSums if we knew the number of columns is much less than rows
    totalSum = rowSums.sum()
    
    # row and col indexes of nonzero values
    rows, cols = matrix.nonzero()
    for row, col in zip(rows, cols):
        pmi = math.log(float(matrix[row, col] * totalSum) / (rowSums[row, 0] * colSums[0, col]))
        if pmi > 0.0:
            matrix[row, col] = pmi
        else:
            matrix[row, col] = 0.0
    print "PPMI transform finished successfully."    
def feature_extractor(use_mention_dictionary=False, use_idf=True, norm='l2', binary=False, sublinear_tf=True, min_df=1, max_df=1.0, BuildCostMatrices=False, vectorizer=None):
    '''
    read train, dev and test directories and extract textual features using tfidfvectorizer.
    '''
    global categories
    global X_train
    global X_dev
    global X_test
    global Y_train
    global Y_dev
    global Y_test
    global U_train
    global U_dev
    global U_test
    
    global costMatrix
    global trainCostMatrix
    global testCostMatrix
    global devCostMatrix
    
    trainDir = path.join(GEOTEXT_HOME, 'processed_data/train')
    testDir = path.join(GEOTEXT_HOME, 'processed_data/test')
    devDir = path.join(GEOTEXT_HOME, 'processed_data/dev')
    
    data_train = load_files(trainDir, encoding=data_encoding)
    target = data_train.target
    data_test = load_files(testDir, encoding=data_encoding)
    data_dev = load_files(devDir, encoding=data_encoding)
    
    categories = data_train.target_names
    
    U_train = [path.basename(data_train.filenames[i]).decode(encoding=data_encoding) for i in range(0, len(data_train.filenames))]
    U_test = [path.basename(data_test.filenames[i]).decode(encoding=data_encoding) for i in range(0, len(data_test.filenames))]
    U_dev = [path.basename(data_dev.filenames[i]).decode(encoding=data_encoding) for i in range(0, len(data_dev.filenames))]

    def size_mb(docs):
        return sum(len(s.encode(encoding=data_encoding)) for s in docs) / 1e6
    
    data_train_size_mb = size_mb(data_train.data)
    data_test_size_mb = size_mb(data_test.data)
    data_dev_size_mb = size_mb(data_dev.data)
    
    print("%d documents - %0.3fMB (training set)" % (
        len(data_train.data), data_train_size_mb))
    print("%d documents - %0.3fMB (test set)" % (
        len(data_test.data), data_test_size_mb))
    print("%d documents - %0.3fMB (dev set)" % (
        len(data_dev.data), data_dev_size_mb))

    print("%d categories" % len(categories))
    print()
    
    # split a training set and a test set
    Y_train = data_train.target
    Y_test = data_test.target
    Y_dev = data_dev.target
    
   
    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time.time()
    
    if vectorizer == None:    
        if use_mention_dictionary:
            extract_mentions()
            vectorizer = TfidfVectorizer(use_idf=use_idf, norm=norm, binary=binary, sublinear_tf=sublinear_tf, min_df=min_df, max_df=max_df, ngram_range=(1, 1), stop_words=None, vocabulary=mentions)
        else:
            vectorizer = TfidfVectorizer(use_idf=use_idf, norm=norm, binary=binary, sublinear_tf=sublinear_tf, min_df=min_df, max_df=max_df, ngram_range=(1, 1), stop_words=None)
    
    X_train = vectorizer.fit_transform(data_train.data)
    print type(vectorizer.vocabulary_)
    keys = vectorizer.vocabulary_.keys()
    for i in range(0, 10):
        print keys[i], vectorizer.vocabulary_[keys[i]]
        
    '''
    #test the sum of a doc feature values
    test = X_train[0].todense()
    print test.shape
    nn = test.shape[0]
    vv = test.shape[1]
    print nn
    print vv
    summm = 0
    for l in range(0, vv):
         summm += test[0, l]
    print summm
    '''
    duration = time.time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()
    
    print("Extracting features from the dev dataset using the same vectorizer")
    t0 = time.time()
    X_dev = vectorizer.transform(data_dev.data)
    duration = time.time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_dev_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_dev.shape)
    print()

    print("Extracting features from the test dataset using the same vectorizer")
    t0 = time.time()
    X_test = vectorizer.transform(data_test.data)
    duration = time.time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()
    ppmitransform = False
    if ppmitransform:
        ppmiTransform(X_train)
        ppmiTransform(X_dev)
        ppmiTransform(X_test)
    
            
    chi = False
    if chi:
        k = 20000
        print("Extracting %d best features by a chi-squared test" % k)
        t0 = time.time()
        ch2 = SelectKBest(chi2, k=k)
        X_train = ch2.fit_transform(X_train, Y_train)
        X_test = ch2.transform(X_test)
        X_dev = ch2.transform(X_dev)
        print("done in %fs" % (time.time() - t0))
        print()
        # feature_names = np.asarray(vectorizer.get_feature_names())
    feature_names = np.asarray(vectorizer.get_feature_names())

    DO_SVD = False
    Reduction_D = 1000
    if DO_SVD:
        print("dimension reduction svd with d=%d" % Reduction_D)
        svd = TruncatedSVD(n_components=Reduction_D, algorithm="randomized", n_iterations=5, random_state=None, tol=0)
        X_train = svd.fit_transform(X_train)
        X_test = svd.transform(X_test)
        X_dev = svd.transform(X_dev)
        print("dimension reduction finished.")
        
    if BuildCostMatrices:
        print "building cost matrix..."
        costMatrix = np.ndarray(shape=(len(classLatMedian), len(classLatMedian)), dtype=float)
        for i in range(0, len(categories)):
            lat = classLatMedian[str(i)]
            lon = classLonMedian[str(categories[i])]
            for j in classLatMedian:
                lat2 = classLatMedian[j]
                lon2 = classLonMedian[j]
                cost = distance(lat, lon, lat2, lon2)
                costMatrix[i, j] = cost
        
        print "building sample based cost matrix..."
        trainCostMatrix = np.ndarray(shape=(X_train.shape[0], len(categories)), dtype=float)
        for i in range(0, trainCostMatrix.shape[0]):
            lat, lon = locationStr2Float(trainUsers[U_train[i]])
            for j in range(0, trainCostMatrix.shape[1]):
                lat2 = classLatMedian[str(j)]
                lon2 = classLonMedian[str(j)]
                trainCostMatrix[i, j] = distance(lat, lon, lat2, lon2)
        
        devCostMatrix = np.ndarray(shape=(X_dev.shape[0], len(categories)), dtype=float)
        for i in range(0, devCostMatrix.shape[0]):
            lat, lon = locationStr2Float(devUsers[U_dev[i]])
            for j in range(0, devCostMatrix.shape[1]):
                lat2 = classLatMedian[str(j)]
                lon2 = classLonMedian[str(j)]
                devCostMatrix[i, j] = distance(lat, lon, lat2, lon2)        
        
        testCostMatrix = np.ndarray(shape=(X_test.shape[0], len(categories)), dtype=float)
        for i in range(0, testCostMatrix.shape[0]):
            lat, lon = locationStr2Float(testUsers[U_test[i]])
            for j in range(0, testCostMatrix.shape[1]):
                lat2 = classLatMedian[str(j)]
                lon2 = classLonMedian[str(j)]
                testCostMatrix[i, j] = distance(lat, lon, lat2, lon2)  
            
    return X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names
    
def abod(probs, preds, U_test):
    # print "running abod to find the Approximate Bayes-Optimal Decision..."
    n_samples = probs.shape[0]
    n_categories = probs.shape[1]
    assert n_categories == len(categories), "fatal error: n_categories is not equal to len(categories) in abod"
    probs = probs ** 1
    preds = preds ** 1
    preds2 = []
    for s in range(0, n_samples):
        minCost = 1000000000.0
        minCostClass = -1
        # select top n prob indices
        # for cf in sorted(range(len(probs[s])), key=lambda i: probs[s, i])[-3:]:
        for cf in range(0, n_categories):
            # print probs[s]
            cost = 0
            for co in range(0, n_categories):
                cost = cost + probs[s, co] * costMatrix[cf, co]
            if cost < minCost:
                minCost = cost
                minCostClass = cf
        preds2.append(minCostClass)
    return loss(preds2, U_test)
    
    
def classify(X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names, granularity=10, DSExpansion=False, DSModification=False, compute_dev=True, report_verbose=False, clf=None):
    
    if DSExpansion:
        X_train, Y_train = dataSpaceExpansion(X_train, Y_train)
    
    if DSModification:
        Y_train = dataSpaceModification(Y_train, U_train)
    if clf==None:
        # clf = LinearSVC(multi_class='ovr', class_weight='auto', C=1.0, loss='l2', penalty='l1', dual=False, tol=1e-3)
        #clf = linear_model.LogisticRegression(C=1.0, penalty='l2')
        clf = SGDClassifier(loss='log', alpha=0.000001, penalty='l1', learning_rate='optimal', n_iter=5, shuffle=False, n_jobs=20)
        # clf = MultiTaskLasso()
        # clf = ElasticNet()
        # clf = linear_model.Lasso(alpha = 0.1)
        
        # clf = SGDClassifier(loss, penalty, alpha, l1_ratio, fit_intercept, n_iter, shuffle, verbose, epsilon, n_jobs, random_state, learning_rate, eta0, power_t, class_weight, warm_start, rho, seed)
        # clf = linear_model.MultiTaskLasso(alpha=0.1)
        # clf = RidgeClassifier(tol=1e-2, solver="auto")
        # clf = RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=1e-2, class_weight=None, solver="auto")
        # clf = SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
        # clf = Perceptron(n_iter=50)
        # clf = PassiveAggressiveClassifier(n_iter=50)
        # clf = KNeighborsClassifier(n_neighbors=10)
        # clf = NearestCentroid()
        #clf = MultinomialNB(alpha=.01)
    print('_' * 80)
    print("Training: ")
    print(clf)
    
    
    t0 = time.time()
    clf.fit(X_train, Y_train)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)
    
    
    
    if compute_dev:
        devPreds = clf.predict(X_dev)
        # score1 = None
        # Score2 = None
        score1 = metrics.f1_score(Y_dev, devPreds)
        score2 = metrics.accuracy_score(Y_dev, devPreds)
        
        if report_verbose:
            print '**********dev************'
            print("classification report:")
            print(metrics.classification_report(Y_dev, devPreds, target_names=categories))
            print("confusion matrix:")
            print(metrics.confusion_matrix(Y_dev, devPreds))
            print("f1-score:   %0.3f" % score1)
            print("Accuracy score:   %0.3f" % score2)
            if hasattr(clf, 'coef_'):
                print("dimensionality: %d" % clf.coef_.shape[1])
                print("density: %f" % density(clf.coef_))
                print("top 10 keywords per class:")
                for i, category in enumerate(categories):
                    top10 = np.argsort(clf.coef_[i])[-10:]
                    print("%s: %s" % (category, " ".join(feature_names[top10])))
    
        # loss(devPreds, U_dev)
    
    
        
    t0 = time.time()
    preds = clf.predict(X_test)
    # scores = clf.decision_function(X_test)
    # probs = clf.predict_proba(X_test)
    probs = None
    # print preds.shape
    test_time = time.time() - t0
    
    
    
    score1 = metrics.f1_score(Y_test, preds)
    score2 = metrics.accuracy_score(Y_test, preds)
    if report_verbose:
        print "************test*************"
        print("test time:  %0.3fs" % test_time)
        print("classification report:")
        print(metrics.classification_report(Y_test, preds, target_names=categories))
        print("confusion matrix:")
        print(metrics.confusion_matrix(Y_test, preds))
        print("f1-score:   %0.3f" % score1)
        print("Accuracy score:   %0.3f" % score2)
        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))
    print "test results"
    meanTest, medianTest = loss(preds, U_test)
    meanDev = -1
    medianDev = -1
    if compute_dev:
        print "development results"
        meanDev, medianDev = loss(devPreds, U_dev)
    # loss(preds)
    # evaluate(preds,U_test, categories, None)
    # abod(probs, preds, U_test)

    return preds, probs, U_test, meanTest, medianTest, meanDev, medianDev
   
# classify()

def loadGPData(DO_SVD=False, Reduction_D=100):
    data = {}
    
    trainlats = []
    trainlongs = []
    traintexts = []
    trainlocs = []
    
    testlats = []
    testlongs = []
    testtexts = []
    testlocs = []
    
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor()
    
    for i in range(0, len(U_train)):
        user = U_train[i]
        loc = trainUsers[user]
        latlon = loc.split(',')
        lat = float(latlon[0])
        lon = float(latlon[1])
        text = trainText[user]
        trainlats.append(lat)
        trainlongs.append(lon)
        traintexts.append(text)
        trainlocs.append([lat, lon])

    for i in range(0, len(U_test)):
        user = U_test[i]
        loc = testUsers[user]
        latlon = loc.split(',')
        lat = float(latlon[0])
        lon = float(latlon[1])
        text = testText[user]
        testlats.append(lat)
        testlongs.append(lon)
        testtexts.append(text)
        testlocs.append([lat, lon])

    
    
    if DO_SVD:
        print("dimension reduction svd with d=%d" % Reduction_D)
        svd = TruncatedSVD(n_components=Reduction_D, algorithm="randomized", n_iterations=5, random_state=None, tol=0)
        X_train = svd.fit_transform(X_train)
        X_test = svd.transform(X_test)
        X_dev = svd.transform(X_dev)
        print("dimension reduction finished.")
    else:
        X_train = X_train.toarray()
        X_test = X_test.toarray()
        X_dev = X_dev.toarray() 
    Y_train = np.asanyarray(trainlocs)
    print("Y_train: n_samples: %d, n_features: %d" % Y_train.shape)
    Y_test = np.asanyarray(testlocs) 
    print("Y_test: n_samples: %d, n_features: %d" % Y_test.shape)
    '''
    if X_train.issparse():
        X_train = X_train.todense()
    if X_test.issparse():
        X_test = X_test.todense()
    '''
    data['Y'] = X_train
    data['Ytest'] = X_test
    data['X'] = Y_train
    data['Xtest'] = Y_test
    return data
        
    
    
def localizeGP(max_iters=100, kernel=None, optimize=False, plot=False):
    """Predict the location of a robot given wirelss signal strength readings."""
        
    # data = GPy.util.datasets.robot_wireless()
    data = loadGPData()
    print data
    # create simple GP Model
    m = GPy.models.GPRegression(data['Y'][:100], data['X'][:100], kernel=kernel)

    # globals().update(locals())

    
    # optimize
    if optimize:
        m.optimize(messages=True, max_iters=max_iters)

    results = m.predict(data['Ytest'], full_cov=True)
    print len(results)
    Xpredict = results[0]
    Xvar = results[1]
    

    Tracer()()
    if plot:
        pb.plot(data['Xtest'][:, 0], data['Xtest'][:, 1], 'r-')
        pb.plot(Xpredict[:, 0], Xpredict[:, 1], 'b-')
        pb.axis('equal')
        pb.title('WiFi Localization with Gaussian Processes')
        pb.legend(('True Location', 'Predicted Location'))
        
    sumDist = 0
    distances = []
    for i in range(0, Xpredict.shape[0]):
        lat1 = Xpredict[i][0]
        lon1 = Xpredict[i][1]
        lat2 = data['Xtest'][i][0]
        lon2 = data['Xtest'][i][1]
        distances.append(distance(lat1, lon1, lat2, lon2))

    print "mean distance is: " + str(mean(distances))
    print "median distance is: " + str(median(distances))
    with open('GP_results-' + ".pkl", 'wb') as outf:
        pickle.dump((Xpredict, Xvar), outf)
    
    # sse = ((data['Xtest'] - Xpredict)**2).sum()
    # aae = np.absolute(data['Xtest'] - Xpredict).sum()
    # print m
    # print('Sum of squares error on test data: ' + str(sse))
    # print('average absolute error on test data: ' + str(aae))
    if plot:
        fig = pb.figure(None)
        pb.title('')
        raw_input()
    return m
def wireless(max_iters=100, kernel=None, optimize=True, plot=True):
    """Predict the location of a robot given wirelss signal strength readings."""
    # data = loadGPData()
    data = GPy.util.datasets.robot_wireless()
    print data
    # create simple GP Model
    m = GPy.models.GPRegression(data['Y'], data['X'], kernel=kernel, normalize_X=True, normalize_Y=True)

    # optimize
    if optimize:
        m.optimize(messages=True, max_iters=max_iters)

    Xpredict, Xvar, Xlow, Xhigh = m.predict(data['Ytest'])
    if plot:
        pb.plot(data['Xtest'][:, 0], data['Xtest'][:, 1], 'r-')
        pb.plot(Xpredict[:, 0], Xpredict[:, 1], 'b-')
        pb.axis('equal')
        pb.title('WiFi Localization with Gaussian Processes')
        pb.legend(('True Location', 'Predicted Location'))
      
    # sse = ((data['Xtest'] - Xpredict)**2).sum()
    aae = np.absolute(data['Xtest'] - Xpredict).sum()
    print m
    # print('Sum of squares error on test data: ' + str(sse))
    print('average absolute error on test data: ' + str(aae))
    if plot:
        fig = pb.figure(None)
        pb.title('')
        raw_input()
    return m

def wirelessSGD(max_iters=100, kernel=None, optimize=True, plot=True):
    """Predict the location of a robot given wirelss signal strength readings."""
    data = loadGPData()
    # data = GPy.util.datasets.robot_wireless()
    print data
    # create simple GP Model
    m = GPy.models.GPMultioutputRegression(data['Y'], data['X'], normalize_X=True, normalize_Y=True)

    # optimize
    if optimize:
        m.optimize(messages=True, max_iters=max_iters)

    Xpredict = m.predict(data['Ytest'])[0]
    if plot:
        pb.plot(data['Xtest'][:, 0], data['Xtest'][:, 1], 'r-')
        pb.plot(Xpredict[:, 0], Xpredict[:, 1], 'b-')
        pb.axis('equal')
        pb.title('WiFi Localization with Gaussian Processes')
        pb.legend(('True Location', 'Predicted Location'))
        

    # sse = ((data['Xtest'] - Xpredict)**2).sum()
    aae = np.absolute(data['Xtest'] - Xpredict).sum()
    print m
    # print('Sum of squares error on test data: ' + str(sse))
    print('average absolute error on test data: ' + str(aae))
    if plot:
        fig = pb.figure(None)
        pb.title('')
        raw_input()
    return m

def initialize(partitionMethod, granularity, write=False):    
    global records
    global lngs
    global ltts
    global keys
    global userFirstTime
    global userLocation
    global locationUser
    global userlon
    global userlat
    global testUsers
    global devUsers
    global trainUsers
    global classLatMedian
    global classLatMean
    global classLonMedian
    global classLonMean
    global devClasses
    global testClasses
    global categories


    records = []
    lngs = []
    ltts = []
    pointText = {}
    keys = []
    userFirstTime = {}
    userLocation = {}
    locationUser = {}
    userlat = {}
    userlon = {}
    testUsers = {}
    trainUsers = {}
    devUsers = {}
    classLatMedian = {}
    classLonMedian = {}
    classLatMean = {}
    classLonMean = {}
    devClasses = {}
    testClasses = {}
    categories = []
    

    #readGeoTextRecords(encoding=data_encoding)
    print 'reading (user_info.) train, dev and test file and building trainUsers, devUsers and testUsers with their locations'
    users(trainfile, 'train')
    users(devfile, 'dev')
    users(testfile, 'test')
    print "the number of train" + " users is " + str(len(trainUsers))
    print "the number of test" + " users is " + str(len(testUsers))
    print "the number of dev" + " users is " + str(len(devUsers))
    #print 'total ' + str(len(userLocation)).strip() + " users."
    #fillUserByLocation()
    #fillTextByUser(encoding=data_encoding)
    create_directories(granularity, partitionMethod, write)        

def classificationBench(granularity, partitionMethod, use_mention_dictionary=False):
    medians = {}
    max_dfs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
    #max_dfs = [1, 5, 10, 20]
    clf1 = MultinomialNB(alpha=.01)
    clf2 = SGDClassifier(loss='log',penalty='l1', learning_rate='optimal', n_iter=5, shuffle=False, n_jobs=20)
    clf3 = SGDClassifier(loss='log',penalty='l2', learning_rate='optimal', n_iter=5, shuffle=False, n_jobs=20)    
    clf4 = NearestCentroid()
    clf5 = LinearSVC(multi_class='ovr', class_weight='auto', C=1.0, loss='l2', penalty='l1', dual=False, tol=1e-3)
    clf6 = LinearSVC(multi_class='ovr', class_weight='auto', C=1.0, loss='l2', penalty='l2', dual=False, tol=1e-3)
    classifiers = [clf1, clf2, clf3, clf4, clf5, clf6]
    for i in range(0, len(classifiers)):
        clf = classifiers[i]
        medians[i] = []
        for max_df in max_dfs:
            X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor(use_mention_dictionary=use_mention_dictionary, max_df=max_df)    
            preds, probs, U_test, meanTest, medianTest, meanDev, medianDev = classify(X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names, granularity, clf=clf)
            meanD, medianD = loss(preds, U_test)
            medians[i].append(medianD)
            del X_train
            del X_test
            del X_dev
    markers = ['b-', 'r-', 'g-', 'c-', 'm-', 'y-', 'k-', 'r--']
    labels = ['MN-NaiveBayes', 'LR-l1', 'LR-l2', 'Nearest Centroid', 'LinearSVM-l1', 'LinearSVM-l2']
    costFunction = 'Median distance (Y axis) as a function of max doc freq (X axis)'

    for i in range(0, len(classifiers)):
        y = medians[i]
        x = max_dfs
        minIndex = np.argmin(y)
        plt.text(x[minIndex], y[minIndex], "min")
        plt.title(costFunction)
        plt.plot(x, y, markers[i], label=labels[i], linewidth=2)
    plt.xlabel('max doc frequency', fontsize=18)
    plt.ylabel('median distance', fontsize=16)
    legend = plt.legend(loc='upper right', shadow=False, fontsize='small')
    # legend.get_frame().set_facecolor('#00FFCC')
    plt.savefig('result.png')
    Tracer()()
    '''
    if len(sys.argv) == 1 or sys.argv[1] != 'server':
        partitionLocView(granularity=granularity, partitionMethod=partitionMethod)
    return preds, probs, U_test, meanTest, medianTest, meanDev, medianDev
    '''
def asclassification(granularity, partitionMethod, use_mention_dictionary=False):

    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor(use_mention_dictionary=use_mention_dictionary, max_df=1.0, min_df=1)    
    preds, probs, U_test, meanTest, medianTest, meanDev, medianDev = classify(X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names, granularity)
    partitionLocView(granularity=granularity, partitionMethod=partitionMethod)
    return preds, probs, U_test, meanTest, medianTest, meanDev, medianDev


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        
        # self.W = theano.shared(value=numpy.random.rand(n_in, n_out), name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]
        self.A = T.transpose(self.W)
        self.squared_euclidean_distances = (self.A ** 2).sum(1).reshape((self.A.shape[0], 1)) + (self.A ** 2).sum(1).reshape((1, self.A.shape[0])) - 2 * self.A.dot(self.A.T)
        
    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        # print T.diag(T.dot(self.p_y_given_x[T.arange(y.shape[0]), 0:24], c[0:24, y]))
        
        # 1 main
        # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # 2 entropy nll + entropy
        # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) - T.mean(T.sum(self.p_y_given_x * T.log(self.p_y_given_x), axis=1))
        # 3 nll + regularization
        # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) + 0.0001 * T.sum(self.W**2)
        # 4 least square error
        # return T.mean(T.sqr(1 - self.p_y_given_x[T.arange(y.shape[0]), y]))
        # 5 nnl + l1regul
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) + 0.001 * T.sum(T.abs_(self.W))
    def example_cost_sensitive(self, y, sc):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        # print T.diag(T.dot(self.p_y_given_x[T.arange(y.shape[0]), 0:24], c[0:24, y]))
        # 1 main
        # return T.mean(T.sum(self.p_y_given_x * sc, axis=1))
        # 2 trevor
        # return -T.mean(T.log(self.p_y_given_x) * T.exp(-0.01 * sc))
        # 3 nll+trevor
        # return -0.1* T.mean(T.log(self.p_y_given_x) * T.exp(-0.01 * sc)) - T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # 6
        # return -T.mean(T.log(self.p_y_given_x) * T.exp(-0.01 * sc)) - T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # 7
        # return T.mean(T.log(self.p_y_given_x) * T.exp(-0.01 * sc)) * T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # 8
        return T.mean(T.log(self.p_y_given_x ** 2) * T.exp(-0.01 * sc)) * T.mean(T.log(self.p_y_given_x ** 2)[T.arange(y.shape[0]), y])
        

    def cost_sensitive_loss(self, y, c):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        
        # mean of costs
        # 1
        return T.mean(T.sum(self.p_y_given_x * T.transpose(c[:, y]), axis=1))
        # 2
        # return T.mean(self.p_y_given_x[T.arange(y.shape[0]), y] * c[self.y_pred, y])
        # 3
        # return T.mean(c[self.y_pred, y])
        # 4 main
        # return 0.001 * T.mean(T.sum(self.p_y_given_x * T.transpose(c[:, y]), axis=1)) - T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # 5
        # return 0.0001 * T.mean(T.sum(self.p_y_given_x * T.transpose(c[:, y]), axis=1)) - T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # 6
        # return -T.mean(T.log(T.diag(T.dot(self.p_y_given_x, c[:, y])) / T.max(T.diag(T.dot(self.p_y_given_x, c[:, y]))))) -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # 7
        # return T.mean(T.diag(T.dot(self.p_y_given_x, c[:, y])))
        # 8
        # return T.min(T.diag(T.dot(self.p_y_given_x, c[:, y])))
        # 9
        # return T.max(T.diag(T.dot(self.p_y_given_x, c[:, y])))
        # 10
        # return T.max(T.diag(T.dot(self.p_y_given_x, c[:, y]))) + T.mean(T.diag(T.dot(self.p_y_given_x, c[:, y])))
        # 11
        # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # 12
        # return T.mean(self.p_y_given_x[T.arange(y.shape[0]), y] * c[self.y_pred, y])
        # 13
        # return T.mean(T.sum(-T.log(self.p_y_given_x) * T.transpose(c[:, y]), axis=1))
        # 14
        # return T.mean(T.sum(self.p_y_given_x * T.exp(0.001 * T.transpose(c[:, y])), axis=1))
        # 15
        # return T.mean(T.sum((self.p_y_given_x**3) * T.transpose(c[:, y]), axis=1))
        # 16
        # return T.mean(T.sum((T.exp(self.p_y_given_x)) * T.transpose(c[:, y]), axis=1))
        # 17
        '''
        similarity = T.dot(T.transpose(self.W), self.W)
        square_mag = T.diag(similarity) + 0.0001
        inv_square_mag = 1 / square_mag
        #inv_square_mag[T.isinf(inv_square_mag)] = 0
        
        inv_mag = T.sqrt(inv_square_mag)
        cosine = similarity * inv_mag
        distance = 1 - cosine
        
        
        distance = euclidean_distances(T.transpose(self.W), T.transpose(self.W))
        #Tracer()()
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) + T.sum(c * distance)
         
        A = T.transpose(self.W)
        AA = T.reshape(T.sum(A*A, axis=1), (A.shape[0], 1))
        BB = T.transpose(AA)
        #BB[[0, 0], :]
        DD = AA[:, T.arange(0, 32 )] + BB[T.arange(0, 32), :] -  2 * T.dot(A,  self.W) 
        #DD = T.tile(AA, (1, 32)) + T.tile(BB, (32, 1)) - 2 * T.dot(A,  self.W) 
        '''
        
        # f_euclidean = theano.function([X, Y], T.sqrt(squared_euclidean_distances))
        # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) + 0.00000000005 * T.sum(c / (self.squared_euclidean_distances + 0.00001))
        
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    '''
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    print type(train_set)
    print type(test_set)
    x, y = train_set
    print type(x)
    print x.shape
    print type(y)
    print y.shape
    '''
    create_directories(granularity=300, write=False, partitionMethod='median')
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor()
    test_set = (X_test.toarray(), Y_test)
    valid_set = (X_dev.toarray(), Y_dev)
    train_set = (X_train.toarray(), Y_train)

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # witch row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target
    # target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    # costMatrixNorm = normalize(costMatrix, norm='l2', axis=1, copy=True)
    missclassificationCostMatrix = theano.shared(costMatrix, borrow=True)
    trainCostMatrix2 = normalize(trainCostMatrix, norm='l2', axis=1, copy=True)
    missclassificationTrainCostMatrix = theano.shared(trainCostMatrix2, borrow=True)
    missclassificationDevCostMatrix = theano.shared(devCostMatrix, borrow=True)
    missclassificationTestCostMatrix = theano.shared(testCostMatrix, borrow=True)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y), Y_test, missclassificationCostMatrix, missclassificationTrainCostMatrix, missclassificationDevCostMatrix, missclassificationTestCostMatrix]
    return rval
def sgd_optimization_mnist(learning_rate=0.1, n_epochs=100,
                           dataset='mnist.pkl.gz',
                           batch_size=1, modelType='nll'):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    Y_test = datasets[3]
    missclassificationCostMatrix = datasets[4]
    missclassificationTrainCostMatrix = datasets[5]
    missclassificationDevCostMatrix = datasets[6]
    missclassificationTestCostMatrix = datasets[7]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    print 'learning_rate = ' + str(learning_rate) + ' batch_size = ' + str(batch_size) + ' epochs = ' + str(n_epochs)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels
    c = T.matrix('c')  # misclassification cost matrix
    sc = T.matrix('sc')  # sample dependent cost
    

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    # classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
    classifier = LogisticRegression(input=x, n_in=train_set_x.get_value(borrow=True).shape[1], n_out=len(categories))

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)
    # cost = classifier.cost_sensitive_loss(y, c)
    # cost = classifier.example_cost_sensitive(y, sc)
    
    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
    
    predict = theano.function(inputs=[],
            outputs=[classifier.y_pred, classifier.errors(y)],
            givens={
                x: test_set_x,
                y: test_set_y})
    
    predictDev = theano.function(inputs=[],
        outputs=[classifier.y_pred, classifier.errors(y)],
        givens={
            x: valid_set_x,
            y: valid_set_y})
    probs = theano.function(inputs=[],
            outputs=classifier.p_y_given_x,
            givens={
                x: test_set_x})
    
    evalCostNLL = theano.function(inputs=[],
            outputs=classifier.negative_log_likelihood(y),
            givens={
                x: test_set_x,
                y: test_set_y})

    evalCostClass = theano.function(inputs=[],
            outputs=classifier.cost_sensitive_loss(y, c),
            givens={
                x: test_set_x,
                y: test_set_y,
                c: missclassificationCostMatrix})
    
    evalCostExample = theano.function(inputs=[],
            outputs=classifier.example_cost_sensitive(y, sc),
            givens={
                x: test_set_x,
                y: test_set_y,
                sc: missclassificationTestCostMatrix})

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]
    
    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    
    train_model = theano.function(inputs=[index],
            outputs=classifier.negative_log_likelihood(y),
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})   
             
    '''
    train_model = theano.function(inputs=[index],
            outputs=classifier.cost_sensitive_loss(y, c),
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size],
                c: missclassificationCostMatrix})
    
    train_model = theano.function(inputs=[index],
            outputs=classifier.example_cost_sensitive(y, sc),
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size],
                sc: missclassificationTrainCostMatrix[index * batch_size:(index + 1) * batch_size]})
    '''
    
    



    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 500000  # look as this many examples regardless
    patience_increase = 4  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.9995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    best_predictions = []
    start_time = time.clock()

    done_looping = False
    epoch = 0
    change2class = False
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        best_predictions , errorRate = predict()
        valid_best_predictions, valid_errorRate = predictDev()
        valid_meanD, valid_medianD = loss(valid_best_predictions, U_dev)
        meanD, medianD = loss(best_predictions, U_test)
        all_probs = probs()
        abodMean, abodMedian = abod(all_probs, best_predictions, U_test)
        print str(epoch) + ',' + str(evalCostNLL()) + ',' + str(evalCostClass()) + ',' + str(evalCostExample()) + ',' + str(errorRate) + ',' + str(meanD) + ',' + str(medianD) + ',' + str(abodMean) + ',' + str(abodMedian) + '\n'
        if (epoch > 100 or errorRate < 0.65) and change2class:
            change2class = False
            print "changed to class based mode"
            cost = classifier.cost_sensitive_loss(y, c)
            train_model = theano.function(inputs=[index],
            outputs=classifier.cost_sensitive_loss(y, c),
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size],
                c: missclassificationCostMatrix})

        
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                '''
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))
                '''
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    '''
                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') % 
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))
                    
                    best_predictions = predict()[0]
                    all_probs = probs()
            
                    print "class predicted for user " + U_test[0] + " : " + categories[best_predictions[0]] + " but it should be " + categories[Y_test[0]]
                    print "class predicted for user " + U_test[1] + " : " + categories[best_predictions[1]] + " but it should be " + categories[Y_test[1]]
                    print "class predicted for user " + U_test[2] + " : " + categories[best_predictions[2]] + " but it should be " + categories[Y_test[2]]
                    print "class predicted for user " + U_test[3] + " : " + categories[best_predictions[3]] + " but it should be " + categories[Y_test[3]]
                    print "class predicted for user " + U_test[4] + " : " + categories[best_predictions[4]] + " but it should be " + categories[Y_test[4]]
                    print "class predicted for user " + U_test[5] + " : " + categories[best_predictions[5]] + " but it should be " + categories[Y_test[5]]
                    print "class predicted for user " + U_test[6] + " : " + categories[best_predictions[6]] + " but it should be " + categories[Y_test[6]]
                    print "class predicted for user " + U_test[7] + " : " + categories[best_predictions[7]] + " but it should be " + categories[Y_test[7]]
                    print "class predicted for user " + U_test[8] + " : " + categories[best_predictions[8]] + " but it should be " + categories[Y_test[8]]
                    print "class predicted for user " + U_test[9] + " : " + categories[best_predictions[9]] + " but it should be " + categories[Y_test[9]]
                    print "class predicted for user " + U_test[10] + " : " + categories[best_predictions[10]] + " but it should be " + categories[Y_test[10]]
                    print "class predicted for user " + U_test[1890] + " : " + categories[best_predictions[1890]] + " but it should be " + categories[Y_test[1890]]
                    print "class predicted for user " + U_test[1891] + " : " + categories[best_predictions[1891]] + " but it should be " + categories[Y_test[1891]]
                    print "class predicted for user " + U_test[1892] + " : " + categories[best_predictions[1894]] + " but it should be " + categories[Y_test[1892]]
                    print "class predicted for user " + U_test[1893] + " : " + categories[best_predictions[1893]] + " but it should be " + categories[Y_test[1893]]
                    print "class predicted for user " + U_test[1894] + " : " + categories[best_predictions[1894]] + " but it should be " + categories[Y_test[1894]]
                    
                    loss(best_predictions, U_test)
                    
                    abod(all_probs, best_predictions, U_test)
                    
                    print "NNL = " + str(evalCostNLL())
                    print "class based cost = " + str(evalCostClass())
                    print "sample based cost = " + str(evalCostExample())
                    print "---------------"
                    '''
            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    '''
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') % 
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' + 
                          os.path.split(__file__)[1] + 
                          ' ran for %.1fs' % ((end_time - start_time)))
    '''
def theano_fit_predict(X_train, Y_train, X_test, Y_test, costM=None, learning_rate=0.1, n_epochs=10,
                           dataset='mnist.pkl.gz',
                           batch_size=20):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    test_set = (X_test, Y_test)
    train_set = (X_train, Y_train)
    
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
    
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    
    # datasets = load_data(dataset)
    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    missclassificationCostMatrix = None
    if costM is not None:
        missclassificationCostMatrix = theano.shared(costM, borrow=True)
    
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels
    c = T.matrix('c')  # misclassification cost matrix

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    # classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
    classifier = LogisticRegression(input=x, n_in=train_set_x.get_value(borrow=True).shape[1], n_out=4)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    if costM == None:
        cost = classifier.negative_log_likelihood(y)
    else:
        cost = classifier.cost_sensitive_loss(y, c)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    
    predict = theano.function(inputs=[],
            outputs=[classifier.y_pred, classifier.errors(y)],
            givens={
                x: test_set_x,
                y: test_set_y})
    probs = theano.function(inputs=[],
            outputs=classifier.p_y_given_x,
            givens={
                x: test_set_x})

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]
    


    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    if costM is None:
         train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})
    else:
        train_model = theano.function(inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size:(index + 1) * batch_size],
                    y: train_set_y[index * batch_size:(index + 1) * batch_size],
                    c: missclassificationCostMatrix})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000000  # look as this many examples regardless
    patience_increase = 4  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.9995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_test_loss = numpy.inf
    test_score = 0.
    best_predictions = []
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index


            test_losses = [test_model(i) for i in xrange(n_test_batches)]
            
            if test_losses < best_test_loss:
                best_test_loss = test_losses
                test_score = numpy.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') % 
                      (epoch, minibatch_index + 1, n_train_batches,
                       test_score * 100.))
            best_predictions = predict()[0]
            
            # all_probs = probs()
            
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') % 
                 (best_test_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' + 
                          os.path.split(__file__)[1] + 
                          ' ran for %.1fs' % ((end_time - start_time))) 
    return best_predictions   
   


    

    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    Y_test = datasets[3]
    missclassificationCostMatrix = datasets[4]
    missclassificationTrainCostMatrix = datasets[5]
    missclassificationDevCostMatrix = datasets[6]
    missclassificationTestCostMatrix = datasets[7]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    # classifier = MLP(rng=rng, input=x, n_in= 28 * 28, n_hidden=n_hidden, n_out=10)
    classifier = MLP(rng=rng, input=x, n_in=train_set_x.get_value(borrow=True).shape[1],
                     n_hidden=n_hidden, n_out=len(categories))

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
    
    predict = theano.function(inputs=[],
        outputs=[classifier.logRegressionLayer.y_pred, classifier.logRegressionLayer.errors(y)],
        givens={
            x: test_set_x,
            y: test_set_y})
    probs = theano.function(inputs=[],
            outputs=classifier.logRegressionLayer.p_y_given_x,
            givens={
                x: test_set_x})

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))
    
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})
    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  
    # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  
    # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    

    
    

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        best_predictions , errorRate = predict()
        meanD, medianD = loss(best_predictions, U_test)
        print "mean distance is: " + str(meanD)
        print "median distance is: " + str(medianD)
        print "error rate: " + str(errorRate)
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % 
                     (epoch, minibatch_index + 1, n_train_batches,
                      this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                           improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') % 
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                    done_looping = True
                    break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') % 
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' + 
                          os.path.split(__file__)[1] + 
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

def locationStr2Float(locationStr):
    latlon = locationStr.split(',')
    lat = float(latlon[0])
    lon = float(latlon[1])
    return lat, lon
def mixtureModel():
    createTrainDir(640)
    createTestDevDir('test')
    createTestDevDir('dev')
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor()

    # Number of samples per component
    X = []
    n_samples = len(trainUsers)
    for u in trainUsers:
        locationStr = trainUsers[u]
        lat, lon = locationStr2Float(locationStr)
        X.append([lat, lon])
    X = np.asarray(X)
    # Generate random sample, two components

    
    
    lowest_bic = np.infty
    bic = []
    n_components_range = range(25, 26)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                print "This is the best covariance type till now: " + cv_type

    
    bic = np.array(bic)
    color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
    clf = best_gmm
    bars = []
    
    
    # Plot the BIC scores
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    
    '''
    # Plot the winner
    splot = plt.subplot(1, 1, 1)
    Y_ = clf.predict(X)
    xpoints = [latlon[0] for latlon in clf.means_]
    ypoints = [latlon[1] for latlon in clf.means_]
    print clf.means_
    plt.scatter(xpoints, ypoints)
    
    for i, (mean, covar, color) in enumerate(zip(clf.means_, clf.covars_,
                                                 color_iter)):
        v, w = linalg.eigh(covar)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
    
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 4
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)
    
    plt.xlim(-10, 10)
    plt.ylim(-3, 6)
    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM: full model 25 components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    '''
    
    plt.savefig(path.join(GEOTEXT_HOME, 'traingmm.png'))
    Xtest = []
    test_means = []
    n_samples = len(testUsers)
    for u in testUsers:
        locationStr = testUsers[u]
        lat, lon = locationStr2Float(locationStr)
        Xtest.append([lat, lon])
    Xtest = np.asarray(Xtest)
    Ytest = clf.predict(Xtest)
    for i in Ytest:
        test_means.append(clf.means_[i])
    lossbycoordinates(test_means)
    
    Xdev = []
    dev_means = []
    n_samples = len(devUsers)
    for u in devUsers:
        locationStr = devUsers[u]
        lat, lon = locationStr2Float(locationStr)
        Xdev.append([lat, lon])
    Xdev = np.asarray(Xdev)
    Ydev = clf.predict(Xdev)
    for i in Ydev:
        dev_means.append(clf.means_[i])
    lossbycoordinates(dev_means)


      

def dbn():
    from nolearn.dbn import DBN
    createTrainDir(640)
    createTestDevDir('test')
    createTestDevDir('dev')
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor()
    '''
    We then configure a neural network with 300 hidden units, a learning rate of 0.3 and 
    a learning rate decay of 0.9, which is the number 
    that the learning rate will be multiplied with after each epoch.
    '''
    clf = DBN(
    [X_train.shape[1], 300, 25],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=1,
    )
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    print "Accuracy:", metrics.zero_one_score(Y_test, preds)
    print "Classification report:"
    print metrics.classification_report(Y_test, preds)
    loss(preds)

def pybi():
    createTrainDir(640)
    createTestDevDir('test')
    createTestDevDir('dev')
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor()
    from pybrain.datasets            import ClassificationDataSet
    from pybrain.utilities           import percentError
    from pybrain.tools.shortcuts     import buildNetwork
    from pybrain.supervised.trainers import BackpropTrainer
    from pybrain.structure.modules   import SoftmaxLayer    
    from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
    from scipy import diag, arange, meshgrid, where
    from numpy.random import multivariate_normal
    
    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    dimension = X_train.shape[1]
    
    # means = [(-1,0),(2,4),(3,1)]
    # cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
    trndata = ClassificationDataSet(dimension, 1, nb_classes=len(categories))
    tstdata = ClassificationDataSet(dimension, 1, nb_classes=len(categories))
    for i in range(0, n_train_samples):
        trndata.addSample(X_train[i].todense().tolist()[0], [Y_train[i]])
    for i in range(0, n_test_samples):
        tstdata.addSample(X_test[i].todense().tolist()[0], [Y_test[i]])
        
    # alldata = ClassificationDataSet(2, 1, nb_classes=3)
    '''
    for n in xrange(400):
        for klass in range(3):
            input = multivariate_normal(means[klass],cov[klass])
            print input
            alldata.addSample(input, [klass])
    tstdata, trndata = alldata.splitWithProportion( 0.25 )
    '''
    print type(trndata)
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()
    print "Number of training patterns: ", len(trndata)
    print "Input and output dimensions: ", trndata.indim, trndata.outdim
    print "First sample (input, target, class):"
    print trndata['input'][0], trndata['target'][0], trndata['class'][0]
    fnn = buildNetwork(trndata.indim, 100, trndata.outdim, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)


    for i in range(20):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(),
                                  trndata['class'])
        tstresult = percentError(trainer.testOnClassData(
               dataset=tstdata), tstdata['class'])
    
        print "epoch: %4d" % trainer.totalepochs, \
              "  train error: %5.2f%%" % trnresult, \
              "  test error: %5.2f%%" % tstresult

        '''
        out = out.reshape(X.shape)
        figure(1)
        ioff()  # interactive graphics off
        clf()   # clear the plot
        hold(True) # overplot on
        for c in [0,1,2]:
            here, _ = where(tstdata['class']==c)
            plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
        if out.max()!=out.min():  # safety check against flat field
            contourf(X, Y, out)   # plot the contour
        ion()   # interactive graphics on
        draw()  # update the plot
        ioff()
        show()

       '''
def dump_svm():
    print "dumping matrices in svmlight format..."
    create_directories(granularity=640, write=False)
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor()
    # svmlight labels start from 1 not zero
    for Y in (Y_train, Y_dev, Y_test):
        print Y.shape
        for i in range(0, Y.shape[0]):
                Y[i] = Y[i] + 1
    # dump_svmlight_file(X_train, Y_train, path.join(GEOTEXT_HOME, 'train.svmlight'))
    # dump_svmlight_file(X_dev, Y_dev, path.join(GEOTEXT_HOME, 'dev.svmlight'))
    # dump_svmlight_file(X_test, Y_test, path.join(GEOTEXT_HOME, 'test.svmlight'))
    with codecs.open(path.join(GEOTEXT_HOME, 'costMatrix.svmlight'), 'w') as inf:
        for i in range(0, costMatrix.shape[0]):
            for j in range(0, costMatrix.shape[1]):
                separator = ' '
                if j == (costMatrix.shape[1] - 1):
                    separator = '\n'
                inf.write(str(int(costMatrix[i, j])).strip() + separator)
def create_toy_cost_sensitive_data():
    n_samples = 800
    n_classes = 4
    Xs = []
    Ys = []
    Ls = []
    var = 1
    X_train = np.ndarray(shape=(n_samples, 2))
    Y_train = np.ndarray(shape=(n_samples, 1), dtype=int)
    costM = np.ndarray(shape=(4, 4))
    costM[0, 0] = 0
    costM[0, 1] = 1
    costM[0, 2] = 1
    costM[0, 3] = 1
    costM[1, 0] = 1
    costM[1, 1] = 0
    costM[1, 2] = 1
    costM[1, 3] = 1
    costM[2, 0] = 1
    costM[2, 1] = 1
    costM[2, 2] = 0
    costM[2, 3] = 1
    costM[3, 0] = 1
    costM[3, 1] = 1
    costM[3, 2] = 1
    costM[3, 3] = 0
    color_list = plt.cm.get_cmap()
    
    
    cent = 2
    centers = [[-cent, cent], [cent, cent], [-cent, -cent], [cent, -cent]]
    colors = ['b', 'k', 'g', 'r']
    
    for c in range(0, n_classes):
        for i in range(c * n_samples / n_classes, (c + 1) * n_samples / n_classes):
            x1 = random.gauss(centers[c][0], var)
            x2 = random.gauss(centers[c][1], var)
            # x2 = 0
            X_train[i, 0] = x1
            X_train[i, 1] = x2
            Xs.append(x1)
            Ys.append(x2)
            Y_train[i, 0] = c
            Ls.append(c)

    cmapname = "prism"
    mrkr = ','
    plt.subplot(6, 1, 1)
    plt.title('Original train data')
    plt.scatter(Xs, Ys, s=20, c=[colors[i] for i in Ls], marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None)
    # plt.show(block=False)
    # X_train = normalize(X_train, norm='l2', axis=1, copy=True)
    
    
    clf = SGDClassifier(loss='log')
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_train)
    preds1 = preds.tolist()
    plt.subplot(6, 1, 2)
    plt.title('SGD with log loss')
    plt.scatter(Xs, Ys, s=20, c=[colors[i] for i in preds1], marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None)

    clf = SGDClassifier(loss='hinge')
    # clf = LinearSVC(multi_class='ovr', class_weight='auto', C=1.0, loss='l2', penalty='l2', dual=True, tol=1e-3)
    # clf = LogisticRegression(penalty='l2')
    # clf = SGDClassifier(loss='log', learning_rate='optimal', n_iter=5)
    # clf = RidgeClassifier(tol=1e-2, solver="auto")
    # clf = RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=1e-2, class_weight=None, solver="auto")
    # clf = SVC(C=1.0, kernel='poly', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
    # clf = Perceptron(n_iter=50)
    # clf = PassiveAggressiveClassifier(n_iter=50)
    # clf = KNeighborsClassifier(n_neighbors=10)
    # clf = NearestCentroid()
    # clf = MultinomialNB(alpha=.01)
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_train)
    preds11 = preds.tolist()
    plt.subplot(6, 1, 3)
    plt.title('SGD with hinge loss')
    plt.scatter(Xs, Ys, s=20, c=[colors[i] for i in preds11], marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None)
    
    Y_train = np.array(Ls)
    preds2 = theano_fit_predict(X_train, Y_train, copy.deepcopy(X_train), copy.deepcopy(Y_train), costM=None)
    plt.subplot(6, 1, 4)
    plt.title('Theano SGD with log loss')
    plt.scatter(Xs, Ys, s=20, c=[colors[i] for i in preds2], marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None)

    preds3 = theano_fit_predict(X_train, Y_train, copy.deepcopy(X_train), copy.deepcopy(Y_train), costM=costM)
    plt.subplot(6, 1, 5)
    plt.title('Theano SGD with cost sensitive learnning cost(i, j) = 1 for all i!=j')
    plt.scatter(Xs, Ys, s=20, c=[colors[i] for i in preds3], marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None)
    # plt.text(10, 0,  str(costM))
    
    costM[1, 0] = 50
    costM[0, 1] = 50
    preds4 = theano_fit_predict(X_train, Y_train, copy.deepcopy(X_train), copy.deepcopy(Y_train), costM=costM)
    plt.subplot(6, 1, 6)
    plt.title('Theano SGD with cost sensitive learnning cost(0,1)=cost(1,0)=50 other misclassification costs = 1 for i!=j')
    plt.scatter(Xs, Ys, s=20, c=[colors[i] for i in preds4], marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None)
    plt.text(10, 0, str(costM))
    plt.show(block=True)

def matrix_test():
    a = np.random.rand(3, 2)
    b = np.random.rand(3, 2)
    b = b.min(axis=0)
    # b.reshape((3, 1))
    print a
    print '-------------'
    print b
    print '-------------'
    # print a * b
    print '-------------'
    # print np.sum(a * b , axis=1)
    print '-------------'
    print a / b
def chart_me():
    data = np.genfromtxt(path.join(GEOTEXT_HOME, 'cost-data/first-nll100or65-then-cost.txt'), dtype=float, delimiter=',')
    markers = ['m-', 'g--', 'g-', 'c-', 'r-', 'b-', 'r--', 'b--']
    labels = ['NLL cost', 'class cost', 'sample cost', 'error rate', 'mean', 'median', 'bayesian mean', 'bayesian median']
    costFunction = '-T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) - T.mean(T.sum(self.p_y_given_x * T.log(self.p_y_given_x), axis=1))'
    costFunction = 'Expected Class Cost: T.mean(T.sum(self.p_y_given_x * T.transpose(c[:, y]), axis=1))'
    costFunction = 'NLL: -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])'
    costFunction = 'Expected Sample Cost: T.mean(T.sum(self.p_y_given_x * sc, axis=1))'
    costFunction = 'Start with NLL then go with expected class cot'
    for i in range(1, data.shape[1]):
        x = data[:, 0]
        y = data[:, i]
        magnificient = ''
        if labels[i - 1] in ['NLL cost']:
            y = y * 100
            magnificient = ' 100x'
        if labels[i - 1] in ['error rate']:
            y = y * 100
        
        labels[i - 1] = labels[i - 1] + magnificient
        minIndex = np.argmin(y)
        plt.text(x[minIndex], y[minIndex], "min")
        plt.title(costFunction)
        plt.plot(x, y, markers[i - 1], label=labels[i - 1], linewidth=2)
    legend = plt.legend(loc='upper right', shadow=False, fontsize='small')
    # legend.get_frame().set_facecolor('#00FFCC')
    plt.show(block=True)

def pdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)
    
    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return float(part1 * np.exp(part2))

def mix_GP_LR():
    # read GP output
    with open(path.join(GEOTEXT_HOME, 'GP_results-True.pkl'), 'rb') as inf:
        means , vars = pickle.load(inf)
    
    print means.shape
    print vars.shape

    gp_probs = numpy.ndarray(shape=(means.shape[0], len(categories)))
    
    for i in range(0, means.shape[0]):
        latlon_mean = means[i]
        
    preds, probs, U_test = asclassification(granularity=640)
    
def wordDist():
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor(encoding=data_encoding)
    vectorizer = TfidfVectorizer(use_idf=False, norm='l2', binary=False, sublinear_tf=True, min_df=1, max_df=1.0, ngram_range=(1, 1), stop_words='english')
    wordsDist = {}
    with codecs.open(path.join(GEOTEXT_HOME, 'full_text_normalized.txt'), 'r', encoding=data_encoding) as inf:
        for line in inf:
            fs = line.split('\t')
            if len(fs) < 6:
                continue
            text = fs[5]
            lat = fs[3]
            lon = fs[4]
            latf, lonf = locationStr2Float(lat + ',' + lon)
            label = assignClass(latf, lonf)
            classLat = classLatMedian[str(label)]
            classLon = classLonMedian[str(label)]
            try:
                vectorizer.fit([text])
                print vectorizer.get_feature_names()
                for word in feature_names:
                    wordsDist[word] = [latf, lonf, classLat, classLon , label]
                    
            except:
                pass
            
            
def cross_validate():
    results = {}
    for partitionMethod in ['median', 'halfway', 'max_margin']:
        for bucketSize in [50, 100, 200, 300, 400, 500, 600]:
            initialize(partitionMethod=partitionMethod, granularity=bucketSize, encoding=data_encoding, write=True)
            preds, probs, U_test, meanTest, medianTest, meanDev, medianDev = asclassification(granularity=bucketSize, partitionMethod=partitionMethod)
            results[partitionMethod + '-' + str(bucketSize)] = 'Test: ' + str(meanTest) + ' ' + str(medianTest) + ' Dev: ' + str(meanDev) + str(medianDev)
    print results   
    # get the classifier output
    # mix them
    # evaluate them
def euclidean():
    a = np.random.randint(5, size=(4, 2))
    b = copy.deepcopy(a)
    # b = np.random.randint(5, size=(4, 2))
    m = 4
    p = 2
    n = 4

    aa = np.sum(a * a, 1).reshape(4, 1)
    bb = np.sum(b * b, 1).reshape(4, 1).transpose()
    # bb = bb.reshape(1, 3)
    # print aa.shape
    # print bb.shape
    # print aa
    # print bb

    # dd = AA(:,ones(1,n)) + BB(ones(1,m),:) - 2*A*B'
    dd = aa[:, np.zeros(n, dtype=int)] + bb[np.zeros(m, dtype=int), :] - 2 * np.dot(a, b.transpose())
    # print a
    # print b
    # print dd
    # print aa[:, np.zeros(n, dtype=int)]
    # print np.tile(aa, n)
    print bb[np.zeros(m, dtype=int), :]
    print np.tile(bb, (m, 1))

def extract_mentions(k=0):
    print "extracting mention information from text"
    global mentions
    text = ''
    #for user in trainUsers:
    #    text += userText[user].lower()
    text = ' '.join(trainText.values())
    text = text.lower()
    token_pattern = r"(?u)\b\w\w+\b"
    #token_pattern = "[\@\#]+\w\w+\b"
    token_pattern = re.compile(token_pattern)
    mentionsList = [word for word in token_pattern.findall(text) if word.startswith('user_')]
    #mentionsList = [word for word in token_pattern.findall(text)]
    mentionsDic = Counter(mentionsList)
    mentions = [word for word in mentionsDic if mentionsDic[word] > k]
def spams_groups(feature_names, X_train,type, k=0):
    if type=="mentions":
        print "building feature group memberships"
        print "the number of features is " + str(len(feature_names))    
        indices = [i for i,feature in enumerate(feature_names) if feature.startswith('user_')]
        
        ngroups = len(indices)
        print "the number of groups is " + str(ngroups)
        groupDesign = X_train[:,indices ]
        eta_g = np.ones(ngroups,dtype=float)
        groups = sparse.csc_matrix(np.zeros((ngroups,ngroups)),dtype = np.bool)
        groups_var = sparse.csc_matrix(X_train.transpose().dot(groupDesign), dtype=np.bool)
        graph = {'eta_g': eta_g,'groups' : groups,'groups_var' : groups_var}
        return graph
    elif type=="count":
        X_train2, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names2 = feature_extractor(use_mention_dictionary=False, min_df=1, max_df=1.0, norm=None, binary=True, use_idf=False, sublinear_tf=False)
        counts = X_train2.sum(axis=0)
        ngroups = counts.max() - counts.min() + 1
        print "the number of features is " + str(len(feature_names2))
        if len(feature_names)!=len(feature_names2):
            print "fatal error"
            sys.exit() 
        print "the number of groups is " + str(ngroups)
        #eta_g = np.ones(ngroups,dtype=float)
        #eta_g = np.arange(1, ngroups + 1,dtype=float)
        print "np.log + 0.5"
        eta_g = np.log(np.arange(1, ngroups + 1,dtype=float)) + 0.5
        groups = sparse.csc_matrix(np.zeros((ngroups,ngroups)),dtype = np.bool)
        groups_var = sparse.csc_matrix(np.zeros((len(feature_names),ngroups)),dtype = np.bool)
        for i in range(0, len(feature_names)):
            groups_var[i, int(counts[0, i])- counts.min()] = 1
        graph = {'eta_g': eta_g,'groups' : groups,'groups_var' : groups_var}
        return graph
        
        
    
def iterative_collective_classification(granularity=300, partitionMethod='median'):
    global feature_names

    # extract content features
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor( use_mention_dictionary=True)

    # classify
    preds, probs, U_test, meanTest, medianTest, meanDev, medianDev = classify(X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names, granularity, compute_dev=False, report_verbose=False)
    loss(preds, U_test)
    # build network based on mentions (a pairwise similarity measure)
    vectorizer = CountVectorizer(encoding=data_encoding, vocabulary=mentions, binary=True)
    X_train_relational, Y_train, U_train, X_dev_relational, Y_dev, U_dev, X_test_relational, Y_test, U_test, categories, feature_names_relational = feature_extractor(use_mention_dictionary=True, use_idf=False, norm=None, binary=True, sublinear_tf=False, min_df=1, BuildCostMatrices=False, vectorizer=vectorizer)
    # vstack train and dev/test data
    X_content = sparse.vstack([X_train, X_test]).tocsr()
    X_relational = sparse.vstack([X_train_relational, X_test_relational]).tocsr()
    Xt_relational = np.transpose(X_relational)
    # find the number of shared mentions for all pairs of users/samples 
    pairs = X_relational.dot(Xt_relational)
    # set the diagonal to zero (the number of shared mentions of a user with herself)
    #equal1_indices = pairs < 2
    #pairs[equal1_indices] = 0
    #pairs = threshold(pairs, 2, 100, 0)
    pairs.setdiag(0)
    content_relational_features = feature_names.tolist()
    for i in range(0, len(categories)):
        content_relational_features.append('new_relational_aggregated_feature' + str(i))
    feature_names = np.asarray(content_relational_features)
    n_iter = 1000
    for i in range(0, n_iter):
        print "iteration " + str(i)
        Y_train_reshaped = Y_train.reshape(Y_train.shape[0], 1)
        Y_test_reshaped = Y_test.reshape(Y_test.shape[0], 1)
        Y = csr_matrix(sparse.vstack([Y_train_reshaped, Y_test_reshaped])).todense()
        XClass = csr_matrix((X_relational.shape[0], len(categories)), dtype=int)
        # XClass matrix n_samples x n_classes where XClass[i, j] = 1 if sample i belongs to class j otherwise zero 
        XClass[np.arange(XClass.shape[0]), np.transpose(Y[np.arange(XClass.shape[0]), 0])] = 1
        # csr matrix n_samples x n_classes where n[i, j]= sum(friendshipStrength) for neighbors of i in class j  
        neighborhoodClassDistribution = csr_matrix(pairs.dot(XClass), shape=(pairs.shape[0], XClass.shape[1]), dtype=float)
        neighborhoodClassDistribution = normalize(neighborhoodClassDistribution, norm='l2', copy=False)
        # new train and test features
        new_content_relation_features = sparse.hstack([X_content, neighborhoodClassDistribution]).tocsr()
        X_train_content_relation = new_content_relation_features[0:len(U_train), :]
        X_test_content_relation = new_content_relation_features[len(U_train):, :]
        X_train_relational_aggregated = neighborhoodClassDistribution[0:len(U_train), :] 
        X_test_relational_aggregated = neighborhoodClassDistribution[len(U_train):, :]
        preds, probs, U_test, meanTest, medianTest, meanDev, medianDev = classify(X_train_content_relation, Y_train, U_train, X_dev, Y_dev, U_dev, X_test_content_relation, Y_test, U_test, categories, feature_names, granularity, compute_dev=False, report_verbose=False)
        loss(preds, U_test)
        Y_test = np.array(preds)
    Tracer()()
    # build the network using pariwise euclidean distance
    
    # iterate until convergance
        # produce aggregated relational features
        # combine the features
        # classify
    
    
def test_re():
    file = '/home/af/Downloads/GeoText.2010-10-12/full_text.txt'
    file = '/home/af/Downloads/Roller Dataset NA/training-big.txt'
    file = '/home/af/Downloads/data/twitter-world/twitter-world-training.data.txt'
    with codecs.open(file, 'r', encoding=data_encoding) as inf:
        text = inf.read()
        #print text
    #token_pattern = r"[\@\#]+\w\w+\b"
    #token_pattern = re.compile(token_pattern)
    #mentionsList = [word for word in token_pattern.findall(text)]

    #print a
    #mentionsList = [word for word in text.split() if word.startswith('@') or word.startswith('#')]
    mentionsList = []
    for word in text.split():
        if (word.startswith('@') or word.startswith('#')) :
            mentionsList.append(word)
    #print mentionsList[:-1]
    print len(mentionsList)

def spams_group_lasso():
    import spams
    import numpy as np
    import scipy.sparse as ssp
    myfloat=float
    
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor(use_mention_dictionary=False, min_df=1, max_df=1.0, norm=None)
    extract_mentions()
    #graph = spams_groups(feature_names,X_train,type="mentions", k=0)
    graph = spams_groups(feature_names,X_train,type="count", k=0)
    # X should be n_feature x n_sample in spams
    #X_train = normalize(X_train, norm='l2')
    X_train = X_train.todense()
    X = np.asfortranarray(X_train)
    X = spams.normalize(X)
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    #X = np.asfortranarray(X - np.tile(np.mean(X,0),(X.shape[0],1)),dtype=myfloat)
    #X = spams.normalize(X1)
    Y = np.asfortranarray(np.ceil(5 * np.random.random(size = (Y_train.shape[0],1))) - 1,dtype=myfloat)
    for i in range(0, Y.shape[0]):
        Y[i, 0] = Y_train[i, 0]
    
    # Multi-Class classification

    param = {'numThreads' : -1,'verbose' : True,
             'lambda1' : 0.05, 'it0' : 10, 'max_it' : 100,
             'L0' : 0.1, 'tol' : 1e-3, 'intercept' : False,
             'pos' : False}
    param['loss'] = 'multi-logistic'
    param['regul'] = 'graph'
    param['lambda1'] = 0.0000001

    print '\nFISTA + Multi-Class Logistic l1'
    print param
    nclasses = np.max(Y[:])+1
    W0 = np.zeros((X.shape[1],nclasses * Y.shape[1]),dtype=myfloat,order="FORTRAN")
    #(W, optim_info) = spams.fistaFlat(Y,X,W0,True,**param)
    (W, optim_info) = spams.fistaGraph(Y,X,W0,graph,True,**param)
    X_test = X_test.todense()
    results = np.dot(X_test, W)
    preds = np.argmax(results, axis=1)
    print "test results"
    loss(preds, U_test)
    print "development results"
    X_dev = X_dev.todense()
    results = np.dot(X_dev, W)
    preds = np.argmax(results, axis=1)
    loss(preds, U_dev)
    print "train results"
    results = np.dot(X_train, W)
    preds = np.argmax(results, axis=1)
    loss(preds, U_train)
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
    print "mindf 1  graph e-7"
    Tracer()()
    '''
    # Multi-Class classification

    param = {'numThreads' : -1,'verbose' : True,
             'lambda1' : 0.05, 'it0' : 10, 'max_it' : 200,
             'L0' : 0.1, 'tol' : 1e-3, 'intercept' : False,
             'pos' : False}
    param['loss'] = 'multi-logistic'
    param['regul'] = 'l1'
    param['lambda1'] = 0.01
    np.random.seed(0)
    m = 100;n = 200
    X = np.asfortranarray(np.random.normal(size = (m,n)))
    X = np.asfortranarray(X - np.tile(np.mean(X,0),(X.shape[0],1)),dtype=myfloat)
    X = spams.normalize(X)
    Y = np.asfortranarray(np.ceil(5 * np.random.random(size = (100,1))) - 1,dtype=myfloat)
    print '\nFISTA + Multi-Class Logistic l1'
    nclasses = np.max(Y[:])+1
    W0 = np.zeros((X.shape[1],nclasses * Y.shape[1]),dtype=myfloat,order="FORTRAN")
    (W, optim_info) = spams.fistaFlat(Y,X,W0,True,**param)
    
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
    Tracer()()
    # can be used of course with other regularization functions, intercept,...
    '''
def fabian_glasso():
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor(use_mention_dictionary=False, min_df=1, max_df=10, norm='l2')
    alpha = .1
    X = normalize(X_train).todense()
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    groups = np.r_[[0, 0], np.arange(X.shape[1] - 2)]
    print groups
    coef = group_lasso.group_lasso(X, Y_train, alpha, groups, max_iter=100, rtol=None, verbose=True)
    Tracer()()
def save_matlab():
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor(use_mention_dictionary=False, min_df=1, max_df=1.0)
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    Y_test = Y_test.reshape((Y_test.shape[0], 1))
    mmwrite(path.join(GEOTEXT_HOME, 'X_train.mtx'), X_train.todense())
    mmwrite(path.join(GEOTEXT_HOME, 'Y_train.mtx'), Y_train)
    mmwrite(path.join(GEOTEXT_HOME, 'X_test.mtx'), X_test.todense())
    mmwrite(path.join(GEOTEXT_HOME, 'Y_test.mtx'), Y_test)
#test_re()
# euclidean()
# cross_validate()
# chart_me()
# sys.exit()
# normalizeText()
initialize(partitionMethod='median', granularity=300, write=True)
#save_matlab()    
#fabian_glasso()
spams_group_lasso()

#iterative_collective_classification()
# wordDist()
# matrix_test()
# create_toy_cost_sensitive_data()
# plt.scatter(Xs, Ys, s=20, c=Ls, marker='o', cmap=pb.cm.get_cmap('prism', lut=None), norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None)
# plt.show(block=True)

# plt.savefig(path.join(GEOTEXT_HOME, 'toy.png'))
# dump_svm()            
# pybi()
# dbn()
# mixtureModel()
#asclassification(granularity=300, partitionMethod='median', use_mention_dictionary=False)
#classificationBench(granularity=300, partitionMethod='median', use_mention_dictionary=False)

# sgd_optimization_mnist(modelType='class')
# gd_optimization_mnist()
# test_mlp()
# print_class_coordinates()
# mix_GP_LR()
# wireless()

# loatGeolocationDataset()
# localizeGP()
# wirelessSGD()
# test_mlp()
