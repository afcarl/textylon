'''
Created on 4 Sep 2014

@author: af
'''
import codecs
import matplotlib as mpl
import shutil
import os
import itertools
from scipy import linalg
from sklearn import mixture
import copy
from math import radians, cos, sin, asin, sqrt
from sklearn import cross_validation
from sklearn import metrics
from  sklearn.datasets import load_files
from sklearn.decomposition import PCA, TruncatedSVD, NMF, SparsePCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.extmath import density
from sklearn.neighbors import NearestNeighbors
#from time import time
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from os import path
import math
#from datetime import datetime
import glob
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import  matplotlib.collections as collections
import matplotlib.ticker as ticker
import pylab as pb
from theano.tensor.basic import dmatrix
pb.ion()
from GPy.core.gp import GP
import csv
from GPy.examples import regression
from sklearn.linear_model.sgd_fast import Log

import numpy as np
import GPy
from GPy import kern, likelihoods
#from GPy.models_modules.gp_regression import GPRegression
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from numpy import float16, float32
from scipy.sparse import csr_matrix
import sys
from scipy import mean
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
GEOTEXT_HOME = ''
if len(sys.argv) == 1 or sys.argv[1] == 'pc':
    GEOTEXT_HOME = '/home/af/Downloads/GeoText.2010-10-12'
elif sys.argv[1] == 'server':
    GEOTEXT_HOME = '/home/arahimi/geolocation/'



#GEOTEXT_HOME = '/home/arahimi/Roller Dataset NA'
users_home = path.join(GEOTEXT_HOME, 'processed_data')
testfile = path.join(users_home, 'user_info.test')
devfile = path.join(users_home, 'user_info.dev')
trainfile = path.join(users_home, 'user_info.train')
userTextDirectory =  path.join(GEOTEXT_HOME, 'userText')
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
userText = {}
devClasses = {}
testClasses = {}
categories = []
costMatrix = None
X_train = None
X_dev = None
X_test = None
Y_train = None
Y_dev = None
Y_test = None
U_train = None
U_dev = None
U_test = None


def readGeoTextRecords():
    with codecs.open(path.join(GEOTEXT_HOME, 'full_text.txt'), 'r', 'latin') as inf:
        i = 0
        j = 0
        for line in inf:
            i += 1
            fs = line.split('\t')
            if len(fs)!=6:
                j += 1
                #print 'format error: ' + line + str(j)
                continue
            user = fs[0]
            ttime = fs[1]
            badloc = fs[2]
            latitude = fs[3].strip()
            longitude = fs[4].strip()
            #print longitude, latitude, badloc
            #if latitude > -75 or latitude<-125:
            #    continue
            #    pass
            #if longitude < 25 or longitude>50:
            #    continue
            #    pass
            lngs.append(longitude)
            ltts.append(latitude)
            text = fs[5].strip()
            
                
            #print time
            #time = datetime.strptime(time,'%Y-%m-%dT%H:%M:%S')
            #if user in userFirstTime:
            #    if time < userFirstTime[user]:
            #        userFirstTime[user] = time
            #        userLocation[user] = str(latitude).strip()+','+str(longitude).strip()
            #        userlon[user] = longitude
            #        userlat[user] = latitude
            #else:
            #    userFirstTime[user] = time
            #    userLocation[user] = str(latitude).strip()+','+str(longitude).strip()
            #    userlon[user] = longitude
            #    userlat[user] = latitude
            records.append((user, ttime, badloc, latitude, longitude , text))
    return records

def readRollerRecords():
    with codecs.open(path.join(GEOTEXT_HOME, 'full_text.txt'), 'r', 'latin') as inf:
        i = 0
        j = 0
        for line in inf:
            i += 1
            fs = line.split('\t')
            if len(fs)!=6:
                j += 1
                #print 'format error: ' + line + str(j)
                continue
            user = fs[0]
            ttime = fs[1]
            badloc = fs[2]
            latitude = fs[3].strip()
            longitude = fs[4].strip()
            #print longitude, latitude, badloc
            #if latitude > -75 or latitude<-125:
            #    continue
            #    pass
            #if longitude < 25 or longitude>50:
            #    continue
            #    pass
            lngs.append(longitude)
            ltts.append(latitude)
            text = fs[5].strip()
            
                
            #print time
            #time = datetime.strptime(time,'%Y-%m-%dT%H:%M:%S')
            #if user in userFirstTime:
            #    if time < userFirstTime[user]:
            #        userFirstTime[user] = time
            #        userLocation[user] = str(latitude).strip()+','+str(longitude).strip()
            #        userlon[user] = longitude
            #        userlat[user] = latitude
            #else:
            #    userFirstTime[user] = time
            #    userLocation[user] = str(latitude).strip()+','+str(longitude).strip()
            #    userlon[user] = longitude
            #    userlat[user] = latitude
            records.append((user, ttime, badloc, latitude, longitude , text))
    return records
        
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
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 

    # 6367 km is the radius of the Earth
    km = 6367 * c
    return km 

def users(file, type='train'):
    with codecs.open(file, 'r', 'latin') as inf:
        for line in inf:
            #print line
            fields = line.split()
            user = fields[0].strip()
            lat = str(float(fields[1])).strip()
            lon = str(float(fields[2])).strip()
            locStr = lat+','+lon
            userLocation[user] = locStr
            if type=='train':
                trainUsers[user] = locStr
            elif type =='test':
                testUsers[user] = locStr
            elif type == 'dev':
                devUsers[user] = locStr
    
#print 'reading train, dev and test file'
#users(trainfile, 'train')
#users(devfile, 'dev')
#users(testfile, 'test')
#print 'total ' + str(len(userLocation)).strip() + " users."            
def fillUserByLocation():
    print 'users indexed by location'
    #fill userLocations dictioanry if there are multiple users for a location they are separated by space
    for user in userLocation:
        loc = userLocation[user]
        if loc in locationUser:
            #print "Warning: we have multiple users with exactly the same location!"
            locationUser[loc] = locationUser[loc] + " " + user
        else:
            locationUser[loc] = user
    print "the number of users/distinct locations is " + str(len(locationUser))
#fillUserByLocation()


def fillTextByUser(writeUserTexts=False):
        for record in records:
            user, ttime, badloc, latitude, longitude, text = record
            if user in userText:
                userText[user] = userText[user] + '\n' + text
            else:
                userText[user] = text
        if writeUserTexts:
            for user in userText:
                with codecs.open(path.join(userTextDirectory, user), 'w', 'latin') as outf:
                    outf.write(userText[user])



  

def textSimilarity():
    NeighborDirectory = GEOTEXT_HOME
    # matplotlib.use('Agg')
    DATA_FOLDER = userTextDirectory
    #DATA_FOLDER = "/GEOTEXT_HOME/af/Downloads/review_polarity/txt_sentoken"
    K_FOLD = 10
    data_target = load_files(DATA_FOLDER, encoding='latin')
    filenames = data_target.filenames
    DO_PCA = True
    DO_SPARSEPCA = False
    Reduction_D = 100
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
    t0 = 0
    vectorizer = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, min_df=2, max_df=0.2, ngram_range=(1, 1), stop_words='english')
    
    # vectorizer = CountVectorizer(min_df=2, max_df=1.0, ngram_range=(1, 4))
    # the output of the fit_transform (x_train) is a sparse csc matrix.
    data = vectorizer.fit_transform(data_target.data)
    print data.dtype
    data = csr_matrix(data, dtype=float32)
    print data.dtype
    duration = 1
    print("done in %fs at %0.3fMB/s" % (duration, data_size_mb / duration))
    print("n_samples: %d, n_features: %d" % data.shape)
    print()
    
    
    
    if DO_PCA:
        print("dimension reduction pca with d=%d" % Reduction_D)
        pca = PCA(n_components=Reduction_D, copy=True, whiten=False)
        print type(data)
        data = pca.fit_transform(data.todense())
    if DO_SPARSEPCA:
        print("dimension reduction sparsepca with d=%d" % Reduction_D)
        spca = SparsePCA(Reduction_D)
        data = spca.fit_transform(data.toarray())
    if DO_SVD:
        print("dimension reduction svd with d=%d" % Reduction_D)
        svd = TruncatedSVD(n_components=Reduction_D, algorithm="randomized", n_iterations=5, random_state=None, tol=0)
        data = svd.fit_transform(data)
    if DO_NMF:
        print("dimension reduction nmf with d=%d" % Reduction_D)
        nmf = NMF(n_components=Reduction_D)
        data = nmf.fit_transform(data)
    
    DO_CHI = False
    if DO_CHI:
        print("Extracting best features by a chi-squared test")
        ch2NumFeatures = 1000 
        ch2 = SelectKBest(chi2, k=ch2NumFeatures)
        #print vectorizer.get_stop_words()
        data = ch2.fit_transform(data, target)
        #print data

    
    KNN = 10
    nn = NearestNeighbors(n_neighbors=KNN + 1, algorithm='ball_tree').fit(data)
    #query and data are the same so every node is counted as its most similar here
    distances, indices = nn.kneighbors(data)
    with codecs.open(path.join(NeighborDirectory, 'neighbors.txt'), 'w', 'latin') as outf:
        nodeIndex = -1
        nodeNeighbors = []
        for neighbors in indices:
            nodeIndex += 1
            outf.write(path.basename(filenames[nodeIndex])+' ')
            for neighbor in neighbors:
                if neighbor == nodeIndex:
                    continue
                else:
                    outf.write(path.basename(filenames[neighbor])+' ')
            outf.write('\n')
        
def plot_points():
    
    #N = 50
    #x = np.random.rand(N)
    #y = np.random.rand(N)
    x = []
    y = []
    for user in userLocation:
        if userlon[user] > -70:
            continue
        y.append(userlat[user])
        x.append(userlon[user])
    colors = np.random.rand(15)
    
    area = 1 # 0 to 15 point radiuses
    
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.show()

            
def merge_text():
    files = glob.glob(userTextDirectory+'/userText/*')
    texts = {}
    for file in files:
        with codecs.open(file, 'r', 'latin') as inf:
            t = ''
            for line in inf:
                t = t +' '+ line.strip()
            texts[path.basename(file)] = t
    with codecs.open(path.join(userTextDirectory, 'linie.txt'), 'w', 'latin') as outf:
        for t in texts:
            outf.write( t + ' ||| ' + texts[t] + '\n')
        
#merge_text()



def partitionLocView(granularity=10):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    filename = '/home/af/Downloads/GeoText.2010-10-12/processed_data/'+str(granularity).strip()+'_clustered.train'
    allpoints = []
    allpointsMinLat = []
    allpointsMaxLat = []
    allpointsMinLon = []
    allpointsMaxLon = []
    with codecs.open(filename, 'r', 'utf-8') as inf:
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
    y= []
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
        ax.set_xlim([-125, -60]) #pylab.xlim([-400, 400])
        ax.set_ylim([25, 50])
         
    #colors = np.random.rand(15)
    
    #area = 1 # 0 to 15 point radiuses
    
    #plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    #ax.set_xlim([-400, -380]) #pylab.xlim([-400, 400])
    #ax.set_ylim([-400, -380]) #pylab.ylim([-400, 400])
    #patches = []
    #polygon = plt.Rectangle((-400, -400), 10, 10, color='yellow') #Rectangle((-400, -400), 10, 10, color='y')
    #patches.append(polygon)
    
    #pol2 = plt.Rectangle((-390, -390), 10, 10, facecolor='yellow', edgecolor='violet', linewidth=2.0)
    #ax.add_artist(pol2)
    
    
    #p = collections.PatchCollection(patches) #, cmap=matplotlib.cm.jet)
    #ax.add_collection(p)
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(20)) # (MultipleLocator(20)) 
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(20)) # (MultipleLocator(20)) 
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('US Map of Twitter Users mean of min & max '+str(granularity).strip()+' person per cluster')
    plt.savefig(filename+'.jpg')
    plt.show() #pylab.show()            







def createTrainDir(granularity=10, create_dir=False):
    global costMatrix
    #readlocationclusters
    filename = path.join(GEOTEXT_HOME, 'processed_data/'+str(granularity).strip()+'_clustered.train')
    allpoints = []
    allpointsMinLat = []
    allpointsMaxLat = []
    allpointsMinLon = []
    allpointsMaxLon = []
    with codecs.open(filename, 'r', 'latin') as inf:
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
        #create a directory
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
            #find user(s) in that collection
            locationStr = location[0]+','+location[1]
            userstr = locationUser[locationStr]
            if not userstr:
                print "fatal error: something is wrong, no user for this location: " + locationStr 
            locusers = []
            if " " in userstr:
                #multiple users separated by space in this location
                locusers = userstr.split()
            else:
                #just one single user in this location
                locusers.append(userstr.strip())
            # for each user in this location find the text
            # groupbyusersText should be true for this to work
            #print "writing user texts in their corresponding geographical class in: " + class_dir
            if create_dir:
                for user in locusers:
                    with codecs.open(path.join(class_dir, user), 'w', 'latin') as inf:
                        inf.write(userText[user])
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
    print "building cost matrix..."
    costMatrix = np.ndarray(shape=(len(classLatMedian), len(classLatMedian)), dtype=float)
    for i in classLatMedian:
        lat = classLatMedian[i]
        lon = classLonMedian[i]
        for j in classLatMedian:
            lat2 = classLatMedian[j]
            lon2 = classLonMedian[j]
            cost = distance(lat, lon, lat2, lon2)
            costMatrix[i, j] = cost
    print "Ideal mean dev distance is " + str(mean(devDistances))
    print "Ideal median dev distance is " + str(median(devDistances))
    
    print "Ideal mean test distance is " + str(mean(testDistances))
    print "Ideal median test distance is " + str(median(testDistances))
        
#createTrainDir()
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
    t_home = path.join(users_home, type+'/')
    shutil.rmtree(t_home, ignore_errors=True)
    os.mkdir(t_home)
    userCollection = {}
    if type=='test':
        userCollection = testUsers
        userClasses = testClasses
    elif type=='dev':
        userCollection = devUsers
        userClasses = devClasses
    else:
        print "fatal error in createTestDevDir type:" + type
        return;
    
    for user in userCollection:
        userClass = userClasses[user]
        text = userText[user]
        classDir = path.join(t_home, str(userClass).strip())
        if not os.path.exists(classDir):
            os.makedirs(classDir)
        with codecs.open(path.join(classDir, user), 'w', 'latin') as inf:
            inf.write(text)
#createTestDevDir('test')
#createTestDevDir('dev')
def create_directories(granularity, write=False):
    createTrainDir(granularity, write)
    if write:
        createTestDevDir(type='dev')
        createTestDevDir(type='test')
def size_mb(docs):
    return sum(len(s.encode('latin')) for s in docs) / 1e6

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
        #gaussian mixture model
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
    #averageMeanDistance = sumMeanDistance / float(len(preds))
    #averageMedianDistance = sumMedianDistance / float(len(preds))
    #print "Average mean distance is " + str(averageMeanDistance)
    #print "Average median distance is " + str(averageMedianDistance)
    print "Mean distance is " + str(mean(distances))
    print "Median distance is " + str(median(distances))

def loss(preds, U_test, loss='median'):
    if len(preds) != len(testUsers): 
        print "The number of test sample predictions is: "+ str(len(preds))
        print "The number of test samples is: " + str(len(testUsers))
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
    #print "Average distance from class mean is " + str(averageMeanDistance)
    #print "Average distance from class median is " + str(averageMedianDistance)
    print "Mean distance is " + str(mean(distances))
    print "Median distance is " + str(median(distances))
    
    if loss == 'median':
        return medianDistance
    elif loss == 'mean':
        return averageMedianDistance  
def lossbycoordinates(coordinates):
    if len(coordinates) != len(testUsers): 
        print "The number of test sample predictions is: "+ str(len(coordinates))
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
        
    #print "Average distance from class mean is " + str(averageMeanDistance)
    #print "Average distance from class median is " + str(averageMedianDistance)
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
def feature_extractor(encoding='utf-8'):
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
    trainDir = path.join(GEOTEXT_HOME, 'processed_data/train')
    testDir = path.join(GEOTEXT_HOME, 'processed_data/test')
    devDir = path.join(GEOTEXT_HOME, 'processed_data/dev')
    
    data_train = load_files(trainDir, encoding=encoding)
    target = data_train.target
    data_test = load_files(testDir, encoding=encoding)
    data_dev = load_files(devDir, encoding=encoding)
    
    categories = data_train.target_names
    
    U_train = [path.basename(data_train.filenames[i]) for i in range(0, len(data_train.filenames))]
    U_test = [path.basename(data_test.filenames[i]) for i in range(0, len(data_test.filenames))]
    U_dev = [path.basename(data_dev.filenames[i]) for i in range(0, len(data_dev.filenames))]
    
    def size_mb(docs):
        return sum(len(s.encode(encoding)) for s in docs) / 1e6
    
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
    vectorizer = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, min_df=2, max_df=1.0, ngram_range=(1, 1), stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
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
        print("Extracting %d best features by a chi-squared test" % 0)
        t0 = time.time()
        ch2 = SelectKBest(chi2, k=k)
        X_train = ch2.fit_transform(X_train, Y_train)
        X_test = ch2.transform(X_test)
        X_dev = ch2.transform(X_dev)
        print("done in %fs" % (time.time() - t0))
        print()
        #feature_names = np.asarray(vectorizer.get_feature_names())
    feature_names = np.asarray(vectorizer.get_feature_names())
    return X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names
    
def abod(probs, preds, U_test):
    print "running abod to find the Approximate Bayes-Optimal Decision..."
    n_samples = probs.shape[0]
    n_categories = probs.shape[1]
    assert n_categories == len(categories), "fatal error: n_categories is not equal to len(categories) in abod"
    preds2 = []
    for s in range(0, n_samples):
        minCost = 1000000000.0
        minCostClass = -1
        #select top n prob indices
        for cf in sorted(range(len(probs[s])), key=lambda i: probs[s, i])[-3:]:
            #print probs[s]
            cost = 0
            for co in range(0, n_categories):
                cost = cost + probs[s, co] * costMatrix[cf, co]
            if cost < minCost:
                minCost = cost
                minCostClass = cf
        preds2.append(minCostClass)
    loss(preds2, U_test)
    return preds2
    
def classify(granularity=10):
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor(encoding='latin1')

    #clf = LinearSVC(multi_class='ovr', class_weight='auto', C=1.0, loss='l2', penalty='l2', dual=True, tol=1e-3)
    #clf = LogisticRegression(penalty='l2')
    clf = SGDClassifier(loss='log', learning_rate='optimal', n_iter=5)
    #clf = RidgeClassifier(tol=1e-2, solver="auto")
    #clf = RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=1e-2, class_weight=None, solver="auto")
    #clf = SVC(C=1.0, kernel='poly', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
    #clf = Perceptron(n_iter=50)
    #clf = PassiveAggressiveClassifier(n_iter=50)
    #clf = KNeighborsClassifier(n_neighbors=10)
    #clf = NearestCentroid()
    #clf = MultinomialNB(alpha=.01)
    print('_' * 80)
    print("Training: ")
    print(clf)
    
    t0 = time.time()
    clf.fit(X_train, Y_train)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time.time()
    preds = clf.predict(X_test)
    
    #scores = clf.decision_function(X_test)
    probs = clf.predict_proba(X_test)
    #print preds.shape
    test_time = time.time() - t0
    print("test time:  %0.3fs" % test_time)
    
    print("classification report:")
    print(metrics.classification_report(Y_test, preds, target_names=categories))
    print("confusion matrix:")
    print(metrics.confusion_matrix(Y_test, preds))
    score = metrics.f1_score(Y_test, preds)
    print("f1-score:   %0.3f" % score)
    score = metrics.accuracy_score(Y_test, preds)
    print("Accuracy score:   %0.3f" % score)
    
    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        print("top 10 keywords per class:")
        for i, category in enumerate(categories):
            top10 = np.argsort(clf.coef_[i])[-10:]
            print("%s: %s" % (category, " ".join(feature_names[top10])))
    loss(preds, U_test)
    #loss(preds)
    #evaluate(preds,U_test, categories, None)
    abod(probs, preds, U_test)
   
#classify()

def loadGPData(DO_SVD=True, Reduction_D=100):
    data = {}
    
    trainlats = []
    trainlongs = []
    traintexts = []
    trainlocs = []
    
    testlats = []
    testlongs = []
    testtexts = []
    testlocs = []
    
    for user in trainUsers:
        loc = trainUsers[user]
        latlon = loc.split(',')
        lat = float(latlon[0])
        lon = float(latlon[1])
        text = userText[user]
        trainlats.append(lat)
        trainlongs.append(lon)
        traintexts.append(text)
        trainlocs.append([lat, lon])

    for user in testUsers:
        loc = testUsers[user]
        latlon = loc.split(',')
        lat = float(latlon[0])
        lon = float(latlon[1])
        text = userText[user]
        testlats.append(lat)
        testlongs.append(lon)
        testtexts.append(text)
        testlocs.append([lat, lon])

    
    vectorizer = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, min_df=10, max_df=1.0, ngram_range=(1, 1), stop_words='english')
    
    print 'vectorizing train and test data...'
    X_train = vectorizer.fit_transform(traintexts)
    print("X_train: n_samples: %d, n_features: %d" % X_train.shape)
    X_test = vectorizer.transform(testtexts)
    print("X_test: n_samples: %d, n_features: %d" % X_test.shape)
    
    if DO_SVD:
        print("dimension reduction svd with d=%d" % Reduction_D)
        svd = TruncatedSVD(n_components=Reduction_D, algorithm="randomized", n_iterations=5, random_state=None, tol=0)
        X_train = svd.fit_transform(X_train)
        X_test = svd.transform(X_test)
        print("dimension reduction finished.")
        
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
        
    
    
def localizeGP(max_iters=100, kernel=None, optimize=True, plot=False):
    """Predict the location of a robot given wirelss signal strength readings."""
    data = loadGPData()
    #data = GPy.util.datasets.robot_wireless()
    print data
    # create simple GP Model
    m = GPy.models.GPRegression(data['Y'], data['X'], kernel=kernel)

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
        
    sumDist = 0
    for i in range(0, Xpredict.shape[0]):
        lat1 = Xpredict[i][0]
        lon1 = Xpredict[i][1]
        lat2 = data['Xtest'][i][0]
        lon2 = data['Xtest'][i][1]
        sumDist += distance(lat1, lon1, lat2, lon2)
    averageDist = float(sumDist) / Xpredict.shape[0]
    print "average distance is: " + str(averageDist)
    #sse = ((data['Xtest'] - Xpredict)**2).sum()
    #aae = np.absolute(data['Xtest'] - Xpredict).sum()
    print m
    #print('Sum of squares error on test data: ' + str(sse))
    #print('average absolute error on test data: ' + str(aae))
    if plot:
        fig = pb.figure(None)
        pb.title('')
        raw_input()
    return m
def wireless(max_iters=100, kernel=None, optimize=True, plot=True):
    """Predict the location of a robot given wirelss signal strength readings."""
    #data = loadGPData()
    data = GPy.util.datasets.robot_wireless()
    print data
    # create simple GP Model
    m = GPy.models.GPRegression(data['Y'], data['X'], kernel=kernel, normalize_X=True, normalize_Y=True)

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
        

    #sse = ((data['Xtest'] - Xpredict)**2).sum()
    aae = np.absolute(data['Xtest'] - Xpredict).sum()
    print m
    #print('Sum of squares error on test data: ' + str(sse))
    print('average absolute error on test data: ' + str(aae))
    if plot:
        fig = pb.figure(None)
        pb.title('')
        raw_input()
    return m

def wirelessSGD(max_iters=100, kernel=None, optimize=True, plot=True):
    """Predict the location of a robot given wirelss signal strength readings."""
    data = loadGPData()
    #data = GPy.util.datasets.robot_wireless()
    print data
    # create simple GP Model
    m = GPy.models.GPMultioutputRegression(data['Y'], data['X'],  normalize_X=True, normalize_Y=True)

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
        

    #sse = ((data['Xtest'] - Xpredict)**2).sum()
    aae = np.absolute(data['Xtest'] - Xpredict).sum()
    print m
    #print('Sum of squares error on test data: ' + str(sse))
    print('average absolute error on test data: ' + str(aae))
    if plot:
        fig = pb.figure(None)
        pb.title('')
        raw_input()
    return m
def finalUserTextFile(home):
    fname = path.join(home, 'loctext.txt')
    with codecs.open(fname, 'w', 'latin1') as inf:
        pass
        #TODO
readGeoTextRecords()
print 'reading train, dev and test file'
users(trainfile, 'train')
users(devfile, 'dev')
users(testfile, 'test')
print 'total ' + str(len(userLocation)).strip() + " users."
fillUserByLocation()
fillTextByUser()            

def asclassification(granularity=10):    
    create_directories(granularity, write=False)
    classify(granularity)
    partitionLocView(granularity)

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

    def negative_log_likelihood(self, y, c):
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
        #print T.diag(T.dot(self.p_y_given_x[T.arange(y.shape[0]), 0:24], c[0:24, y]))
        
        return T.mean(T.diag(T.dot(self.p_y_given_x, c[:, y])))
        #return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        

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
    create_directories(granularity=640, write=False)
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names  = feature_extractor(encoding='latin1')
    test_set = (X_test.toarray(), Y_test)
    valid_set = (X_dev.toarray(), Y_dev)
    train_set = (X_train.toarray(), Y_train)
    #cost_set = (costMatrix, Y_test)

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

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
    
    #missclassificationCostMatrix, dummyvar = shared_dataset(cost_set)
    missclassificationCostMatrix = theano.shared(costMatrix, borrow=True)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y), Y_test, missclassificationCostMatrix]
    return rval
def sgd_optimization_mnist(learning_rate=0.2, n_epochs=100000,
                           dataset='mnist.pkl.gz',
                           batch_size=1):
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
    c = T.matrix('c') #misclassification cost matrix

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    #classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
    classifier = LogisticRegression(input=x, n_in=train_set_x.get_value(borrow=True).shape[1], n_out=25)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y, c)

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
    best_validation_loss = numpy.inf
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

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

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
                    
                    abod(all_probs,best_predictions, U_test)

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))
    
   
def gd_optimization_mnist(learning_rate=0.01, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
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



    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    #classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
    classifier = LogisticRegression(input=x, n_in=train_set_x.get_value(borrow=True).shape[1], n_out=25)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x,
                y: test_set_y})

    validate_model = theano.function(inputs=[],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x,
                y: valid_set_y})
    
    predict = theano.function(inputs=[],
            outputs=[classifier.y_pred, classifier.errors(y)],
            givens={
                x: test_set_x,
                y: test_set_y})

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
    train_model = theano.function(inputs=[],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x,
                y: train_set_y})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'


    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    best_predictions = []
    start_time = time.clock()

    epoch = 0
    while (epoch < n_epochs):
        epoch = epoch + 1

        minibatch_avg_cost = train_model()


        # compute zero-one loss on validation set
        validation_losses = [validate_model()]
        this_validation_loss = numpy.mean(validation_losses)

        print('epoch %i, validation error %f %%' % \
            (epoch, 
            this_validation_loss * 100.))
        
        test_losses = [test_model()]
        test_score = numpy.mean(test_losses)

        print(('     epoch %i, test error of' ' model %f %%') %(epoch,test_score * 100.))
        pred, err = predict()
        print "test error is: "+ str(err)
        predictions = copy.deepcopy(pred)
        loss(predictions)
        
        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
            #improve patience if loss improvement is good enough
            best_validation_loss = this_validation_loss
            # test it on the test set
            best_test_loss = [test_model()]
            test_score = numpy.mean(best_test_loss)
            print(('     epoch %i, test error of best' ' model till now %f %%') %(epoch,test_score * 100.))
            pred, best_err = predict()
            print "test error is: "+ str(best_err)
            best_predictions = copy.deepcopy(pred)
            loss(best_predictions)
            


    print "best test error: " + str(best_err)
    loss(best_predictions)
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))
    


    

    
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
    print test_set_y
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
    #classifier = MLP(rng=rng, input=x, n_in= 28 * 28, n_hidden=n_hidden, n_out=10)
    classifier = MLP(rng=rng, input=x, n_in=train_set_x.get_value(borrow=True).shape[1],
                     n_hidden=n_hidden, n_out=25)

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
    
    predict = theano.function([], classifier.logRegressionLayer.y_pred,
                   givens={x: test_set_x})

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
        loss(predict())
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
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
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
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names  = feature_extractor(encoding='latin1')

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
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
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
    dev_means= []
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
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names  = feature_extractor(encoding='latin1')
    '''
    We then configure a neural network with 300 hidden units, a learning rate of 0.3 and 
    a learning rate decay of 0.9, which is the number 
    that the learning rate will be multiplied with after each epoch.
    '''
    clf = DBN(
    [X_train.shape[1],300, 25],
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
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names  = feature_extractor(encoding='latin1')
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
    
    #means = [(-1,0),(2,4),(3,1)]
    #cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
    trndata = ClassificationDataSet(dimension,1, nb_classes=len(categories))
    tstdata = ClassificationDataSet(dimension,1, nb_classes=len(categories))
    for i in range(0, n_train_samples):
        trndata.addSample(X_train[i].todense().tolist()[0], [Y_train[i]])
    for i in range(0, n_test_samples):
        tstdata.addSample(X_test[i].todense().tolist()[0], [Y_test[i]])
        
    #alldata = ClassificationDataSet(2, 1, nb_classes=3)
    '''
    for n in xrange(400):
        for klass in range(3):
            input = multivariate_normal(means[klass],cov[klass])
            print input
            alldata.addSample(input, [klass])
    tstdata, trndata = alldata.splitWithProportion( 0.25 )
    '''
    print type(trndata)
    trndata._convertToOneOfMany( )
    tstdata._convertToOneOfMany( )
    print "Number of training patterns: ", len(trndata)
    print "Input and output dimensions: ", trndata.indim, trndata.outdim
    print "First sample (input, target, class):"
    print trndata['input'][0], trndata['target'][0], trndata['class'][0]
    fnn = buildNetwork( trndata.indim, 100, trndata.outdim, outclass=SoftmaxLayer )
    trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)


    for i in range(20):
        trainer.trainEpochs( 1 )
        trnresult = percentError( trainer.testOnClassData(),
                                  trndata['class'] )
        tstresult = percentError( trainer.testOnClassData(
               dataset=tstdata ), tstdata['class'] )
    
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
def test_matrix():
    y = np.ndarray(shape=(7,5), dtype=float)
    print y
test_matrix()
#pybi()
#dbn()
#mixtureModel()
#asclassification(640)
sgd_optimization_mnist()
#gd_optimization_mnist()
#test_mlp()
#print_class_coordinates()
'''
wireless()

#loatGeolocationDataset()
localizeGP()
#wirelessSGD()

'''