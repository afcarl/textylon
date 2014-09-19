'''
Created on 4 Sep 2014

@author: af
'''
import codecs
import matplotlib
import shutil
import os
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
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from sklearn.neighbors import NearestNeighbors
from time import time
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
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
pb.ion()
from GPy.core.gp import GP
import csv
from GPy.examples import regression


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


home = '/home/af/Downloads/GeoText.2010-10-12'
home = '/home/arahimi/geolocation/'
users_home = path.join(home, 'processed_data')
testfile = path.join(users_home, 'user_info.test')
devfile = path.join(users_home, 'user_info.dev')
trainfile = path.join(users_home, 'user_info.train')
userTextDirectory =  path.join(home, 'userText')
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

def readRecords():
    with codecs.open(path.join(home, 'full_text.txt'), 'r', 'latin') as inf:
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
    print 'number of '
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
    NeighborDirectory = '/home/af/Downloads/GeoText.2010-10-12/'
    # matplotlib.use('Agg')
    DATA_FOLDER = userTextDirectory
    #DATA_FOLDER = "/home/af/Downloads/review_polarity/txt_sentoken"
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



def partitionLocView():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    filename = '/home/af/Downloads/GeoText.2010-10-12/processed_data/meanminmax/160_clustered.train'
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
    plt.title('US Map of Twitter Users mean of min & max 160 person per cluster')
    plt.show() #pylab.show()            







def createTrainDir():
    #readlocationclusters
    
    filename = '/home/af/Downloads/GeoText.2010-10-12/processed_data/meanminmax/10_clustered.train'
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
    home = filename.replace(".train", "")
    shutil.rmtree(home)
    os.mkdir(home)
    i = 0
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
        
        class_dir = path.join(home, str(i).strip())
        
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
            for user in locusers:
                with codecs.open(path.join(class_dir, user), 'w', 'latin') as inf:
                    inf.write(userText[user])
    print "train directories created and class median and mean lat,lon computed. trainfile: " + filename
#createTrainDir()
def createTestDevDir(type='test'):
    print 'creating ' + type + ' collection.'
    t_home = path.join(users_home, type+'/'+type)
    shutil.rmtree(t_home, ignore_errors=True)
    os.mkdir(t_home)
    userCollection = {}
    if type=='test':
        userCollection = testUsers
    elif type=='dev':
        userCollection = devUsers
    else:
        print "fatal error in createTestDevDir type:" + type
        return;
    
    for user in userCollection:
        text = userText[user]
        with codecs.open(path.join(t_home, user), 'w', 'latin') as inf:
            inf.write(text)
#createTestDevDir('test')
#createTestDevDir('dev')

def size_mb(docs):
    return sum(len(s.encode('latin')) for s in docs) / 1e6



def classify():
    trainDir = '/home/af/Downloads/GeoText.2010-10-12/processed_data/meanminmax/10_clustered'
    testDir = '/home/af/Downloads/GeoText.2010-10-12/processed_data/test'
    data_train = load_files(trainDir, encoding='latin')
    target = data_train.target
    data_test = load_files(testDir, encoding='latin')

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
    
    
    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time()
    vectorizer = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, min_df=2, max_df=1.0, ngram_range=(1, 1))
    X_train = vectorizer.fit_transform(data_train.data)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()
    
    print("Extracting features from the test dataset using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(data_test.data)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()
    chi = False
    if chi:
        k = 1000
        print("Extracting %d best features by a chi-squared test" % 0)
        t0 = time()
        ch2 = SelectKBest(chi2, k=0)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        print("done in %fs" % (time() - t0))
        print()
        feature_names = np.asarray(vectorizer.get_feature_names())
    
    clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)
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

    #score = metrics.f1_score(y_test, pred)
    #print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

      
    sumMeanDistance = 0
    sumMedianDistance = 0
    for i in range(0, len(pred)):
        user = path.basename(data_test.filenames[i])
        location = userLocation[user].split(',')
        lat = float(location[0])
        lon = float(location[1])
        prediction = categories[pred[i]]
        medianlat = classLatMedian[prediction]  
        medianlon = classLonMedian[prediction]  
        meanlat = classLatMean[prediction] 
        meanlon = classLonMean[prediction]      
        
        sumMedianDistance = sumMedianDistance + distance(lat, lon, medianlat, medianlon)
        sumMeanDistance = sumMeanDistance + distance(lat, lon, meanlat, meanlon)
    averageMeanDistance = sumMeanDistance / float(len(pred))
    averageMedianDistance = sumMedianDistance / float(len(pred))
    print "Average mean distance is " + str(averageMeanDistance)
    print "Average median distance is " + str(averageMedianDistance)
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

    
    vectorizer = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, min_df=10, max_df=50000, ngram_range=(1, 1), stop_words='english')
    
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
def finalUserTextFile():
    fname = path.join(home, 'loctext.txt')
    with codecs.open(fname, 'w', 'latin1') as inf:
        pass
        #TODO
wireless()
readRecords()
print 'reading train, dev and test file'
users(trainfile, 'train')
users(devfile, 'dev')
users(testfile, 'test')
print 'total ' + str(len(userLocation)).strip() + " users."
#fillUserByLocation()
fillTextByUser()            

#loatGeolocationDataset()
localizeGP()
#wirelessSGD()