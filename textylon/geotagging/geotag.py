'''
Created on 4 Sep 2014

@author: af
'''
import codecs
import matplotlib as mpl
from scipy.sparse.lil import lil_matrix
from sklearn.feature_extraction import stop_words
from _collections import defaultdict
from sklearn.decomposition.factor_analysis import FactorAnalysis
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
import string
import networkx as nx
import matplotlib.patches as mpatches
from datetime import datetime
import sets
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.decomposition import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, strip_accents_ascii, strip_accents_unicode
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import linear_model
from sklearn.linear_model import Perceptron
from sklearn.datasets import dump_svmlight_file
from diagrams import *
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC, SVC
from IPython.core.debugger import Tracer
from sklearn.utils.extmath import density
import scipy.sparse as sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import SelectKBest
#this is just for printing colored in shell, you don't want it you comment it and its init line
from colorama import init, Fore, Back, Style
import logging
init()
import group_lasso 
# from extract import get_tokens
# from time import time
from math import sqrt
from matplotlib.ticker import NullFormatter
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from os import path
import math
from datetime import datetime
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
from scipy.sparse import csr_matrix, coo_matrix
from math import radians, sin, cos, sqrt, asin
import sys
from scipy import mean
__docformat__ = 'restructedtext en'
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

import cPickle
import gzip
import os
import sys
import time
import pickle
import numpy
import nltk
import theano
import theano.tensor as T
print str(datetime.now())
script_start_time = time.time()

def downsize_train(populationSize=6713, sampleSize=3356):
    population_lines = xrange(0, populationSize)
    sample_lines = random.sample(population_lines, sampleSize)
    line_number = 0 
    with codecs.open(trainfile+'-6713', 'r', data_encoding) as inf:
        with codecs.open(trainfile + '-' + str(sampleSize), 'w', data_encoding) as outf:
            for line in inf:
                if line_number in sample_lines:
                    outf.write(line)
                line_number += 1
            
            
def extract_entity_names(t):
    entity_names = []
    
    if hasattr(t, 'node') and t.node:
        if t.node == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                childentities = extract_entity_names(child)
                if childentities == None:
                    childentities = []
                entity_names.extend(childentities)
        if entity_names == None:
            entity_names = []            
        return entity_names
def get_named_entities(sample):
    sentences = nltk.sent_tokenize(sample)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
    entity_names = []
    for tree in chunked_sentences:
        # Print results per sentence
        # print extract_entity_names(tree)
        
        treeentities = extract_entity_names(tree)
        if treeentities is None:
            treeentities = []
        entity_names.extend(treeentities)
     
    # Print all entity names
    # print entity_names
     
    # Print unique entity names
    return entity_names      
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
    '''
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a)) 

    # 6367 km is the radius of the Earth
    km = 6367 * c
    '''
    
    #downloaded from http://rosettacode.org/wiki/Haversine_formula#Python
    R = 6372.8 # Earth radius in kilometers
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))
    
    return R * c
    #return km 

def users(file, type='train', write=False, readText=True, downSampleTextCoefficient=1.0):
    global testUsers
    global trainUsers
    global devUsers
    global userLocation
    global testText
    global devText
    global trainText
    global locationUser
    if readText:
        print("Text is being read.")
        if downSampleTextCoefficient < 1.0:
            print("Text is being downSampled with coefficient %d" %(downSampleTextCoefficient))
    with codecs.open(file, 'r', encoding=data_encoding) as inf:
        for line in inf:
            # print line
            fields = line.split('\t')
            if len(fields) != 4:
                print fields
            user = fields[0].strip()
            lat = fields[1]
            lon = fields[2]
            if readText:
                text = fields[3].strip()
                if downSampleTextCoefficient < 1.0:
                    text = text[0: int(len(text) * downSampleTextCoefficient)]
            if TEXT_ONLY and readText:
                text = ' '.join([t for t in text.split() if not t.startswith('@')])
            locStr = lat + ',' + lon
            userLocation[user] = locStr
            if type == 'train':
                trainUsers[user] = locStr
                if readText:
                    trainText[user] = text
                users_in_loc = locationUser.get(locStr, [])
                users_in_loc.append(user)
                locationUser[locStr] = users_in_loc
                '''
                if locStr in locationUser:
                    # print "Warning: we have multiple users with exactly the same location!"
                    locationUser[locStr] = locationUser[locStr] + " " + user
                else:
                    locationUser[locStr] = user
                '''
            elif type == 'test':
                testUsers[user] = locStr
                if readText:
                    testText[user] = text
            elif type == 'dev':
                devUsers[user] = locStr
                if readText:
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

            

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    import matplotlib.cm as cmx
    color_norm  = matplotlib.colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def partitionLocView(granularity, partitionMethod, convexhull=False):
    fig = plt.figure() #figsize=(4,4.2)
    points_np_arr = []
    print fig.get_size_inches()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    filename = users_home + '/' + str(granularity).strip() + '_' + partitionMethod + '_clustered.train'
    point_print_percent = 1.0
    print_lats = []
    print_lons = []
    allpoints = []
    allpointsMinLat = []
    allpointsMaxLat = []
    allpointsMinLon = []
    allpointsMaxLon = []
    print_classes = []
    with codecs.open(filename, 'r', encoding=data_encoding) as inf:
        class_line_number = -1
        for line in inf:
            class_line_number += 1
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
                if random.random() < point_print_percent:
                    print_lats.append(lat)
                    print_lons.append(lon)
                    print_classes.append(class_line_number)
            allpointsMinLat.append(minlat)
            allpointsMaxLat.append(maxlat)
            allpointsMaxLon.append(maxlon)
            allpointsMinLon.append(minlon)
            allpoints.append(points)
    
            if convexhull:
                point_arr = np.asarray(points)
                points_np_arr.append(point_arr)
    logging.info("The number of regions is %d" %(len(allpoints)))
    if not convexhull:
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
            #should be uncommented if we want the rectangles
            #ax.add_artist(rect)
            ax.set_xlim([-125, -60])  # pylab.xlim([-400, 400])
            ax.set_ylim([25, 50])
            xticks = ax.xaxis.get_major_ticks()
            yticks = ax.yaxis.get_major_ticks()
            for i in range(2, len(xticks)-1):
                 xticks[i].label1.set_visible(False)
            for i in range(1, len(yticks)-1):
                 yticks[i].label1.set_visible(False)
            plt.tick_params(axis='both', which='major', labelsize=9)

        
         
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
    ##plt.xlabel('Longitude', fontsize=9)
    ##plt.ylabel('Latitude', fontsize=9)
    ##print_point_sizes = [1] * len(print_lats)
    ##colors = [int(i % 23) for i in print_classes]
    #median k-dtree plotting
    ##plt.plot(print_lons, print_lats , 'k.', markersize=0.1)
    #svd plotting
    #plt.scatter(print_lons, print_lats, c=colors, marker='x', s=2)
    #convex hull plotting
    if convexhull:
        cs = get_cmap(len(points_np_arr)) 
        for i in range(len(points_np_arr)):
            points = points_np_arr[i]
            #points = np.transpose(points)
            
            #print points.shape
            if points.shape[0] < 3:
                continue
            convex_hull = ConvexHull(points)
            #Tracer()()
            for simplex in convex_hull.simplices:
                plt.plot(points[simplex,1], points[simplex,0], color=cs(i))
                
                #Tracer()()
            #patch = mpatches.Polygon(points[convex_hull.vertices])
            #ax.add_patch(patch)
            
                #plt.plot(points[:,0], points[:, 1])
    
    #plt.title('US Map of Twitter Users partitioned by ' + partitionMethod + ' method: ' + str(granularity).strip() + ' person per cluster')
    plt.savefig(filename + '.pdf', format='pdf')
    plt.close()
    print "the plot saved in " + filename + '.jpg'
    # plt.show()  # pylab.show()            
    







def createTrainDir(granularity, partitionMethod, create_dir=False):
    # readlocationclusters
    global classLatMean
    global classLatMedian
    global classLonMean
    global classLonMedian
    global testClasses
    global devClasses
    global trainClasses
    global categories
    categories = []
    filename = path.join(GEOTEXT_HOME, 'processed_data/' + str(granularity).strip() + '_' + partitionMethod + '_clustered.train')
    print "reading " + filename
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
            points = [loc.strip().split(',') for loc in fields]
            '''
            for field in fields:
                dims = field.split(',')
                try:
                    lat = float(dims[0].strip())
                except:
                    print dims
                    print line
                    sys.exit()                                                                                                                                                                                                      
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
            '''
            lats = [float(point[0]) for point in points]
            lons = [float(point[1]) for point in points]
            minlat = min(lats)
            maxlat = max(lats)
            minlon = min(lons)
            maxlon = max(lons)
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
        label = str(i).strip()
        categories.append(label)
        classLatMedian[label] = medianlat
        classLonMedian[label] = medianlon
        classLatMean[label] = meanlat
        classLonMean[label] = meanlon    
        
        if create_dir:
            class_dir = path.join(trainhome, label)
            os.mkdir(class_dir)
        
        for location in cluster:
            # find user(s) in that collection
            locationStr = location[0] + ',' + location[1]
            '''
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
            '''
            locusers = locationUser[locationStr]
            user_class = dict(zip(locusers, [i] * len(locusers)))
            trainClasses.update(user_class)
            
            #Tracer()()
            #Tracer()()
            #for user in locusers:
            #    trainClasses[user] = i
            # for each user in this location find the text
            # groupbyusersText should be true for this to work
            # print "writing user texts in their corresponding geographical class in: " + class_dir
            if create_dir:
                for user in locusers:
                    with codecs.open(path.join(class_dir, user), 'w', encoding=data_encoding) as inf:
                        inf.write(trainText[user])
    print "train directories created and class median and mean lat,lon computed. trainfile: " + filename
    MULTI_LABEL = False
    if MULTI_LABEL:
        print "multiple labeling in progress"
        for user in trainClasses:
            latitude, longitude = locationStr2Float(trainUsers[user])
            assignMultipleClass(latitude, longitude, user)
        print "multiple labeling finished"
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

            


        
    Report_Ideal = False
    if Report_Ideal:
        print "Ideal mean dev distance is " + str(mean(devDistances))
        print "Ideal median dev distance is " + str(median(devDistances))
        print "Ideal Acc@161 dev is " + str(len([dist for dist in devDistances if dist < 161]) / (len(devDistances) + 0.0))
        
        print "Ideal mean test distance is " + str(mean(testDistances))
        print "Ideal median test distance is " + str(median(testDistances))
        print "Ideal Acc@161 test is " + str(len([dist for dist in testDistances if dist < 161]) / (len(testDistances) + 0.0))
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

def assignMultipleClass(latitude, longitude, user):
    '''
    Given a coordinate find the probability of the user belonging to each class and save it in user_multiple_label dictionary
    '''
    global user_multi_labels
    #labels and their probability/score
    multi_labels = {}
    minDistance = 1000000
    classIndex = -1
    min_threshold = 10
    scores = []
    labels = []
    for i in classLatMedian:
        lat = classLatMedian[str(i).strip()]
        lon = classLonMedian[str(i).strip()]
        dist = distance(latitude, longitude, lat, lon)
        #if dist < min_threshold:
        #    multi_labels[i] = 1
        score = 10 - dist
        if score < 0:
            score = 0
        scores.append(score)
        labels.append(i)
    sum_scores = sum(scores)
    max_scores = max(scores)
    if sum_scores==0:
        probs = [0.0] * len(scores)
    else:
        probs = [score/sum_scores for score in scores]
    
    multi_labels = dict(zip(labels, probs))
    label = trainClasses[user]
    #if sum_scores==0:
    multi_labels[label] = 1.0
    user_multi_labels[user] = multi_labels
            

        
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
    #cProfile.run('createTrainDir(granularity, partitionMethod, write)')
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
            
def error(predicted_label, user):
    lat1, lon1 = locationStr2Float(userLocation[user])
    lat2 = classLatMedian[predicted_label]  
    lon2 = classLonMedian[predicted_label]
    return distance(lat1, lon1, lat2, lon2)         
def loss(preds, U_test, loss='median', save=False):
    if len(preds) != len(U_test): 
        print "The number of test sample predictions is: " + str(len(preds))
        print "The number of test samples is: " + str(len(U_test))
        print "fatal error!"
        sys.exit()
    sumMeanDistance = 0
    sumMedianDistance = 0
    distances = []
    user_location = {}
    acc = 0.0
    for i in range(0, len(preds)):
        user = U_test[i]
        location = userLocation[user].split(',')
        lat = float(location[0])
        lon = float(location[1])
        if preds[i] == int(Y_test[i]):
            acc += 1
        # print str(Y_test[i]) + " " + str(preds[i])
        prediction = categories[preds[i]]
        medianlat = classLatMedian[prediction]  
        medianlon = classLonMedian[prediction]  
        user_location[user] = (medianlat, medianlon)
        meanlat = classLatMean[prediction] 
        meanlon = classLonMean[prediction]
        predictionCoordinate = 'median'
        if predictionCoordinate == 'median':      
            distances.append(distance(lat, lon, medianlat, medianlon))
        elif predictionCoordinate == 'mean':
            distances.append(distance(lat, lon, meanlat, meanlon))
    if save:
        print "dumping the results in preds.pkl"
        with open(path.join(GEOTEXT_HOME, 'preds.pkl'), 'wb') as outf:
            pickle.dump(user_location, outf) 
    averageMeanDistance = sumMeanDistance / float(len(preds))
    averageMedianDistance = sumMedianDistance / float(len(preds))
    medianDistance = median(distances)
    # print "Average distance from class mean is " + str(averageMeanDistance)
    # print "Average distance from class median is " + str(averageMedianDistance)
    print "Mean distance is " + str(mean(distances))
    print "Median distance is " + str(median(distances))
    print "Accuracy @ 161 k.m. is " + str(100 * len([d for d in distances if d < 161]) / float(len(distances)))
    print "Classification Accuracy is " + str(100 * acc / len(preds))
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
    print "loading train, dev and test files..."
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
            print "using @ mention dictionary as vocab..."
            extract_mentions()
            vectorizer = TfidfVectorizer(use_idf=use_idf, norm=norm, binary=binary, sublinear_tf=sublinear_tf, min_df=min_df, max_df=max_df, ngram_range=(1, 1), stop_words=None, vocabulary=mentions)
        else:
            print "mindf: " + str(min_df) + " maxdf: " + str(max_df)
            vectorizer = TfidfVectorizer(use_idf=use_idf, norm=norm, binary=binary, sublinear_tf=sublinear_tf, min_df=min_df, max_df=max_df, ngram_range=(1, 1), stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
    # keys = vectorizer.vocabulary_.keys()

        
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
        k = 200000
        print("Extracting %d best features by a chi-squared test" % k)
        t0 = time.time()
        ch2 = SelectKBest(chi2, k=k)
        X_train = ch2.fit_transform(X_train, Y_train)
        X_test = ch2.transform(X_test)
        X_dev = ch2.transform(X_dev)
        print("done in %fs" % (time.time() - t0))
        print()
        feature_names = np.asarray(vectorizer.get_feature_names())
    else:
        feature_names = np.asarray(vectorizer.get_feature_names())

    DO_SVD = False
    Reduction_D = 1000
    if DO_SVD:
        print("dimension reduction svd with d=%d" % Reduction_D)
        svd = TruncatedSVD(n_components=Reduction_D, algorithm="randomized", n_iter=5, random_state=None, tol=0)
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

def feature_extractor2(use_mention_dictionary=False, use_idf=True, norm='l2', binary=False, sublinear_tf=True, min_df=1, max_df=1.0, BuildCostMatrices=False, vectorizer=None, stop_words=None, novectorization=False):
    '''
    read train, dev and test directories and extract textual features using tfidfvectorizer.
    '''
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
    

    
    
    U_train = [u for u in sorted(trainUsers)]
    U_test = [u for u in sorted(testUsers)]
    U_dev = [u for u in sorted(devUsers)]

    

    print("%d categories" % len(categories))
    print()
    
    # split a training set and a test set
    Y_train = np.asarray([trainClasses[u] for u in U_train])
    Y_test = np.asarray([testClasses[u] for u in U_test])
    Y_dev = np.asarray([devClasses[u] for u in U_dev])
    
    if novectorization:
        X_train = None
        X_test = None
        X_dev = None
        feature_names = None
        return X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names  
    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time.time()
    
    if vectorizer == None:    
        if use_mention_dictionary:
            print "using @ mention dictionary as vocab..."
            extract_mentions()
            vectorizer = TfidfVectorizer(use_idf=use_idf, norm=norm, binary=binary, sublinear_tf=sublinear_tf, min_df=min_df, max_df=max_df, ngram_range=(1, 1), vocabulary=mentions, stop_words=stop_words)
        else:
            print "mindf: " + str(min_df) + " maxdf: " + str(max_df)
            vectorizer = TfidfVectorizer(use_idf=use_idf, norm=norm, binary=binary, sublinear_tf=sublinear_tf, min_df=min_df, max_df=max_df, ngram_range=(1, 1), stop_words=stop_words)

    X_train = vectorizer.fit_transform([trainText[u] for u in U_train])
    # keys = vectorizer.vocabulary_.keys()

            
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
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()
    
    print("Extracting features from the dev dataset using the same vectorizer")
    t0 = time.time()
    X_dev = vectorizer.transform([devText[u] for u in U_dev])
    duration = time.time() - t0
    print("n_samples: %d, n_features: %d" % X_dev.shape)
    print()

    print("Extracting features from the test dataset using the same vectorizer")
    t0 = time.time()
    X_test = vectorizer.transform([testText[u] for u in U_test])
    duration = time.time() - t0
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()
    ppmitransform = False
    if ppmitransform:
        ppmiTransform(X_train)
        ppmiTransform(X_dev)
        ppmiTransform(X_test)
    
            
    chi = False
    if chi:
        k = 200000
        print("Extracting %d best features by a chi-squared test" % k)
        t0 = time.time()
        ch2 = SelectKBest(chi2, k=k)
        X_train = ch2.fit_transform(X_train, Y_train)
        X_test = ch2.transform(X_test)
        X_dev = ch2.transform(X_dev)
        print("done in %fs" % (time.time() - t0))
        print()
        feature_names = np.asarray(vectorizer.get_feature_names())
    else:
        feature_names = np.asarray(vectorizer.get_feature_names())

    DO_SVD = False
    Reduction_D = 1000
    if DO_SVD:
        print("dimension reduction svd with d=%d" % Reduction_D)
        svd = TruncatedSVD(n_components=Reduction_D, algorithm="randomized", n_iter=5, random_state=None, tol=0)
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
    
    
def classify(X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names, granularity=10, DSExpansion=False, DSModification=False, compute_dev=False, report_verbose=False, clf=None, regul=0.00001):
    compute_dev=True
    if DSExpansion:
        X_train, Y_train = dataSpaceExpansion(X_train, Y_train)
    
    if DSModification:
        Y_train = dataSpaceModification(Y_train, U_train)
    if clf == None:
        # clf = LinearSVC(multi_class='ovr', class_weight='auto', C=1.0, loss='l2', penalty='l2', dual=False, tol=1e-3)
        # clf = linear_model.LogisticRegression(C=1.0, penalty='l2')
        alpha = 0.0000001
        if DATASET_NUMBER == 1:
            alpha = 0.00005
        elif DATASET_NUMBER == 2:
            alpha = 0.000001
        elif DATASET_NUMBER == 3:
            alpha = 0.000001
        # alpha = 0.000001
        clf = SGDClassifier(loss='log', alpha=regul, penalty='l1', learning_rate='optimal', n_iter=10, shuffle=False, n_jobs=60 )
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
        # clf = MultinomialNB(alpha=.01)
    print('_' * 80)
    print("Training: ")
    print(clf)
    
    
    t0 = time.time()
    clf.fit(X_train, Y_train)
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)
    
    
    
    if compute_dev:
        devPreds = clf.predict(X_dev)
        devProbs = clf.predict_proba(X_dev)
        if report_verbose:
            score1 = metrics.f1_score(Y_dev, devPreds)
            score2 = metrics.accuracy_score(Y_dev, devPreds)
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

    
    
    print "predicting test labels"  
    t0 = time.time()
    preds = clf.predict(X_test)
    #scores = clf.decision_function(X_test)
    testProbs = clf.predict_proba(X_test)
    probs = None
    # print preds.shape
    test_time = time.time() - t0
    print("test time: %0.3fs" % test_time)
    
    
    

    if report_verbose:
        score1 = metrics.f1_score(Y_test, preds)
        score2 = metrics.accuracy_score(Y_test, preds)
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
    meanTest, medianTest = loss(preds, U_test, save=True)
    meanDev = -1
    medianDev = -1
    if compute_dev:
        print "development results"
        meanDev, medianDev = loss(devPreds, U_dev)
    result_dump_file = path.join(GEOTEXT_HOME, 'results-' + DATASETS[DATASET_NUMBER-1] + '-' + str(BUCKET_SIZE) + '.pkl')
    print "dumping preds (preds, devPreds, U_test, U_dev, testProbs, devProbs) in " + result_dump_file
    with open(result_dump_file, 'wb') as outf:
        pickle.dump((preds, devPreds, U_test, U_dev, testProbs, devProbs), outf)
    # evaluate(preds,U_test, categories, None)
    # abod(probs, preds, U_test)
    #Tracer()()
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
        svd = TruncatedSVD(n_components=Reduction_D, algorithm="randomized", n_iter=5, random_state=None, tol=0)
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

def initialize(partitionMethod, granularity, write=False, readText=True, downSampleTextCoefficient=1.0, reload_init=False):    
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
    


    lngs = []
    ltts = []
    pointText = {}
    keys = []
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
    reload_file = path.join(GEOTEXT_HOME + '/init_' + DATASETS[DATASET_NUMBER - 1]  + '_' + str(BUCKET_SIZE) + '.pkl')
    if reload_init and not readText:
        if path.exists(reload_file):
            with open(reload_file, 'rb') as inf:
                #print('reading info from %s' %(reload_file) )
                print(Fore.RED + 'reading info from %s' %(reload_file))
                print(Fore.RESET)
                classLatMean, classLonMedian, classLatMedian, classLonMean, userLocation, categories, testUsers, testClasses, devUsers, devClasses = pickle.load(inf)
                return
    # readGeoTextRecords(encoding=data_encoding)
    print 'reading (user_info.) train, dev and test file and building trainUsers, devUsers and testUsers with their locations'
    users(trainfile, 'train', write, readText=readText, downSampleTextCoefficient=downSampleTextCoefficient)
    users(devfile, 'dev', write, readText=readText)
    users(testfile, 'test', write, readText=readText)
    print "the number of train" + " users is " + str(len(trainUsers))
    print "the number of test" + " users is " + str(len(testUsers))
    print "the number of dev" + " users is " + str(len(devUsers))
    # print 'total ' + str(len(userLocation)).strip() + " users."
    # fillUserByLocation()
    # fillTextByUser(encoding=data_encoding)
    create_directories(granularity, partitionMethod, write)  
    if reload_init and not readText:
        print('writing init info in %s' %(reload_file))
        with open(reload_file, 'wb') as outf:
            pickle.dump((classLatMean, classLonMedian, classLatMedian, classLonMean, userLocation, categories, testUsers, testClasses, devUsers, devClasses), outf)
    print "initialization finished"

def classificationBench(granularity, partitionMethod, use_mention_dictionary=False):
    medians = {}
    max_dfs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
    # max_dfs = [1, 5, 10, 20]
    clf1 = MultinomialNB(alpha=.01)
    clf2 = SGDClassifier(loss='log', penalty='l1', learning_rate='optimal', n_iter=5, shuffle=False, n_jobs=20)
    clf3 = SGDClassifier(loss='log', penalty='l2', learning_rate='optimal', n_iter=5, shuffle=False, n_jobs=20)    
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
def learn_a_dictionary(X, transformees):
    #dic_learner = DictionaryLearning(n_components=100, alpha=1, max_iter=1000, tol=1e-8, fit_algorithm='lars', transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, n_jobs=30, code_init=None, dict_init=None, verbose=True, split_sign=None, random_state=None)
    dic_learner = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500, fit_algorithm='lars', n_jobs=60, batch_size=3, shuffle=True, dict_init=None, transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, verbose=False, split_sign=False, random_state=None)
    if sparse.issparse(X):
        X = X.toarray()
    dic_learner.fit(X)
    results = []
    for feature_matrix in transformees:
        feature_matrix = feature_matrix.toarray()
        feature_matrix = dic_learner.transform(feature_matrix)
        results.append(feature_matrix)
    return results
def asclassification(granularity, partitionMethod, use_mention_dictionary=False):


    stops = 'english'
    # partitionLocView(granularity=granularity, partitionMethod=partitionMethod)
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor2(norm='l2', use_mention_dictionary=use_mention_dictionary, min_df=10, max_df=0.2, stop_words=stops)    
    X_train, X_dev, X_test = learn_a_dictionary(X_train, transformees=[X_train, X_dev, X_test])
    
    for regul in [reguls[DATASET_NUMBER-1]]:
        preds, probs, U_test, meanTest, medianTest, meanDev, medianDev = classify(X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names, granularity, regul=regul)
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

def extract_mentions(k=0, addTest=False, addDev=False):
    if addTest and addDev:
        print "addTest and addDev can not be True in the same time"
        sys.exit(0)
    print "extracting mention information from text"
    global mentions
    # if it is there load it and return
    mention_file_address = path.join(GEOTEXT_HOME, 'mentions.pkl')
    if addDev:
        mention_file_address = mention_file_address + '.dev'
    RELOAD_MENTIONS = True
    if RELOAD_MENTIONS:
        if os.path.exists(mention_file_address):
            print "reading mentions from pickle"
            with open(mention_file_address, 'rb') as inf:
                mentions = pickle.load(inf)
                return
    text = ''
    # for user in trainUsers:
    #    text += userText[user].lower()
    if addTest:
        text = ' '.join(trainText.values() + testText.values())
    if addDev:
        text = ' '.join(trainText.values() + devText.values())
    if not addTest and not addDev:
        text = ' '.join(trainText.values())
    # text = text.lower()
    '''
    if data_encoding in ['utf-8']:
        text = strip_accents_unicode(text)
    elif data_encoding in ['latin', 'latin1']:
        text = strip_accents_ascii(text)
    '''
    token_pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
    token_pattern = re.compile(token_pattern)
    print "finding mentions"
    mentionsList = [word.lower() for word in token_pattern.findall(text)]
    print "building the counter"
    mentionsDic = Counter(mentionsList)
    print "frequency thresholding"
    if k > 0:
        mentions = [word for word in mentionsDic if mentionsDic[word] > k]
    else:
        mentions = mentionsDic.keys()

    with open(mention_file_address, 'wb') as outf:
        print "writng mentions to pickle"
        pickle.dump(mentions, outf)
    
def spams_groups(feature_names, X_train, type, k=0):
    if type == "mentions":
        print "building feature group memberships"
        print "the number of features is " + str(len(feature_names))    
        indices = [i for i, feature in enumerate(feature_names) if feature.startswith('user_')]
        
        ngroups = len(indices)
        print "the number of groups is " + str(ngroups)
        groupDesign = X_train[:, indices ]
        eta_g = np.ones(ngroups, dtype=float)
        groups = sparse.csc_matrix(np.zeros((ngroups, ngroups)), dtype=np.bool)
        groups_var = sparse.csc_matrix(X_train.transpose().dot(groupDesign), dtype=np.bool)
        graph = {'eta_g': eta_g, 'groups' : groups, 'groups_var' : groups_var}
        return graph
    elif type == "count":
        X_train2, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names2 = feature_extractor(use_mention_dictionary=False, min_df=1, max_df=1.0, norm=None, binary=True, use_idf=False, sublinear_tf=False)
        counts = X_train2.sum(axis=0)
        ngroups = counts.max() - counts.min() + 1
        print "the number of features is " + str(len(feature_names2))
        if len(feature_names) != len(feature_names2):
            print "fatal error"
            sys.exit() 
        print "the number of groups is " + str(ngroups)
        # eta_g = np.ones(ngroups,dtype=float)
        # eta_g = np.arange(1, ngroups + 1,dtype=float)
        print "np.log + 0.5"
        eta_g = np.log(np.arange(1, ngroups + 1, dtype=float)) + 0.5
        groups = sparse.csc_matrix(np.zeros((ngroups, ngroups)), dtype=np.bool)
        groups_var = sparse.csc_matrix(np.zeros((len(feature_names), ngroups)), dtype=np.bool)
        for i in range(0, len(feature_names)):
            groups_var[i, int(counts[0, i]) - counts.min()] = 1
        graph = {'eta_g': eta_g, 'groups' : groups, 'groups_var' : groups_var}
        return graph
        
        
    
def iterative_collective_classification(granularity=600, partitionMethod='median'):
    global feature_names

     # extract content features
    print "content features"
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor(use_mention_dictionary=False, min_df=1)

    # vstack train and dev/test data
    X_content = sparse.vstack([X_train, X_test]).tocsr()
    print "deleting some references"
    del X_train
    del X_dev
    del X_test   
    # classify
    # preds, probs, U_test, meanTest, medianTest, meanDev, medianDev = classify(X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names, granularity, compute_dev=False, report_verbose=False)
    # print "content only results"
    # loss(preds, U_test)
    # build network based on mentions (a pairwise similarity measure)
    extract_mentions(k=0)
    vectorizer = CountVectorizer(encoding=data_encoding, vocabulary=mentions, binary=True)
    X_train_relational, Y_train, U_train, X_dev_relational, Y_dev, U_dev, X_test_relational, Y_test, U_test, categories, feature_names_relational = feature_extractor(use_mention_dictionary=True, use_idf=False, norm=None, binary=True, sublinear_tf=False, min_df=1, BuildCostMatrices=False, vectorizer=vectorizer)
    preds_rel, probs_rel, U_test_rel, meanTest, medianTest, meanDev, medianDev = classify(X_train_relational, Y_train, U_train, X_dev_relational, Y_dev, U_dev, X_test_relational, Y_test, U_test, categories, feature_names, granularity, compute_dev=False, report_verbose=False)
    print "network only results"
    loss(preds_rel, U_test)
    

    
    X_relational = sparse.vstack([X_train_relational, X_test_relational]).tocsr()
    del X_train_relational
    del X_test_relational
    Xt_relational = np.transpose(X_relational)
    X_relational_shape0 = X_relational.shape[0]
    # find the number of shared mentions for all pairs of users/samples 
    print "computing the pairs matrix"
    pairs = X_relational.dot(Xt_relational)
    
    print "deleting relational references"
    del X_relational
    del Xt_relational
    # set the diagonal to zero (the number of shared mentions of a user with herself)
    # equal1_indices = pairs < 2
    # pairs[equal1_indices] = 0
    # pairs = threshold(pairs, 2, 100, 0)
    pairs.setdiag(0)
    print "adding collective features to the design matrix"
    content_relational_features = feature_names.tolist()
    for i in range(0, len(categories)):
        content_relational_features.append('new_relational_aggregated_feature' + str(i))
    feature_names = np.asarray(content_relational_features)
    n_iter = 10
    for i in range(0, n_iter):
        print "iteration " + str(i)
        Y_train_reshaped = Y_train.reshape(Y_train.shape[0], 1)
        Y_test_reshaped = Y_test.reshape(Y_test.shape[0], 1)
        Y = csr_matrix(sparse.vstack([Y_train_reshaped, Y_test_reshaped])).todense()
        XClass = csr_matrix((X_relational_shape0, len(categories)), dtype=int)
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
        # Y_test = np.array(preds)
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
        # print text
    # token_pattern = r"[\@\#]+\w\w+\b"
    # token_pattern = re.compile(token_pattern)
    # mentionsList = [word for word in token_pattern.findall(text)]

    # print a
    # mentionsList = [word for word in text.split() if word.startswith('@') or word.startswith('#')]
    mentionsList = []
    for word in text.split():
        if (word.startswith('@') or word.startswith('#')) :
            mentionsList.append(word)
    # print mentionsList[:-1]
    print len(mentionsList)

def spams_group_lasso():
    import spams
    import numpy as np
    import scipy.sparse as ssp
    myfloat = float
    # print "first solve the dense/sparse issue"
    # sys.exit()
    X_train, Y_train, U_train, X_dev, Y_dev, U_dev, X_test, Y_test, U_test, categories, feature_names = feature_extractor2(use_mention_dictionary=False, min_df=10, max_df=1.0, norm=None)
    # extract_mentions()
    # graph = spams_groups(feature_names,X_train,type="mentions", k=0)
    # graph = spams_groups(feature_names,X_train,type="count", k=0)
    # X should be n_feature x n_sample in spams
    # X_train = normalize(X_train, norm='l2')
    # X_train = X_train.todense()
    # X = np.asfortranarray(X_train)
    # X2 = ssp.csc_matrix(X)
    # X = spams.normalize(X)
    X_train = X_train.tocsc()
    X_test = X_test.tocsc()
    X_dev = X_dev.tocsc()
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    # X = np.asfortranarray(X - np.tile(np.mean(X,0),(X.shape[0],1)),dtype=myfloat)
    # X = spams.normalize(X1)
    Y = np.asfortranarray(np.ceil(5 * np.random.random(size=(Y_train.shape[0], 1))) - 1, dtype=myfloat)
    for i in range(0, Y.shape[0]):
        Y[i, 0] = Y_train[i, 0]
    
    # Multi-Class classification

    param = {'numThreads' :-1, 'verbose' : True,
             'lambda1' : 0.05, 'it0' : 10, 'max_it' : 200,
             'L0' : 0.1, 'tol' : 1e-3, 'intercept' : False,
             'pos' : False}
    param['loss'] = 'multi-logistic'
    param['regul'] = 'l1'
    param['lambda1'] = 0.0000000001

    print '\nFISTA + Multi-Class Logistic l1'
    print param
    nclasses = np.max(Y[:]) + 1
    W0 = np.zeros((X_train.shape[1], nclasses * Y.shape[1]), dtype=myfloat, order="FORTRAN")
    (W, optim_info) = spams.fistaFlat(Y, X_train, W0, True, **param)
    W = ssp.csr_matrix(W)
    results = X_test.dot(W)
    results = results.todense()
    preds = np.argmax(results, axis=1)
    print "test results"
    loss(preds, U_test)
    print "development results"
    # X_dev = X_dev.todense()
    results = X_dev.dot(W)
    results = results.todense()
    preds = np.argmax(results, axis=1)
    loss(preds, U_dev)
    report_train = False
    if report_train:
        print "train results"
        results = X_train.dot(W)
        results = results.todense()
        preds = np.argmax(results, axis=1)
        loss(preds, U_train)
        print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' % (np.mean(optim_info[0, :]), np.mean(optim_info[2, :]), np.mean(optim_info[3, :]))
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

def data_id(i, j):
    if i > j:
        return str(j) + '\t' + str(i)
    else:
        return str(i) + '\t' + str(j)
def prepare_adsorption_data_collapsed(DEVELOPMENT=False, ADD_TEXT_LEARNER=False, CELEBRITY_THRESHOLD=100000, build_networkx_graph=False, DIRECT_GRAPH_WEIGHTED=False):
    global trainUsers
    global testUsers
    global trainText
    global testText
    global mentions
    global U_train
    global U_test
    global U_dev
    global categories
    global mention_graph
    DIRECT_MULTIPLICATION = False
    PAIRWISE = False
    DIRECT_GRAPH = True
    CONFIDENCE = 0.01
    MULTI_LABEL = False
    DONGLE = True
    dongle_nodes = None
    dongle_preds = None
    dongle_probs = None
    U_train = [u for u in sorted(trainUsers)]
    U_test = [u for u in sorted(testUsers)]
    U_dev = [u for u in sorted(devUsers)]
    text_str = ''
    if ADD_TEXT_LEARNER:
        text_str = '.text'
    weighted_str = ''
    if DIRECT_GRAPH_WEIGHTED:
        weighted_str = '.weighted'
    


    
    # split a training set and a test set
    Y_train = np.asarray([trainClasses[u] for u in U_train])
    Y_test = np.asarray([testClasses[u] for u in U_test])
    Y_dev = np.asarray([devClasses[u] for u in U_dev])

    
    trainUsersLower = [u.lower() for u in U_train]
    testUsersLower = [u.lower() for u in U_test]
    devUsersLower = [u.lower() for u in U_dev]

    n_jobz = 2
    if DIRECT_GRAPH:
        mentions = []
    else:
        extract_mentions(k=0, addTest=not DEVELOPMENT, addDev=DEVELOPMENT)
    new_mentions = []
    if DEVELOPMENT:
        new_mentions = set(mentions) - set(trainUsersLower) - set(devUsersLower)
        new_mentions = trainUsersLower + devUsersLower + list(new_mentions)
        new_mentions = list(set(new_mentions))
        devStr = '.dev'
    else:
        new_mentions = set(mentions) - set(trainUsersLower) - set(testUsersLower)
        new_mentions = trainUsersLower + testUsersLower + list(new_mentions)
        new_mentions = list(set(new_mentions))
        devStr = ''
    vocab_cnt = len(new_mentions)
    idx = range(vocab_cnt)
    vocab = dict(zip(new_mentions, idx))
    vectorizer = CountVectorizer(encoding=data_encoding, vocabulary=vocab, binary=False)
    
    if ADD_TEXT_LEARNER:
        print "dongle is " + str(DONGLE)
        #read users and predictions
        result_dump_file = path.join(GEOTEXT_HOME, 'results-' + DATASETS[DATASET_NUMBER-1] + '-' + str(BUCKET_SIZE) + '.pkl')
        t_preds, d_preds, t_users, d_users, t_probs, d_probs = None, None, None, None, None, None
        with open(result_dump_file, 'rb') as inf:
            t_preds, d_preds, t_users, d_users, t_probs, d_probs = pickle.load(inf)

        if DEVELOPMENT:
            dongle_nodes = d_users
            dongle_preds = d_preds
            dongle_probs = d_probs
        else:
            dongle_nodes = t_users
            dongle_preds = t_preds
            dongle_probs = t_probs
            
                      
    if DIRECT_GRAPH:
        pass
        # we don't want Xs
        #X_train_relational, Y_train, U_train, X_dev_relational, Y_dev, U_dev, X_test_relational, Y_test, U_test, categories, feature_names_relational = feature_extractor2(use_mention_dictionary=True, use_idf=False, norm=None, binary=True, sublinear_tf=False, min_df=1, BuildCostMatrices=False, vectorizer=vectorizer, novectorization=True)
    else:
        X_train_relational, Y_train, U_train, X_dev_relational, Y_dev, U_dev, X_test_relational, Y_test, U_test, categories, feature_names_relational = feature_extractor2(use_mention_dictionary=True, use_idf=False, norm=None, binary=True, sublinear_tf=False, min_df=1, BuildCostMatrices=False, vectorizer=vectorizer)
        if DEVELOPMENT:
            X_relational = sparse.vstack([X_train_relational, X_dev_relational]).tocsr()
        else:
            X_relational = sparse.vstack([X_train_relational, X_test_relational]).tocsr()
        
    print "writing id_user"
    with codecs.open(path.join(GEOTEXT_HOME, 'id_user_' + str(BUCKET_SIZE) + devStr + text_str + weighted_str), 'w', 'ascii') as outf:
        for i in range(0, len(trainUsersLower)):
            outf.write(str(i) + '\t' + trainUsersLower[i] + '\t' + '1.0' + '\n')
        if DEVELOPMENT:
            for i in range(0, len(devUsersLower)):
                outf.write(str(i + len(trainUsersLower)) + '\t' + devUsersLower[i] + '\t' + '1.0' + '\n')
        else:
            for i in range(0, len(testUsersLower)):
                outf.write(str(i + len(trainUsersLower)) + '\t' + testUsersLower[i] + '\t' + '1.0' + '\n')

    print "writing seeds"
    with codecs.open(path.join(GEOTEXT_HOME, 'seeds_' + str(BUCKET_SIZE) + devStr + text_str + weighted_str), 'w', 'ascii') as outf:
        for i in range(0, len(trainUsersLower)):
            if MULTI_LABEL:
                user = U_train[i]
                multi_labels = user_multi_labels[user]
                for label, p in multi_labels.iteritems():
                    if p!=0.0:
                        outf.write(str(i) + '\t' + str(label) + '\t' + str(p) + '\n')
            else:
                outf.write(str(i) + '\t' + str(Y_train[i]) + '\t' + '1.0' + '\n')
        if ADD_TEXT_LEARNER:
            for i in range(0, len(dongle_nodes)):
                if DONGLE:
                    outf.write(str(i + len(trainUsersLower))+'.T' + '\t' + str(dongle_preds[i]) + '\t' + str(np.max(dongle_probs[i])) + '\n')
                else:
                    outf.write(str(i + len(trainUsersLower)) + '\t' + str(dongle_preds[i]) + '\t' + str(np.max(dongle_probs[i])) + '\n')
            
    print "writing gold_labels"
    with codecs.open(path.join(GEOTEXT_HOME, 'gold_labels_' + str(BUCKET_SIZE) + devStr + text_str + weighted_str), 'w', 'ascii') as outf:
        if DEVELOPMENT:
            for i in range(0, len(devUsersLower)):
                outf.write(str(i + len(trainUsersLower)) + '\t' + str(Y_dev[i]) + '\t' + '1.0' + '\n')
        else:
            for i in range(0, len(testUsersLower)):
                outf.write(str(i + len(trainUsersLower)) + '\t' + str(Y_test[i]) + '\t' + '1.0' + '\n')
    REALVALUED = False
    if REALVALUED:
        print "writing seeds_real_valued"
        with codecs.open(path.join(GEOTEXT_HOME, 'seeds_' + str(BUCKET_SIZE) + devStr + '.real'), 'w', 'ascii') as outf:
            for i in range(0, len(trainUsersLower)):
                lat, lon = locationStr2Float(userLocation[U_train[i]])
                outf.write(str(i) + '\t' + 'lat' + '\t' + str(lat) + '\n')
                outf.write(str(i) + '\t' + 'lon' + '\t' + str(lon) + '\n')

        print "writing gold_labels_real_valued"
        with codecs.open(path.join(GEOTEXT_HOME, 'gold_labels_' + str(BUCKET_SIZE) + devStr + '.real'), 'w', 'ascii') as outf:
            for i in range(0, len(testUsersLower)):
                lat, lon = locationStr2Float(userLocation[U_test[i]])
                outf.write(str(i + len(trainUsersLower)) + '\t' + 'lat' + '\t' + str(lat) + '\n')
                outf.write(str(i + len(trainUsersLower)) + '\t' + 'lon' + '\t' + str(lon) + '\n')

    

        
    
    if DIRECT_MULTIPLICATION:
        Xt_relational = np.transpose(X_relational)
    
        # find the number of shared mentions for all pairs of users/samples 
        print "computing the pairs matrix"
        pairs = X_relational.dot(Xt_relational)
        pairs = pairs.tocoo()
        print "number of nonzero cells is " + str(pairs.nnz)
        xindx, yindx = pairs.nonzero()
        tenpercent = pairs.nnz / 100
        # xindx = xindx.tolist()
        # yindx = yindx.tolist()
        with codecs.open(path.join(GEOTEXT_HOME, 'input_graph_' + str(BUCKET_SIZE) + devStr + text_str + weighted_str), 'w', 'ascii') as outf:
            i = 1
            for xindx, yindx, w in zip(pairs.row, pairs.col, pairs.data):
                # w = pairs[xindx, yindx]
                if i % tenpercent == 0:
                    print "processing " + str(i / tenpercent) + "%"
                i += 1
                if xindx == yindx:
                    continue
                outf.write(str(xindx) + '\t' + str(yindx) + '\t' + str(w) + '\n')
    if PAIRWISE:
        print "computing pairwise similarities"
        distances = pairwise_distances(X_relational, Y=None, metric='cosine', n_jobs=n_jobz)
        sims = 1 - distances
        sims = coo_matrix(sims)

        tenpercent = sims.nnz / 10
        i = 1
        with codecs.open(path.join(GEOTEXT_HOME, 'input_graph_' + str(BUCKET_SIZE) + devStr), 'w', 'ascii') as outf:
            for xindx, yindx, w in zip(sims.row, sims.col, sims.data):
                # w = pairs[xindx, yindx]
                if i % tenpercent == 0:
                    print "processing " + str(10 * i / tenpercent) + "%"
                i += 1
                if xindx == yindx:
                    continue
                outf.write(str(xindx) + '\t' + str(yindx) + '\t' + str(w) + '\n')

    if DIRECT_GRAPH and not DIRECT_GRAPH_WEIGHTED:
        doubles = 0
        trainIdx = range(len(trainUsersLower))
        trainUsersLowerDic = dict(zip(trainUsersLower, trainIdx))
        if DEVELOPMENT:
            devStr = '.dev'
            for i in range(0, len(devUsersLower)):
                u = devUsersLower[i]
                if u in trainUsersLowerDic:
                    devUsersLower[i] = u + '_double00'
                    doubles += 1
            u_unknown = devUsersLower
            u_text_unknown = devText
            U_all = U_train + U_dev
        else:
            for i in range(0, len(testUsersLower)):
                u = testUsersLower[i]
                if u in trainUsersLowerDic:
                    testUsersLower[i] = u + '_double00'
                    doubles += 1
            u_text_unknown = testText
            u_unknown = testUsersLower
            U_all = U_train + U_test  
        U_all_lower = [u.lower() for u in U_all]
        print "The number of test users found in train users is " + str(doubles)
        vocab_cnt = len(U_all)
        idx = range(vocab_cnt)
        if build_networkx_graph:
            mention_graph = nx.Graph()
            mention_graph.add_nodes_from(idx)
        node_id = dict(zip(U_all, idx))
        node_lower_id = {}
        #data and indices of a coo matrix to be populated
        coordinates = set()
        data = []
        for node, id in node_id.iteritems():
            node_lower_id[node.lower()] = id
        assert (len(node_id) == len(trainUsersLower) + len(u_unknown)), 'number of unique users is not eq u_train + u_test'
        print "the number of nodes is " + str(vocab_cnt)
        print "building the direct graph"
        token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        token_pattern1 = re.compile(token_pattern1)
        mention_users = defaultdict(set)
        l = len(trainText)
        tenpercent = l / 10
        i = 1
        for user, text in trainText.iteritems():
            user_id = node_id[user]
            if i % tenpercent == 0:
                print str(10 * i / tenpercent) + "%"
            i += 1  
            mentions = [u.lower() for u in token_pattern1.findall(text)] 
            mentionDic = Counter(mentions)
            for mention in mentionDic:
                # check if mention is a user node
                mention_id = node_lower_id.get(mention, -1)
                if mention_id != -1:	
                    if mention_id != user_id:
                        if user_id < mention_id:
                            coordinates.add((user_id, mention_id))
                        elif mention_id < user_id:
                            coordinates.add((mention_id, user_id))
                    
                mention_users[mention].add(user_id)
        
            
        print "adding the test graph"
        for user, text in u_text_unknown.iteritems():
            user_id = node_id[user]
            mentions = [u.lower() for u in token_pattern1.findall(text)]
            mentionDic = Counter(mentions)
            for mention in mentionDic:
                mention_id = node_lower_id.get(mention, -1)
                if mention_id != -1:
                    if mention_id != user_id:
                        if user_id < mention_id:
                            coordinates.add((user_id, mention_id))
                        elif mention_id < user_id:
                            coordinates.add((mention_id, user_id))
                mention_users[mention].add(user_id)
        
        
        
        

        print "setting binary relationships."
        l = len(mention_users)
        tenpercent = l / 10
        i = 1
        celebrities_count = 0
        for mention, user_ids in mention_users.iteritems():
            if i % tenpercent == 0:
                print str(10 * i / tenpercent) + "%"
            i += 1  
            
            if len(user_ids) > CELEBRITY_THRESHOLD:
                celebrities_count += 1
                continue
            
            for user_id1 in user_ids:
                for user_id2 in user_ids:
                    if user_id1 < user_id2:
                        coordinates.add((user_id1, user_id2))
                    elif user_id2 < user_id1:
                        coordinates.add((user_id2, user_id1))
        
        print "The number of celebrities is " + str(celebrities_count) + " ."
        rows = [row for row,col in coordinates]
        cols = [col for row,col in coordinates]
        data = [True] * len(rows)
        print "The number of edges is " + str(len(rows))
        data = np.asarray(data)
        rows = np.asarray(rows)
        cols = np.asarray(cols)  
        #Tracer()()               
        pairs = coo_matrix((data,(rows,cols)), shape=(rows.shape[0],cols.shape[0]), dtype=np.bool_)
        print "writing the binary graph"
        xindx, yindx = pairs.nonzero()
        tenpercent = pairs.nnz / 10
        # xindx = xindx.tolist()
        # yindx = yindx.tolist()
        pairs_dtype = pairs.dtype
        with codecs.open(path.join(GEOTEXT_HOME, 'input_graph_' + str(BUCKET_SIZE) + devStr + text_str + weighted_str), 'w', 'ascii') as outf:
            i = 1
            for xindx, yindx, w in zip(pairs.row, pairs.col, pairs.data):
                # w = pairs[xindx, yindx]
                if i % tenpercent == 0:
                    print "processing " + str(10 * i / tenpercent) + "%"
                i += 1
                if pairs_dtype == np.bool_:
                    w = 1.0
                outf.write(str(xindx) + '\t' + str(yindx) + '\t' + str(w) + '\n')
                if build_networkx_graph:
                    mention_graph.add_edge(xindx, yindx, attr_dict = {'w':1})
            if ADD_TEXT_LEARNER and DONGLE:
                for i in range(0, len(dongle_nodes)):
                    outf.write(str(i + len(trainUsersLower))+'.T' + '\t' + str(i + len(trainUsersLower)) + '\t' + '1.0' + '\n')                  
             
        if build_networkx_graph:
            DRAW_NETWORK = False
            if DRAW_NETWORK:
                #pos=nx.spring_layout(graph) # positions for all nodes
                pos = nx.spectral_layout(mention_graph)
                nx.draw_networkx_nodes(mention_graph,pos,
                           nodelist=range(0, len(trainUsersLower)),
                           node_color='g',
                           node_size=1,
                       alpha=0.8)
                nx.draw_networkx_nodes(mention_graph,pos,
                           nodelist=range(len(trainUsersLower), len(trainUsersLower) + len(u_text_unknown)),
                           node_color='r',
                           node_size=1,
                       alpha=0.8)
                nx.draw_networkx_edges(mention_graph, pos,
                            edgelist=None,
                            width=0.01,
                            edge_color='k',
                            style='solid',
                            alpha=0.1,
                            edge_cmap=None,
                            edge_vmin=None,
                            edge_vmax=None,
                            ax=None,
                            arrows=False,
                            label=None)
                #nx.draw(graph, pos) 
                #print "saving the graph in " + GEOTEXT_HOME + "/Graph.pdf"
                plt.savefig(GEOTEXT_HOME + "/graph.pdf", format="pdf")
                plt.close()
            
            print "the number of components is %d" %(len(list(nx.connected_components(mention_graph))))
            
            #degree
            DRAW_DEGREE = False
            if DRAW_DEGREE:
                print "computing degrees of connectivity"
                degrees = sorted(mention_graph.degree().values())
                degree_counter = Counter(degrees)
                labels, values = zip(*degree_counter.items())
                indexes = np.arange(len(labels))
                width = 1
                plt.bar(indexes, values, width)
                plt.xticks(indexes + width * 0.5, labels)
                plt.savefig(GEOTEXT_HOME + '/degrees.pdf', format='pdf') 
                plt.close()
                    
            #shortest paths
            DRAW_SHORTEST_PATH = False
            if DRAW_SHORTEST_PATH:
                shortest_paths = nx.all_pairs_shortest_path_length(mention_graph, cutoff=None)
            
    if DIRECT_GRAPH_WEIGHTED:
        doubles = 0
        trainIdx = range(len(trainUsersLower))
        trainUsersLowerDic = dict(zip(trainUsersLower, trainIdx))
        if DEVELOPMENT:
            devStr = '.dev'
            for i in range(0, len(devUsersLower)):
                u = devUsersLower[i]
                if u in trainUsersLowerDic:
                    devUsersLower[i] = u + '_double00'
                    doubles += 1
            u_unknown = devUsersLower
            u_text_unknown = devText
            U_all = U_train + U_dev
        else:
            for i in range(0, len(testUsersLower)):
                u = testUsersLower[i]
                if u in trainUsersLowerDic:
                    testUsersLower[i] = u + '_double00'
                    doubles += 1
            u_text_unknown = testText
            u_unknown = testUsersLower
            U_all = U_train + U_test  
        U_all_lower = [u.lower() for u in U_all]
        print "The number of test users found in train users is " + str(doubles)
        vocab_cnt = len(U_all)
        idx = range(vocab_cnt)
        if build_networkx_graph:
            mention_graph = nx.Graph()
            mention_graph.add_nodes_from(idx)
        node_id = dict(zip(U_all, idx))
        node_lower_id = {}
        #data and indices of a coo matrix to be populated
        coordinates = Counter()
        data = []
        for node, id in node_id.iteritems():
            node_lower_id[node.lower()] = id
        assert (len(node_id) == len(trainUsersLower) + len(u_unknown)), 'number of unique users is not eq u_train + u_test'
        print "the number of nodes is " + str(vocab_cnt)
        print "building the direct graph"
        token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        token_pattern1 = re.compile(token_pattern1)
        mention_users = defaultdict(Counter)
        l = len(trainText)
        tenpercent = l / 10
        i = 1
        for user, text in trainText.iteritems():
            user_id = node_id[user]
            if i % tenpercent == 0:
                print str(10 * i / tenpercent) + "%"
            i += 1  
            mentions = [u.lower() for u in token_pattern1.findall(text)] 
            mentionDic = Counter(mentions)
            for mention in mentionDic:
                # check if mention is a user node
                mention_id = node_lower_id.get(mention, -1)
                if mention_id != -1:    
                    if mention_id != user_id:
                        if user_id < mention_id:
                            coordinates[(user_id, mention_id)] += 1
                        elif mention_id < user_id:
                            coordinates[(mention_id, user_id)] += 1
                    
                mention_users[mention][user_id] += 1
        
            
        print "adding the test graph"
        for user, text in u_text_unknown.iteritems():
            user_id = node_id[user]
            mentions = [u.lower() for u in token_pattern1.findall(text)]
            mentionDic = Counter(mentions)
            for mention in mentionDic:
                mention_id = node_lower_id.get(mention, -1)
                if mention_id != -1:
                    if mention_id != user_id:
                        if user_id < mention_id:
                            coordinates[(user_id, mention_id)] += 1
                        elif mention_id < user_id:
                            coordinates[(mention_id, user_id)] += 1
                mention_users[mention][user_id] += 1
        
        
        
        

        print "setting weighted relationships."
        l = len(mention_users)
        tenpercent = l / 10
        i = 1
        celebrities_count = 0
        for mention, user_ids in mention_users.iteritems():
            if i % tenpercent == 0:
                print str(10 * i / tenpercent) + "%"
            i += 1  
            if len(user_ids) > CELEBRITY_THRESHOLD:
                celebrities_count += 1
                continue
            
            for user_id1, freq1 in user_ids.iteritems():
                for user_id2, freq2 in user_ids.iteritems():
                    if user_id1 < user_id2:
                        coordinates[(user_id1, user_id2)] += (freq1 + freq2)/2 
                    elif user_id2 < user_id1:
                        coordinates[(user_id2, user_id1)] += (freq1 + freq2)/2
        
        print "The number of celebrities is " + str(celebrities_count) + " ."
        rows = [row for row,col in coordinates]
        cols = [col for row,col in coordinates]
        data = [coordinates[(row,col)] for row,col in coordinates]
        print "The number of edges is " + str(len(rows))
        data = np.asarray(data)
        rows = np.asarray(rows)
        cols = np.asarray(cols)  
        #Tracer()()               
        pairs = coo_matrix((data,(rows,cols)), shape=(rows.shape[0],cols.shape[0]), dtype=np.float)
        print "writing the weighted graph"
        xindx, yindx = pairs.nonzero()
        tenpercent = pairs.nnz / 10
        # xindx = xindx.tolist()
        # yindx = yindx.tolist()
        pairs_dtype = pairs.dtype
        with codecs.open(path.join(GEOTEXT_HOME, 'input_graph_' + str(BUCKET_SIZE) + devStr + text_str + weighted_str), 'w', 'ascii') as outf:
            i = 1
            for xindx, yindx, w in zip(pairs.row, pairs.col, pairs.data):
                # w = pairs[xindx, yindx]
                if i % tenpercent == 0:
                    print "processing " + str(10 * i / tenpercent) + "%"
                i += 1
                if pairs_dtype == np.bool_:
                    w = 1.0
                outf.write(str(xindx) + '\t' + str(yindx) + '\t' + str(w) + '\n')
                if build_networkx_graph:
                    mention_graph.add_edge(xindx, yindx, attr_dict = {'w':1})
            if ADD_TEXT_LEARNER and DONGLE:
                for i in range(0, len(dongle_nodes)):
                    outf.write(str(i + len(trainUsersLower))+'.T' + '\t' + str(i + len(trainUsersLower)) + '\t' + '1.0' + '\n')                  
             
        if build_networkx_graph:
            DRAW_NETWORK = False
            if DRAW_NETWORK:
                #pos=nx.spring_layout(graph) # positions for all nodes
                pos = nx.spectral_layout(mention_graph)
                nx.draw_networkx_nodes(mention_graph,pos,
                           nodelist=range(0, len(trainUsersLower)),
                           node_color='g',
                           node_size=1,
                       alpha=0.8)
                nx.draw_networkx_nodes(mention_graph,pos,
                           nodelist=range(len(trainUsersLower), len(trainUsersLower) + len(u_text_unknown)),
                           node_color='r',
                           node_size=1,
                       alpha=0.8)
                nx.draw_networkx_edges(mention_graph, pos,
                            edgelist=None,
                            width=0.01,
                            edge_color='k',
                            style='solid',
                            alpha=0.1,
                            edge_cmap=None,
                            edge_vmin=None,
                            edge_vmax=None,
                            ax=None,
                            arrows=False,
                            label=None)
                #nx.draw(graph, pos) 
                #print "saving the graph in " + GEOTEXT_HOME + "/Graph.pdf"
                plt.savefig(GEOTEXT_HOME + "/graph.pdf", format="pdf")
                plt.close()
            
            print "the number of components is %d" %(len(list(nx.connected_components(mention_graph))))
            
            #degree
            DRAW_DEGREE = False
            if DRAW_DEGREE:
                print "computing degrees of connectivity"
                degrees = sorted(mention_graph.degree().values())
                degree_counter = Counter(degrees)
                labels, values = zip(*degree_counter.items())
                indexes = np.arange(len(labels))
                width = 1
                plt.bar(indexes, values, width)
                plt.xticks(indexes + width * 0.5, labels)
                plt.savefig(GEOTEXT_HOME + '/degrees.pdf', format='pdf') 
                plt.close()
                    
            #shortest paths
            DRAW_SHORTEST_PATH = False
            if DRAW_SHORTEST_PATH:
                shortest_paths = nx.all_pairs_shortest_path_length(mention_graph, cutoff=None)
                    
  
    
def prepare_adsorption_data():
    global trainUsers
    global testUsers
    global trainText
    global testText
    option = 4
    print "option: " + str(option) 
    if option == 4:
        global mentions
        DEVELOPMENT = True
        extract_mentions(k=0, addTest=False, addDev=True)
        
        extract_mentions(k=0, addTest=True, addDev=False)
        trainUsersLower = [u.lower() for u in sorted(trainUsers)]
        testUsersLower = [u.lower() for u in sorted(testUsers)]
        devUsersLower = [u.lower() for u in sorted(devUsers)]
        if DEVELOPMENT:
            mentions = set(mentions) - set(trainUsersLower) - set(devUsersLower)
            mentions = trainUsersLower + devUsersLower + list(mentions)
        else:
            mentions = set(mentions) - set(trainUsersLower) - set(testUsersLower)
            mentions = trainUsersLower + testUsersLower + list(mentions)
        sortedNodes = mentions
        vocab_cnt = len(sortedNodes)
        idx = range(vocab_cnt)
        vocab = dict(zip(sortedNodes, idx))
        vectorizer = CountVectorizer(encoding=data_encoding, vocabulary=vocab, binary=False)
        X_train_relational, Y_train, U_train, X_dev_relational, Y_dev, U_dev, X_test_relational, Y_test, U_test, categories, feature_names_relational = feature_extractor2(use_mention_dictionary=True, use_idf=False, norm=None, binary=True, sublinear_tf=False, min_df=1, BuildCostMatrices=False, vectorizer=vectorizer)
        if DEVELOPMENT:
            print "producing the dev graph"
            X_relational = sparse.vstack([X_train_relational, X_dev_relational])
            U_all = U_train + U_dev                
            U = [u.lower() for u in U_all]
            userDic = dict(zip(U, range(0, len(U))))
            print "writing id_user file"
            with codecs.open(path.join(GEOTEXT_HOME, 'id_user_' + str(BUCKET_SIZE) + '.dev'), 'w', 'utf-8') as outf:
                for u in U:
                    outf.write(str(userDic[u]) + '\t' + u + '\n')
    
            assert (len(U) == X_relational.shape[0]), 'number of users is not eq to the number of samples'
            assert (vocab_cnt == X_relational.shape[1]), 'the number of features is not equal to the number of mentions'
            already_added = []
            print "writing seed file"
            with codecs.open(path.join(GEOTEXT_HOME, 'seeds_' + str(BUCKET_SIZE) + '.dev'), 'w', 'ascii') as outf:
                for i in range(0, Y_train.shape[0]):
                    outf.write(str(i) + '\t' + str(Y_train[i]) + '\t' + '1.0' + '\n')
            print "writing gold_labels"
            with codecs.open(path.join(GEOTEXT_HOME, 'gold_labels_' + str(BUCKET_SIZE) + '.dev'), 'w', 'ascii') as outf:
                for i in range(0, Y_test.shape[0]):
                    outf.write(str(i + Y_train.shape[0]) + '\t' + str(Y_test[i]) + '\t' + '1.0' + '\n')
            X_relational = X_relational.tolil()
            print "writing input_graph"
            print "number of nonzero cells is " + str(X_relational.nnz)
            tenpercent = X_relational.nnz / 100
            xindx, yindx = X_relational.nonzero()
            xindx = xindx.tolist()
            yindx = yindx.tolist()
            JUST_USERS = False
            if JUST_USERS:
                with codecs.open(path.join(GEOTEXT_HOME, 'input_graph_' + str(BUCKET_SIZE) + '.dev'), 'w', 'ascii') as outf:
                    i = 1
                    for x, y in zip(xindx, yindx):
                        if i % tenpercent == 0:
                            print "processing " + str(i / (10 * tenpercent)) + "%"
                        i += 1
                        d_id = data_id(x, y)
                        mention = sortedNodes[y]
                        u = U[x]
                        w = X_relational[x, y]
                        X_relational[x, y] = 0
                        if mention in U and u in sortedNodes:
                            w_r = X_relational[userDic[mention], vocab[u]]
                            w += w_r
                            if w_r > 0:
                                X_relational[userDic[mention], vocab[u]] = 0   
                        if w > 0:
                            outf.write(d_id + '\t' + str(w) + '\n')        
            else:
                with codecs.open(path.join(GEOTEXT_HOME, 'input_graph_' + str(BUCKET_SIZE) + '.dev'), 'w', 'ascii') as outf:
                    i = 1
                    user_user_mention_count = 0
                    for x, y in zip(xindx, yindx):
                        if i % tenpercent == 0:
                            print "processing " + str(i / tenpercent) + "%"
                        i += 1
                        d_id = data_id(x, y)
                        w = X_relational[x, y]
                        X_relational[x, y] = 0
                        if y < len(U):
                            user_user_mention_count += 1
                            w_r = X_relational[y, x]
                            w += w_r
                            if w_r > 0:
                                X_relational[y, x] = 0   
                        if w > 0:
                            outf.write(d_id + '\t' + str(w) + '\n')    
            print "development data is ready."  

   
            
        # test graph
        print "producing the test graph"
        X_relational = sparse.vstack([X_train_relational, X_test_relational])
        U_all = U_train + U_test                
        U = [u.lower() for u in U_all]
        userDic = dict(zip(U, range(0, len(U))))
        print "writing id_user file"
        with codecs.open(path.join(GEOTEXT_HOME, 'id_user_' + str(BUCKET_SIZE)), 'w', 'utf-8') as outf:
            for u in U:
                outf.write(str(userDic[u]) + '\t' + u + '\n')

        assert (len(U) == X_relational.shape[0]), 'number of users is not eq to the number of samples'
        assert (vocab_cnt == X_relational.shape[1]), 'the number of features is not equal to the number of mentions'
        already_added = []
        print "writing seed file"
        with codecs.open(path.join(GEOTEXT_HOME, 'seeds_' + str(BUCKET_SIZE)), 'w', 'ascii') as outf:
            for i in range(0, Y_train.shape[0]):
                outf.write(str(i) + '\t' + str(Y_train[i]) + '\t' + '1.0' + '\n')
        print "writing gold_labels"
        with codecs.open(path.join(GEOTEXT_HOME, 'gold_labels_' + str(BUCKET_SIZE)), 'w', 'ascii') as outf:
            for i in range(0, Y_test.shape[0]):
                outf.write(str(i + Y_train.shape[0]) + '\t' + str(Y_test[i]) + '\t' + '1.0' + '\n')
        X_relational = X_relational.tolil()
        print "writing input_graph"
        print "number of nonzero cells is " + str(X_relational.nnz)
        tenpercent = X_relational.nnz / 100
        xindx, yindx = X_relational.nonzero()
        xindx = xindx.tolist()
        yindx = yindx.tolist()
        JUST_USERS = False
        if JUST_USERS:
            with codecs.open(path.join(GEOTEXT_HOME, 'input_graph_' + str(BUCKET_SIZE)), 'w', 'ascii') as outf:
                i = 1
                for x, y in zip(xindx, yindx):
                    if i % tenpercent == 0:
                        print "processing " + str(i / (10 * tenpercent)) + "%"
                    i += 1
                    d_id = data_id(x, y)
                    mention = sortedNodes[y]
                    u = U[x]
                    w = X_relational[x, y]
                    X_relational[x, y] = 0
                    if mention in U and u in sortedNodes:
                        w_r = X_relational[userDic[mention], vocab[u]]
                        w += w_r
                        if w_r > 0:
                            X_relational[userDic[mention], vocab[u]] = 0   
                    if w > 0:
                        outf.write(d_id + '\t' + str(w) + '\n')        
        else:
            with codecs.open(path.join(GEOTEXT_HOME, 'input_graph_' + str(BUCKET_SIZE)), 'w', 'ascii') as outf:
                i = 1
                user_user_mention_count = 0
                for x, y in zip(xindx, yindx):
                    if i % tenpercent == 0:
                        print "processing " + str(10 * i / tenpercent) + "%"
                        print user_user_mention_count
                    i += 1
                    d_id = data_id(x, y)
                    w = X_relational[x, y]
                    X_relational[x, y] = 0
                    if y < len(U):
                        user_user_mention_count += 1
                        w_r = X_relational[y, x]
                        w += w_r
                        if w_r > 0:
                            X_relational[y, x] = 0   
                    if w > 0:
                        outf.write(d_id + '\t' + str(w) + '\n')      

    elif option == 3:

        '''
        extract_mentions(k=0, addTest=True, addDev=False)
        sortedNodes = sorted(mentions)
        vocab_cnt = len(sortedNodes)
        idx = range(vocab_cnt)
        vocab = dict(zip(sortedNodes, idx))
        vectorizer = CountVectorizer(encoding=data_encoding, vocabulary=vocab, binary=False)
        X_train_relational, Y_train, U_train, X_dev_relational, Y_dev, U_dev, X_test_relational, Y_test, U_test, categories, feature_names_relational = feature_extractor2(use_mention_dictionary=True, use_idf=False, norm=None, binary=True, sublinear_tf=False, min_df=1, BuildCostMatrices=False, vectorizer=vectorizer)
        '''
        '''
        results cmu 
        not weighted:
        mean distance is 668.581279917
        median distance is 277.116433909
        weighted
        mean distance is 676.225954801
        median distance is 255.724939021
        
        NA
        mean distance is 748.483205478
        median distance is 449.083574218
        
        '''
        weighted = True
        print "building the direct graph"
        token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        token_pattern1 = re.compile(token_pattern1)
        # token_pattern2 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))#([A-Za-z]+[A-Za-z0-9_]+)'
        # token_pattern2 = re.compile(token_pattern2)
        netgraph = {}
        l = len(trainText)
        tenpercent = l / 10
        i = 1
        for user, text in trainText.iteritems():
            user = user.lower()
            if i % tenpercent == 0:
                print str(10 * i / tenpercent) + "%"
            i += 1  
            mentions = [u.lower() for u in token_pattern1.findall(text)] 
            mentionDic = Counter(mentions)
            for mention in mentionDic:
                if mention > user:
                    k = user + '\t' + mention
                    netgraph[k] = netgraph.get(k, 0) + mentionDic[mention]
                else:
                    k = mention + '\t' + user
                    netgraph[k] = netgraph.get(k, 0) + mentionDic[mention]
            
        print "adding the test graph"
        for user, text in testText.iteritems():
            user = user.lower()
            mentions = [u.lower() for u in token_pattern1.findall(text)]
            mentionDic = Counter(mentions)
            for mention in mentionDic:
                if mention > user:
                    k = user + '\t' + mention
                    netgraph[k] = netgraph.get(k, 0) + mentionDic[mention]
                else:
                    k = mention + '\t' + user
                    netgraph[k] = netgraph.get(k, 0) + mentionDic[mention]
        train_test_users = sorted([u.lower() for u in trainUsers.keys()]) + sorted([u.lower() for u in testUsers.keys()])
        keysStr = '\t'.join(train_test_users + netgraph.keys())
        U = set(keysStr.split('\t'))
        userDic = dict(zip(train_test_users, range(0, len(train_test_users))))
        mentions_non_user = [u for u in U if u not in userDic]
        mentionDic = dict(zip(mentions_non_user, range(len(train_test_users), len(train_test_users + mentions_non_user))))
        trainIndexClasses = {}
        testIndexClasses = {}
        for u, c in trainClasses.iteritems():
            u = u.lower()
            idx = userDic[u]
            trainIndexClasses[idx] = c

        for u, c in testClasses.iteritems():
            u = u.lower()
            idx = userDic[u]
            testIndexClasses[idx] = c
        # Tracer()()
        print "writing id_user file"
        with codecs.open(path.join(GEOTEXT_HOME, 'id_user'), 'w', 'utf-8') as outf:
            for u in train_test_users:
                outf.write(str(userDic[u]) + '\t' + u + '\n')
        print "writing seed file"
        with codecs.open(path.join(GEOTEXT_HOME, 'seeds'), 'w', 'ascii') as outf:
            for i, c in trainIndexClasses.iteritems():
                outf.write(str(i) + '\t' + str(c) + '\t' + '1.0' + '\n')
        print "writing gold_labels"
        with codecs.open(path.join(GEOTEXT_HOME, 'gold_labels'), 'w', 'ascii') as outf:
            for i, c in testIndexClasses.iteritems():
                outf.write(str(i) + '\t' + str(c) + '\t' + '1.0' + '\n')
        print "writing input_graph"
        with codecs.open(path.join(GEOTEXT_HOME, 'input_graph'), 'w', 'ascii') as outf:        
            for nodes, w in netgraph.iteritems():
                nodes = nodes.split('\t')
                nodeIndices = []
                if nodes[0] in userDic:
                    nodeIndices.append(userDic[nodes[0]])
                else:
                    nodeIndices.append(mentionDic[nodes[0]])

                if nodes[1] in userDic:
                    nodeIndices.append(userDic[nodes[1]])
                else:
                    nodeIndices.append(mentionDic[nodes[1]])
                    
                outf.write(str(nodeIndices[0]) + '\t' + str(nodeIndices[1]) + '\t' + str(w) + '\n')
        Tracer()()
        
            

def direct_graph():
    '''
    results cmu 
    not weighted:
    mean distance is 668.581279917
    median distance is 277.116433909
    weighted
    mean distance is 676.225954801
    median distance is 255.724939021
    
    '''
    graph_file_address = path.join(GEOTEXT_HOME, 'direct_graph')
    if os.path.exists(graph_file_address):
        print "reading netgraph from pickle"
        with open(graph_file_address, 'rb') as inf:
            netgraph, trainUsers, testUsers = pickle.load(inf)
    else:
        print "building the direct graph"
        token_pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        token_pattern = re.compile(token_pattern)
        netgraph = {}
        l = len(trainText)
        tenpercent = l / 10
        i = 1
        for user, text in trainText.iteritems():
            user = user.lower()
            if i % tenpercent == 0:
                print str(10 * i / tenpercent) + "%"
            i += 1
                
            mentions = [u.lower() for u in token_pattern.findall(text)]
            mentionDic = Counter(mentions)
            for mention in mentionDic:
                if mention > user:
                    k = user + '\t' + mention
                    netgraph[k] = netgraph.get(k, 0) + mentionDic[mention]
                else:
                    k = mention + '\t' + user
                    netgraph[k] = netgraph.get(k, 0) + mentionDic[mention]
        print "adding the test graph"
        for user, text in testText.iteritems():
            mentions = [u.lower() for u in token_pattern.findall(text)]
            mentionDic = Counter(mentions)
            for mention in mentionDic:
                if mention > user:
                    k = user + '\t' + mention
                    netgraph[k] = netgraph.get(k, 0) + mentionDic[mention]
                else:
                    k = mention + '\t' + user
                    netgraph[k] = netgraph.get(k, 0) + mentionDic[mention]
        
        print "pickling the graph with edge size " + str(len(netgraph))
        with open(path.join(GEOTEXT_HOME, 'direct_graph'), 'wb') as outf:
            pickle.dump((netgraph, trainUsers, testUsers), outf)    
    
    trainUsersLower = {}
    testUsersLower = {}
    trainLats = []
    trainLons = []
    node_location = {}
    
    for user, loc in trainUsers.iteritems():
        user = user.lower()
        lat, lon = locationStr2Float(loc)
        trainLats.append(lat)
        trainLons.append(lon)
        trainUsersLower[user] = (lat, lon)
        node_location[user] = (lat, lon)
        
    for user, loc in testUsers.iteritems():
        user = user.lower()
        lat, lon = locationStr2Float(loc)
        testUsersLower[user] = (lat, lon)

    medianLat = median(trainLats)
    medianLon = median(trainLons)
    
    from pygraph.classes.graph import graph
    gr = graph()
    print "adding nodes"
    nodes = []
    tenpercent = len(netgraph) / 10
    i = 1
    for edg in netgraph:
        if i % tenpercent == 0:
            print str(10 * i / tenpercent) + "%"
        i += 1            
        n1n2 = edg.split('\t')
        n1 = n1n2[0]
        n2 = n1n2[1]
        nodes.append(n1)
        nodes.append(n2)
    
    nodes = sets.Set(nodes)
    nodes_known = trainUsersLower.keys()
    nodes_unknown = [node for node in nodes if node not in nodes_known]
    gr.add_nodes(nodes)    
    print "adding edges"
    tenpercent = len(netgraph) / 10
    i = 1
    for edg in netgraph:
        if i % tenpercent == 0:
            print str(10 * i / tenpercent) + "%"
        i += 1            
        n1n2 = edg.split('\t')
        n1 = n1n2[0]
        n2 = n1n2[1]
        edge = (n1, n2)
        gr.add_edge(edge, netgraph[edg], label='', attrs=[])

    converged = False
    weighted = True
    print "weighted " + str(weighted)
    max_iter = 30
    iter_num = 1
    print "iterating with max_iter = " + str(max_iter)
    while not converged:
        print "iter: " + str(iter_num)
        located_nodes_count = len(node_location)
        print str(located_nodes_count) + " nodes have location"
        for node in nodes_unknown:
            nbrs = gr.neighbors(node)
            nbrlats = []
            nbrlons = []
            for nbr in nbrs:
                if nbr in node_location:
                    lat, lon = node_location[nbr]
                    w = 1
                    if weighted:
                        w = gr.get_edge_properties((nbr, node))['weight']
                    for i in range(0, w):
                        nbrlats.append(lat)
                        nbrlons.append(lon)
            if len(nbrlons) > 0:
                node_location[node] = (median(nbrlats), median(nbrlons))
        iter_num += 1
        if iter_num == max_iter:
            converged = True
        if len(node_location) == located_nodes_count:
            print "converged. No new nodes added in this iteration."
            # converged = True
        distances = []
        for user, loc in testUsersLower.iteritems():
            lat, lon = loc
            predicted_lat, predicted_lon = node_location.get(user, (medianLat, medianLon))
            dist = distance(lat, lon, predicted_lat, predicted_lon)
            distances.append(dist)
        
        print "mean distance is " + str(mean(distances))
        print "median distance is " + str(median(distances))
    print "pickling (testUsersLower, node_location)"
    with open(path.join(GEOTEXT_HOME, 'node_location.pkl'), 'wb') as outf:
        pickle.dump((testUsersLower, node_location), outf)
def save_graph(gr):
    i = 0
    tenpercent = len(gr) / 10
    nodes = sets.Set()
    node_numbers = {}
    for edg, weight in gr.iteritems():
        if i % tenpercent == 0:
            print str(10 * i / tenpercent) + "%"
        i += 1          
        n1n2 = edg.split('\t')
        n1 = n1n2[0]
        n2 = n1n2[1]
        nodes.add(n1)
        nodes.add(n2)
    i = 0
    for node in nodes:
        node_numbers[node] = str(i).strip()
        i += 1
    
    i = 0
    print "writing edges. the number of edges is " + str(len(gr))
    with codecs.open(path.join(GEOTEXT_HOME, 'edges.txt'), 'w', 'ascii') as outf:
        for edg, weight in gr.iteritems():
            if i % tenpercent == 0:
                print str(10 * i / tenpercent) + "%"
            i += 1          
            n1n2 = edg.split('\t')
            n1 = node_numbers[n1n2[0]]
            n2 = node_numbers[n1n2[1]]
            w = 1
            weighted = True
            if weighted:
                w = weight
            outf.write(n1 + '\t' + n2 + '\t' + str(w) + '\n')
    print "writing nodes. the number of nodes is " + str(len(nodes))
    with codecs.open(path.join(GEOTEXT_HOME, 'nodes.txt'), 'w', 'ascii') as outf:
        for node in nodes:
            outf.write(node_numbers[node] + '\n')
def ideal_network_errors():
    graph_file_address = path.join(GEOTEXT_HOME, 'direct_graph')
    if os.path.exists(graph_file_address):
        print "reading netgraph from pickle"
        with open(graph_file_address, 'rb') as inf:
            netgraph, trainUsers, testUsers = pickle.load(inf)
    ideal_distances = []
    acc161 = 0
    tenpercent = len(testUsers) / 10
    i = 0
    for utest, uloc in testUsers.iteritems():
        i += 1
        if i % tenpercent == 0:
            print str(100 * i / len(testUsers))
        lat1, lon1 = locationStr2Float(uloc)
        dists = []
        for utrain, utrainloc in trainUsers.iteritems():
            lat2, lon2 = locationStr2Float(utrainloc)
            d = distance(lat1, lon1, lat2, lon2)
            dists.append(distance(lat1, lon1, lat2, lon2))
            if d < 1:
                break
        minDist = min(dists)
        if minDist < 161:
            acc161 += 1
        ideal_distances.append(minDist)
    print "distance number" + str(len(ideal_distances))
    print "mean " + str(mean(ideal_distances))
    print "median " + str(median(ideal_distances))
    print "Acc @ 161 " + str((acc161 + 0.0) / len(ideal_distances))
    
    print "reading prior text-based locations"
    prior_file_path = path.join(GEOTEXT_HOME, 'preds.pkl')

                
def direct_graph2():
    global trainUsers
    global testUsers
    global trainText
    global testText
    save_gr = False

    '''
    results cmu 
    not weighted:
    mean distance is 668.581279917
    median distance is 277.116433909
    weighted
    mean distance is 676.225954801
    median distance is 255.724939021
    
    NA
    mean distance is 748.483205478
    median distance is 449.083574218
    
    '''
    weighted = True
    graph_file_address = path.join(GEOTEXT_HOME, 'direct_graph')
    if os.path.exists(graph_file_address):
        print "reading netgraph from pickle"
        with open(graph_file_address, 'rb') as inf:
            netgraph, trainUsers, testUsers = pickle.load(inf)
            if save_gr:
                print "saving the graph"
                save_graph(netgraph)
    else:
        print "building the direct graph"
        token_pattern1 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        token_pattern1 = re.compile(token_pattern1)
        token_pattern2 = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))#([A-Za-z]+[A-Za-z0-9_]+)'
        token_pattern2 = re.compile(token_pattern2)
        netgraph = {}
        l = len(trainText)
        tenpercent = l / 10
        i = 1
        for user, text in trainText.iteritems():
            user = user.lower()
            if i % tenpercent == 0:
                print str(10 * i / tenpercent) + "%"
            i += 1  
            mentions = [u.lower() for u in token_pattern1.findall(text)] 
            mentionDic = Counter(mentions)
            for mention in mentionDic:
                if mention > user:
                    k = user + '\t' + mention
                    netgraph[k] = netgraph.get(k, 0) + mentionDic[mention]
                else:
                    k = mention + '\t' + user
                    netgraph[k] = netgraph.get(k, 0) + mentionDic[mention]
            
        print "adding the test graph"
        for user, text in testText.iteritems():
            mentions = [u.lower() for u in token_pattern1.findall(text)]
            mentionDic = Counter(mentions)
            for mention in mentionDic:
                if mention > user:
                    k = user + '\t' + mention
                    netgraph[k] = netgraph.get(k, 0) + mentionDic[mention]
                else:
                    k = mention + '\t' + user
                    netgraph[k] = netgraph.get(k, 0) + mentionDic[mention]
        
        print "pickling the graph with edge size " + str(len(netgraph))
        with open(path.join(GEOTEXT_HOME, 'direct_graph'), 'wb') as outf:
            pickle.dump((netgraph, trainUsers, testUsers), outf)    
    
    trainUsersLower = {}
    testUsersLower = {}
    trainLats = []
    trainLons = []
    node_location = {}
    
    PRIOR = False
    if PRIOR:
        print "reading prior text-based locations"
        prior_file_path = path.join(GEOTEXT_HOME, 'preds.pkl')
        if os.path.exists(prior_file_path):
            with open(prior_file_path, 'rb') as inf:
                test_user_prior_locations = pickle.load(inf)
                for user, location in test_user_prior_locations.iteritems():
                    if user in testUsers:
                        node_location[user.lower()] = location
                    else:
                        print "prior information of user couldn't be assigned because user was not found in test users"
        else:
            print "prior file not found."
    for user, loc in trainUsers.iteritems():
        user = user.lower()
        lat, lon = locationStr2Float(loc)
        trainLats.append(lat)
        trainLons.append(lon)
        trainUsersLower[user] = (lat, lon)
        node_location[user] = (lat, lon)
        
    for user, loc in testUsers.iteritems():
        user = user.lower()
        lat, lon = locationStr2Float(loc)
        testUsersLower[user] = (lat, lon)
    
    
    print "the number of train nodes is " + str(len(trainUsers))
    print "the number of test nodes is " + str(len(testUsers))
    medianLat = median(trainLats)
    medianLon = median(trainLons)
    
    print "building the graph"
    graphDic = {}
    tenpercent = len(netgraph) / 10
    i = 1
    for edg in netgraph:
        if i % tenpercent == 0:
            print str(10 * i / tenpercent) + "%"
        i += 1          
        n1n2 = edg.split('\t')
        n1 = n1n2[0]
        n2 = n1n2[1]
        w = 1
        if weighted:
            w = netgraph[edg]
        n1_nbrs = graphDic.get(n1, [])
        n2_nbrs = graphDic.get(n2, [])
        for j in range(0, w):
            n1_nbrs.append(n2)
            n2_nbrs.append(n1)
        graphDic[n1] = n1_nbrs
        graphDic[n2] = n2_nbrs
        
    print "finding unlocated nodes"
    nodes_unknown = [node for node in graphDic if node not in trainUsersLower]
  

    converged = False
    print "weighted " + str(weighted)
    max_iter = 10
    iter_num = 1
    print "iterating with max_iter = " + str(max_iter)
    while not converged:
        print "iter: " + str(iter_num)
        located_nodes_count = len(node_location)
        print str(located_nodes_count) + " nodes have location"
        for node in nodes_unknown:
            nbrs = graphDic.get(node, [])
            nbrlats = []
            nbrlons = []
            for nbr in nbrs:
                if nbr in node_location:
                    lat, lon = node_location[nbr]
                    nbrlats.append(lat)
                    nbrlons.append(lon)
            if len(nbrlons) > 0:
                node_location[node] = (median(nbrlats), median(nbrlons))
        iter_num += 1
        if iter_num == max_iter:
            converged = True
        if len(node_location) == located_nodes_count:
            print "converged. No new nodes added in this iteration."
            # converged = True
        distances = []
        isolated = 0
        for user, loc in testUsersLower.iteritems():
            lat, lon = loc
            if user not in node_location:
                Tracer()()
                isolated += 1
            predicted_lat, predicted_lon = node_location.get(user, (medianLat, medianLon))
            dist = distance(lat, lon, predicted_lat, predicted_lon)
            distances.append(dist)
        
        print "mean distance is " + str(mean(distances))
        print "median distance is " + str(median(distances))
        print "Accuracy @ 161 k.m. is " + str(100 * len([d for d in distances if d < 161]) / float(len(distances)))
        print "isolated test users are " + str(isolated)
    print "pickling (testUsersLower, node_location2)"
    with open(path.join(GEOTEXT_HOME, 'node_location2.pkl'), 'wb') as outf:
        pickle.dump((testUsersLower, node_location, graphDic), outf)   
def junto_postprocessing(multiple=False, dev=False, text_confidence=0.1):
    EVALUATE_REAL_VALUED = False
    global Y_dev
    global Y_test
    global U_dev
    global U_test
    text_errors = {}
    text_preds = None
    text_probs = None
    network_errors = {}
    nodes_degree = {}
    degree_errors = defaultdict(list)
    textlength_errors = defaultdict(list)
    user_text_probs = {}
    trainUsersLower = [u.lower() for u in sorted(trainUsers)]
    U_test = [u for u in sorted(testUsers)]
    U_dev = [u for u in sorted(devUsers)]
    result_dump_file = path.join(GEOTEXT_HOME, 'results-' + DATASETS[DATASET_NUMBER-1] + '-' + str(BUCKET_SIZE) + '.pkl')
    if text_confidence < 1:
        print "reading (preds, devPreds, U_test, U_dev, testProbs, devProbs) from " + result_dump_file
        with open(result_dump_file, 'rb') as inf:
            preds_text, devPreds_text, U_test_text, U_dev_text, testProbs_text, devProbs_text = pickle.load(inf)
            if dev:
                text_preds = devPreds_text
                text_probs = devProbs_text
                U_eval = U_dev
            else:
                text_preds = preds_text
                text_probs = testProbs_text
                U_eval = U_test
    
    # split a training set and a test set
    Y_test = np.asarray([testClasses[u] for u in U_test])
    Y_dev = np.asarray([devClasses[u] for u in U_dev])
    
    devStr = ''
    if dev:
        devStr = '.dev'
    if not multiple:
        files = ['/home/arahimi/git/junto-master/examples/simple/data/label_prop_output_' + DATASETS[DATASET_NUMBER - 1] + '_' + str(BUCKET_SIZE) + devStr]
    else:
        junto_output_dir = '/home/arahimi/git/junto-master/examples/simple/data/outputs_' + DATASETS[DATASET_NUMBER - 1]
        files = glob.glob(junto_output_dir + '/label_prop_output*')
        files = sorted(files)
    # feature_extractor2(min_df=50)

    for junto_output_file in files:   
        id_name_file = '/home/arahimi/git/junto-master/examples/simple/data/' + DATASETS[DATASET_NUMBER - 1] + '/id_user_' + str(BUCKET_SIZE) 
        if dev:
            id_name_file = id_name_file + '.dev'
        print "output file: " + junto_output_file
        print "id_name file: " + id_name_file
        name_id = {}
        id_pred = {}
        name_pred = {}
        id_name = {}
        with codecs.open(id_name_file, 'r', 'utf-8') as inf:
            for line in inf:
                fields = line.split()
                name_id[fields[1]] = fields[0]
                id_name[fields[0]] = fields[1]
        
        if dev:
            usersLower = [u.lower() for u in U_dev]
        else:
            usersLower = [u.lower() for u in U_test]
        dummy_count = 0
        with codecs.open(junto_output_file, 'r', 'utf-8') as inf:                 
            # real valued results were not good
            if EVALUATE_REAL_VALUED:
                distances = []
                for line in inf:
                    fields = line.split()
                    if len(fields) == 11:
                        uid = fields[0]
                        if '.T' in uid:
                            continue
                        lat = float(fields[4])
                        lon = float(fields[8])
                        u = U_test[int(uid) - len(trainUsers)]
                        
                        lat2, lon2 = locationStr2Float(userLocation[u])
                        distances.append(distance(lat, lon, lat2, lon2))
                print "resultssssssssssssssssssssssss"
                print str(mean(distances))
                print str(median(distances))
            else:
                nopred = []
                if text_confidence < 1:
                    for u in usersLower:
                        user_index = usersLower.index(u)
                        text_predition = str(text_preds[user_index])
                        test_index = user_index
                        text_errors[u] = error(text_predition,U_eval[test_index] )
                        text_probability = np.max(text_probs[user_index])
                        user_text_probs[u] = text_probability
                    errors_t=[text_errors[u] for u in usersLower]
                    probs_t=[user_text_probs[u] for u in usersLower]
                    plt.plot(errors_t, probs_t, '.', markersize=1)
                    plt.xlabel('LR Error Distance in km')
                    axes = plt.gca()
                    axes.set_xlim([0,161])
                    plt.ylabel('LR Prediction Confidence')
                    plot_name = 'error_prob_text_161'
                    plot_file = GEOTEXT_HOME + '/' + plot_name
                    #scatter_histo(errors_t, probs_t, plot_file + '-diagrams.pdf')
                    #hist2d(errors_t, probs_t, plot_file + '-diagrams.pdf')
                    print "saving the plot in " + plot_file
                    plt.title(DATASETS[DATASET_NUMBER-1])
                    plt.savefig(plot_file + '.pdf', format='pdf')
                    plt.close()
                    plt.plot(probs_t, errors_t, '.', markersize=1)
                    plt.ylabel('LR Error Distance in km')
                    axes = plt.gca()
                    axes.set_ylim([0,161])
                    plt.xlabel('LR Prediction Confidence')
                    plot_name = 'prob_error_text_161'
                    plot_file = GEOTEXT_HOME + '/' + plot_name
                    print "saving the plot in " + plot_file
                    plt.title(DATASETS[DATASET_NUMBER-1])
                    plt.savefig(plot_file + '.pdf', format='pdf')
                    plt.close()
                    
                for line in inf:
                    fields = line.split('\t')
                    uid = fields[0]
                    if '.T' in uid:
                        continue
                    u = id_name[uid]
                    test_user_inconsistency = 0
                    if u in usersLower:
                    # if fields[-2] == 'true':

                        
                        label_scores = fields[-3]
                        label = fields[-3].split()[0]
                        labelProb = float16(fields[-3].split()[1])
                        #Tracer()()
                        if label == '__DUMMY__':
                            dummy_count += 1
                            label = str(len(categories) / 2)
                                
                        id_pred[fields[0]] = label
                    else:
                        if u not in trainUsersLower:
                            nopred.append((uid, u))
                print "no predition for these nodes:" + str(nopred)
                print "no prediction for above nodes."
                if text_confidence < 1:
                    acc_network = [u for u in network_errors if network_errors[u] < 161]
                    acc_text = [u for u in text_errors if text_errors[u] < 161]

                    n1t1 = [u for u in acc_text if u in acc_network]
                    n0t1 = [u for u in acc_text if u not in acc_network]
                    n1t0 = [u for u in acc_network if u not in acc_text]
                    n0t0 = [ u for u in usersLower if u not in acc_text if u not in acc_network]
                    print 'n1t1 ' + str(len(n1t1))
                    print 'n0t1 ' + str(len(n0t1))
                    print 'n1t0 ' + str(len(n1t0))
                    print 'n0t0 ' + str(len(n0t0)) 
                print "the number of inconsistent test users is " + str(test_user_inconsistency)
                
                DRAW_TEXT_VS_NETWORK = False
                if DRAW_TEXT_VS_NETWORK and text_confidence < 1:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    #ax.set_xlim([0, 161])
                    #ax.set_ylim([0, 161])
                    #ax.set_xscale('log')
                    #ax.set_yscale('log')
                    plt.xlabel('text-based prediction error distance in KM')
                    plt.ylabel('network-based prediction error distance in KM')
                    text_errors2 = [text_errors[u] for u in network_errors]
                    network_errors2 = [network_errors[u] for u in network_errors]
                    m, b = np.polyfit(text_errors2, network_errors2, 1)
                    plt.plot(text_errors2, [m * d + b for d in text_errors2], '-')
                    plt.plot(text_errors2, network_errors2, '.',  markersize=1)
                    print "text vs network diagram saved in " + GEOTEXT_HOME + '/text_vs_network.pdf'
                    plt.savefig(GEOTEXT_HOME + '/text_vs_network.pdf')
                # Tracer()()
                preds = []
                if dev:
                    print "working on dev data, changing U_test to U_dev"
                    Users = U_dev
                else:
                    Users = U_test
                user_not_in_network = 0
                doubles_found = 0
                for u in Users:
                    if u.lower() not in name_id:
                        ud = u + '_double00'
                        doubles_found += 1
                        uid = name_id[ud.lower()]
                    else:
                        uid = name_id[u.lower()]
                    if uid in id_pred:
                        pred = id_pred[uid]
                        name_pred[u] = pred
                    else:
                        #print 'user %d not in network predictions.'
                        user_not_in_network += 1
                        pred = str(len(categories) / 2)  
                        name_pred[u] = pred

                    preds.append(int(pred))
                    if text_confidence< 1:
                        u_eval_id = int(uid) - len(trainUsers)
                        if u_eval_id < len(usersLower) and u_eval_id > -1:
                            network_errors[u.lower()] = error(pred,U_test[u_eval_id] )
                            if mention_graph!=None:
                                n_degree = mention_graph.degree(int(uid))
                                nodes_degree[u.lower()] = n_degree
                                degree_errors[n_degree].append(network_errors[u.lower()])
                        
                        else:
                            print(Fore.RED + 'fatal error u_eval_id not in range')
                            print(Fore.RESET)
                #Tracer()()
                print "doubles found is " + str(doubles_found)
                print "users with dummy labels: " + str(dummy_count)
                print "users not in network: " + str(user_not_in_network)
                print "total number of users: " + str(len(usersLower))
                # print preds
                # print [int(i) for i in Y_test.tolist()]    
                loss(preds, Users)
                DRAW_DEGREE_ERRORS = False
                if DRAW_DEGREE_ERRORS:
                    x = []
                    y1 = []
                    y2 = []
                    degrees = sorted(degree_errors.keys())
                    for degree in degrees:
                        errors = degree_errors[degree]
                        x.append(degree)
                        y1.append(np.mean(errors))
                        y2.append(len(errors))
                    plt.plot(x,y1,'r-')
                    plt.plot(x, y2, 'k-')
                    plot_name = 'degree_error'
                    plot_file = GEOTEXT_HOME + '/' + plot_name 
                    #xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                    #xs_labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
                    #ax = plt.gca()
                    #y1 = [25.0, 28.3, 30.7, 31.2, 33.1, 36.0, 37.6, 38.8, 38.9, 39.7]
                    #y2 = [41.8, 45.41, 46.35, 47.72, 48.14, 48.33, 48.77, 49.25, 49.5, 49.6]
                    #y3 = [46.48, 51.72, 54.09, 55.57, 56.5, 57.04, 57.65, 58.0, 58.2, 58.4]
                    #y_labels = ['GEOTEXT', 'Twitter-US', 'Twitter-WORLD']
                    #p1 = plt.plot(xs, y1, 'k--', label=y_labels[0])
                    #p2 = plt.plot(xs, y2, 'r-', label=y_labels[1])
                    #p3 = plt.plot(xs, y3, 'b-', label=y_labels[2])
                    #plt.axis().xaxis.set_ticks(xs)
                    #plt.legend((y_labels[0], y_labels[1], y_labels[1]), 'upper left', shadow=True)
                    plt.xlabel('user degree in @-mention network')
                    plt.ylabel('distance error in km')
                    print "saving the plot in " + plot_file
                    plt.title(DATASETS[DATASET_NUMBER-1])
                    plt.savefig(plot_file + '.pdf', format='pdf')
                    plt.close()
                    plt.title(DATASETS[DATASET_NUMBER-1])
                    plt.plot(x, y2, 'r-')
                    plt.xlabel('user degree in @-mention network')
                    plt.ylabel('#users')
                    plt.savefig(plot_file + '_num_users.pdf', format='pdf')
                    
                DRAW_TEXTLENGTH_ERRORS = False
                if DRAW_TEXTLENGTH_ERRORS:
                    for u in Users:
                        if dev:
                            text = devText[u]
                        else:
                            text = testText[u]
                            Tracer()()
                        textlength_errors[len(text) / 100].append(text_errors[u.lower()])
                    
                    #Tracer()()
                    x = []
                    y1 = []
                    y2 = []
                    textlengths = sorted(textlength_errors.keys())
                    for textlength in textlengths:
                        errors = textlength_errors[textlength]
                        x.append(textlength)
                        y1.append(median(errors))
                        y2.append(len(errors))
                    plt.plot(x,y1,'r-', markersize=0.2)
                    plt.plot(x, y2, 'k-', markersize=0.2)
                    plot_name = 'textlength_error'
                    plot_file = GEOTEXT_HOME + '/' + plot_name 
                    #xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                    #xs_labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
                    #ax = plt.gca()
                    #y1 = [25.0, 28.3, 30.7, 31.2, 33.1, 36.0, 37.6, 38.8, 38.9, 39.7]
                    #y2 = [41.8, 45.41, 46.35, 47.72, 48.14, 48.33, 48.77, 49.25, 49.5, 49.6]
                    #y3 = [46.48, 51.72, 54.09, 55.57, 56.5, 57.04, 57.65, 58.0, 58.2, 58.4]
                    #y_labels = ['GEOTEXT', 'Twitter-US', 'Twitter-WORLD']
                    #p1 = plt.plot(xs, y1, 'k--', label=y_labels[0])
                    #p2 = plt.plot(xs, y2, 'r-', label=y_labels[1])
                    #p3 = plt.plot(xs, y3, 'b-', label=y_labels[2])
                    #plt.axis().xaxis.set_ticks(xs)
                    #plt.legend((y_labels[0], y_labels[1], y_labels[1]), 'upper left', shadow=True)
                    plt.xlabel('user text length in test set')
                    plt.ylabel('median distance error in km')
                    print "saving the plot in " + plot_file
                    plt.title(DATASETS[DATASET_NUMBER-1])
                    plt.savefig(plot_file + '.pdf', format='pdf')
                    plt.close()
                    plt.title(DATASETS[DATASET_NUMBER-1])
                    plt.plot(x, y2, 'r-', markersize=0.2)
                    plt.xlabel('user text length in test set')
                    plt.ylabel('#users')
                    plt.savefig(plot_file + '_num_users.pdf', format='pdf')
                    
                
                    

        # Tracer()()
def create_junto_config_files_for_tuning():
    '''read one config file
    for each configuration
        write a new config file
    '''
    junto_config_file = '/home/arahimi/git/junto-master/examples/simple/simple_config'
    with codecs.open(junto_config_file, 'r', 'latin1') as inf:
        conf_str = inf.read()
    mu2s = ['10', '1', '0.1']
    mu3s = ['0']
    for mu2 in mu2s:
        for mu3 in mu3s:
            result = conf_str.replace('mu2 = 1e-2', 'mu2 = ' + mu2)
            result = result.replace('mu3 = 1e-2', 'mu3 = ' + mu3)
            result = result.replace('data/label_prop_output', 'data/outputs_' + DATASETS[DATASET_NUMBER-1] + '/label_prop_output_' + mu2 + '_' + mu3)
            with codecs.open('/home/arahimi/git/junto-master/examples/simple/confs_' + DATASETS[DATASET_NUMBER-1] + '/simple_config_' + mu2 + '_' + mu3, 'w', 'latin1') as outf:
                outf.write(result)
    print "config files created!"

# test_re()
# euclidean()
# cross_validate()
# chart_me()
# sys.exit()
# normalizeText()
def analysis():
    num_users = 429000
    xs = [num_users/128, num_users/64, num_users/32, num_users/16, num_users/8, num_users/4, num_users/2, num_users]
    xs_labels = ['3k','6k', '13k', '26k', '53k', '107k', '214k', '429k']
    ax = plt.gca()
    acc_text = [27.4, 33.2, 37.2, 40.78, 41.7, 45, 48.19, 50.3]
    acc_network = [11.28, 14.2, 18.9, 25.2, 32.47, 40, 48.19, 53.9]
    p1 = plt.plot(xs, acc_text, 'k--', label='LR')
    p2 = plt.plot(xs, acc_network, 'r-', label='MADCEL')
    #plt.axis().xaxis.set_ticks(xs)
    plt.legend(('text', 'network'), 'upper left', shadow=True)
    plt.xlabel('#users')
    plt.ylabel('Acc@161')
    ax.set_xticklabels(xs_labels)
    #plt.title('The effect of training size on geolocation accuracy in text-based and nework-based models')
    #axis([0,2,-1,1])
    plt.show(block=True)
    plt.savefig('a.pdf', format='pdf')

def plot_numbers():
    plot_name = 'downsample_text'
    plot_file = GEOTEXT_HOME + '/' + plot_name + '.pdf'
    xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    xs_labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    ax = plt.gca()
    y1 = [25.0, 28.3, 30.7, 31.2, 33.1, 36.0, 37.6, 38.8, 38.9, 39.7]
    y2 = [41.8, 45.41, 46.35, 47.72, 48.14, 48.33, 48.77, 49.25, 49.5, 49.6]
    y3 = [46.48, 51.72, 54.09, 55.57, 56.5, 57.04, 57.65, 58.0, 58.2, 58.4]
    y_labels = ['GEOTEXT', 'Twitter-US', 'Twitter-WORLD']
    p1 = plt.plot(xs, y1, 'k--', label=y_labels[0])
    p2 = plt.plot(xs, y2, 'r-', label=y_labels[1])
    p3 = plt.plot(xs, y3, 'b-', label=y_labels[2])
    #plt.axis().xaxis.set_ticks(xs)
    plt.legend((y_labels[0], y_labels[1], y_labels[1]), 'upper left', shadow=True)
    plt.xlabel('Text Downsample Rate')
    plt.ylabel('Acc@161')
    ax.set_xticklabels(xs_labels)
    #plt.title('The effect of training size on geolocation accuracy in text-based and nework-based models')
    #axis([0,2,-1,1])
    #plt.show(block=True)
    print "saving the plot in " + plot_file
    plt.savefig(plot_file, format='pdf')

#analysis()
def cluster_train_points():
    from sklearn.cluster import DBSCAN
    
    points = []
    for user, location in trainUsers.iteritems():
        lat, lon = locationStr2Float(location)
        points.append([lat, lon])
    X = np.array(points)
    db = DBSCAN(eps=0.4, min_samples=50).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    ##############################################################################
    # Plot result
    import matplotlib.pyplot as plt
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=1)
    
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.savefig('clusters.pdf', format='pdf')
    #Tracer()()
#plot_numbers()

DATASET_NUMBER = 2
TEXT_ONLY = False
DATA_HOME = '/home/arahimi/datasets'
DATASETS = ['cmu', 'na', 'world']
ENCODINGS = ['latin1', 'utf-8', 'utf-8']
buckets = [300 , 2400, 2400]
reguls = [5e-5, 1e-6, 1e-7]
BUCKET_SIZE = buckets[DATASET_NUMBER - 1]
GEOTEXT_HOME = path.join(DATA_HOME, DATASETS[DATASET_NUMBER - 1])
data_encoding = ENCODINGS[DATASET_NUMBER - 1]
# GEOTEXT_HOME = '/home/arahimi/Roller Dataset NA'
users_home = path.join(GEOTEXT_HOME, 'processed_data')
testfile = path.join(users_home, 'user_info.test')
devfile = path.join(users_home, 'user_info.dev')
trainfile = path.join(users_home, 'user_info.train')

print "dataset: " + DATASETS[DATASET_NUMBER - 1]
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
trainClasses = {}
devClasses = {}
testClasses = {}
categories = []
mentions = []
testText = {}
devText = {}
trainText = {}
user_multi_labels = {}

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

mention_graph = None
methods = {'svd':TruncatedSVD, 'pca':PCA, 'factoranalysis':FactorAnalysis, 'median':None }
#methods  = ['svd', 'pca', 'factoranalysis', 'median']

for partitionMethod in ['svd', 'pca', 'factoranalysis']:
    #for downsample in [1.0]:
        #print(downsample)
    #downsize_train()
    initialize(partitionMethod=partitionMethod, granularity=BUCKET_SIZE, write=False, readText=False, downSampleTextCoefficient=1.0, reload_init=False)
    
    #cluster_train_points()
    partitionLocView(granularity=BUCKET_SIZE, partitionMethod=partitionMethod, convexhull=True)
    #asclassification(granularity=BUCKET_SIZE, partitionMethod=partitionMethod, use_mention_dictionary=False)
    #prepare_adsorption_data_collapsed(DEVELOPMENT=False, ADD_TEXT_LEARNER=True  , CELEBRITY_THRESHOLD=5, build_networkx_graph=False, DIRECT_GRAPH_WEIGHTED=True)
    #direct_graph2()
        #junto_postprocessing(multiple=False, dev=False, text_confidence=1.0)
    #Tracer()()

# direct_graph2()
# ideal_network_errors()

# save_matlab()    
# fabian_glasso()
# spams_group_lasso()
# prepare_adsorption_data()

#prepare_adsorption_data_collapsed()
#junto_postprocessing(multiple=False, dev=False)
#create_junto_config_files_for_tuning()

# iterative_collective_classification()
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
#asclassification(granularity=BUCKET_SIZE, partitionMethod='median', use_mention_dictionary=False)
# classificationBench(granularity=300, partitionMethod='median', use_mention_dictionary=False)

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
print str(datetime.now())
script_end_time = time.time()
script_execution_hour = (script_end_time - script_start_time) / 3600.0
print "the script execution hours is " + str(script_execution_hour)
