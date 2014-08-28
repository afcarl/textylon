'''
Created on Feb 6, 2014
This class vectorizes the Movie Review Dataset and stores the model and the pos/neg labels in a pickle so that it can be used directly in Theano
We should partition the movie review dataset in 3 train, validation and test subdirectory within each there is a 0 directory for pos samples and a 1 directory
for neg samples. if you happen to want to change this for multiple classes you should use numbers for lables so that it becomes theano-compatible.
so that this class can do its job properly so the directory structure should look like this:

       Movie Review
           train
               0
               1
           validation
               0
               1
           test
               0
               1
As there are about 1000 positive and 1000 negative reviews i decided to partition in into 1400 reviews for train, 300 for validation and 300 for test
where each includes an equal share of pos and neg reviews (e.g. 300 tests consists of 150 pos and 150 neg) 
@author: af
'''
import os, cPickle
from  sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from time import time

import numpy as np


class MovieReviewBagOfWordsVectorizer(object):
    '''
    classdocs
    '''
    


    def __init__(self, datasetAddress, TFIDF=False):
        '''
        Constructor
        '''
        self.DATA_FOLDER = datasetAddress
        self.TRAIN_FOLDER = os.path.join(self.DATA_FOLDER, "train")
        self.VALIDATION_FOLDER = os.path.join(self.DATA_FOLDER, "validation")
        self.TEST_FOLDER = os.path.join(self.DATA_FOLDER, "test")
        self.TFIDF = TFIDF
        # the datatype should be np.float32 so that it is compatible with theano
        if(self.TFIDF):
            self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', min_df=1, dtype=np.float32)
        else:
            self.vectorizer = CountVectorizer(min_df=1, dtype=np.float32)
        
        self.data_train = load_files(self.TRAIN_FOLDER, encoding='latin1')
        self.data_validation = load_files(self.VALIDATION_FOLDER, encoding='latin1')
        self.data_test = load_files(self.TEST_FOLDER, encoding='latin1')
        self.categories = self.data_train.target_names
        self.data_train_size_mb = self.size_mb(self.data_train.data)
        self.data_validation_size_mb = self.size_mb(self.data_validation.data)
        self.data_test_size_mb = self.size_mb(self.data_test.data)
        
        print("%d documents - %0.3fMB (training set)" % (
            len(self.data_train.data), self.data_train_size_mb))
        print("%d documents - %0.3fMB (test set)" % (
            len(self.data_test.data), self.data_test_size_mb))
        print("%d categories" % len(self.categories))
        print()
        
        # split a training set and a test set
        self.y_train = self.data_train.target.astype(np.int64)
        print("Extracted %d train labels" % len(self.y_train))
        self.y_validation = self.data_validation.target.astype(np.int64)
        print("Extracted %d validation labels" % len(self.y_validation))
        self.y_test = self.data_test.target.astype(np.int64)
        print("Extracted %d test labels" % len(self.y_test))
        
        
    def vectorize(self):
        print("Extracting features from the training dataset using a sparse vectorizer")
        t0 = time()
        # the output of the fit_transform (x_train) is a sparse csc matrix.
        self.X_train = self.vectorizer.fit_transform(self.data_train.data)
        duration = time() - t0
        print("done in %fs at %0.3fMB/s" % (duration, self.data_train_size_mb / duration))
        print("n_samples: %d, n_features: %d" % self.X_train.shape)
        print()
        
        print("Extracting features from the validation dataset using the same vectorizer")
        t0 = time()
        self.X_validation = self.vectorizer.transform(self.data_validation.data)
        duration = time() - t0
        print("done in %fs at %0.3fMB/s" % (duration, self.data_validation_size_mb / duration))
        print("n_samples: %d, n_features: %d" % self.X_validation.shape)
        print()
        
        print("Extracting features from the test dataset using the same vectorizer")
        t0 = time()
        self.X_test = self.vectorizer.transform(self.data_test.data)
        duration = time() - t0
        print("done in %fs at %0.3fMB/s" % (duration, self.data_test_size_mb / duration))
        print("n_samples: %d, n_features: %d" % self.X_test.shape)
        print()
    def saveDataset(self, outputFile='movie_review.pkl'):
        print("dumping the dataset to " + outputFile)
        outputDataset = open(outputFile, 'wb')
        theanoFormatDataset = ((self.X_train.toarray(), self.y_train), (self.X_validation.toarray(), self.y_validation), (self.X_test.toarray(), self.y_test))
        cPickle.dump(theanoFormatDataset, outputDataset)
        outputDataset.close()
        print("dumping successfully finished.")
    
    def size_mb(self, docs):
        return sum(len(s.encode('utf-8')) for s in docs) / 1e6

if __name__ == "__main__":
    datasetAddress = "/home/af/Downloads/moviereview-theano/txt_sentoken"
    vectorizer = MovieReviewBagOfWordsVectorizer(datasetAddress, TFIDF=True)
    vectorizer.vectorize()
    vectorizer.saveDataset()
    
        
