'''
Created on Feb 6, 2014
This module is intended to be my first experiences with Python Pickles so that i can save my data into Pickles, reload them and rebuild them from Pickles

Each set is a tuple of a MxN numpy float array (M being the number of 
examples, N the input dimension) and a one-dimensional numpy integer 
array containing the class for each example. The total data file is a 
pickled tuple containing the 3 sets. 

You might want to inspect the data structures interactively. Just start 
python and do: 

        >>> import gzip, cPickle 
        >>> sets = cPickle.load(gzip.open("mnist.pkl.gz")) 
        >>> print sets 
        >>> train, valid, test = sets 
        >>> print train 
        >>> train_x, train_y = train 
        >>> print train_x 
        >>> print type(train_x) 
@author: af
'''

import cPickle
import gzip
import pickle

import numpy as np


def simple_pickle_test():
    l1 = [1, 2, 3, 4]
    l2 = [5, 6, 7, 8]
    outputFile = open('test1.pkl', 'wb')
    pickle.dump(l1, outputFile)
    pickle.dump(l2, outputFile)
    outputFile.close()
    inputFile = open('test1.pkl', 'rb')
    ll1 = pickle.load(inputFile)
    ll2 = pickle.load(inputFile)
    inputFile.close()
    print ll1, ll2

def loadMnistFromTheano(minstAddress):
    data = gzip.open(minstAddress, 'rb')
    dataset = cPickle.load(data)
    data.close()
    train, valid, test = dataset
    train_x, train_y = train
    test_x, test_y = test
    print type(dataset)
    print type(train)
    print type(train_x)
    print type(train_y)
    print type(train_y[0])
    x = np.array([1, 2, 3], dtype=np.float32)
    print type(x)
    print len(train_y)
    print train_y
    print test_y


    
    
if __name__ == '__main__':
    # simple_pickle_test()
    loadMnistFromTheano('/home/af/DeepLearningTutorials/data/mnist.pkl.gz')
    loadMnistFromTheano('/home/af/movie_review.pkl.gz')
