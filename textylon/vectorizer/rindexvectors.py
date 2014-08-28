'''
Created on Dec 2, 2013

@author: Afshin Rahimi
'''
import Queue
from collections import defaultdict
import copy
import logging
from math import sqrt
from numpy.core.numeric import dtype, int8, int16
from numpy.random import random_integers
import pickle
from scipy.sparse import csr_matrix, csc_matrix
import threading
from time import time

from abstractvectorizer import  AbstractVectorizer
import scipy as sp


logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
hdlr = logging.FileHandler('log.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)




class RandomIndexVectorizer(AbstractVectorizer):
    '''
    This class given a data provider, tokenizer, normalizer, the size of vectors, the size of context window and vector type
    generates word vectors. This class supports order, directional and context vectors depending on the  
    For further information about Random Indexing refer to:
    https://en.wikipedia.org/wiki/Random_indexing
    '''
    dimension = None
    windowSize = None
    nonZero = None
    contexts = {}
    
    RIGHT_DIRECTION = +1
    LEFT_DIRECTION = -1
    vectorType = None
    DIRECTIONAL_VECTOR_TYPE = 'directional'
    ORDER_VECTOR_TYPE = 'order'
    CONTEXT_VECTOR_TYPE = 'context'
    DOCUMENT_VECTOR_TYPE = 'document'
    vectorTypes = [DIRECTIONAL_VECTOR_TYPE, ORDER_VECTOR_TYPE, CONTEXT_VECTOR_TYPE, DOCUMENT_VECTOR_TYPE]
    docNum = -1
    def __init__(self, tokenizer, normalizer, dimension=300, nonZero=10, windowSize=2, vectorType='directional'):
        '''
        Constructor
        '''
        super(RandomIndexVectorizer, self).__init__(tokenizer, normalizer)
        self.windowSize = windowSize
        self.dimension = dimension
        self.nonZero = nonZero
        if vectorType not in self.vectorTypes:
            logger.error("vector type should be either directional, order or context.")
            raise()
        self.vectorType = vectorType
        self.finished = False
        self.words = {}
            
    def initContextVector(self, sparse=True):
        if sparse:
            return csr_matrix((1, self.dimension))
        else:
            return sp.zeros(self.dimension, dtype=int)
        
    def getNonZeroItems(self):
        '''
        This function creates nonZero +1 and -1 integers and nonZero indices between 0 and dimensions - 1.
        This is the random index for each context type.
        for example for dimensions = 300 and nonZero = 10 it builds the following lists:
        [34, 65, 4, 79, 120, 72, 270, 160, 55, 164]
        [-1, -1, +1, -1, +1, +1, +1, -1, -1, +1]
        The sparse vector has 300 columns of which 10 are nonZero with uniform distribution of nonZero items with values +1 and -1.
        '''
        nonZeroItems = []
        nonZeroData = []
        nonZeroIndices = []
        
        nonZeroIndices = random_integers(0, self.dimension - 1, self.nonZero)
        values = random_integers(0, 1, self.nonZero)
        
        for value in values:
            if value == 1:
                nonZeroData.append(1)
            else:
                nonZeroData.append(-1)
        del values
        nonZeroItems = [nonZeroIndices, nonZeroData]
        return nonZeroItems
             

    def vectorize(self, text):
        logger.debug("indexing started...")
        tokens, textLength = self.prepareTokens(text)
        self.docNum += 1
        
        # extract tokens and their context in order to vectorize them.
        for i in range(0, textLength):
            if i % 10000 == 0:
                logger.debug("vectorizing " + str (100 * float(i) / float(textLength)) + "%")
            word = tokens[i]
            if self.vectorType == self.DOCUMENT_VECTOR_TYPE:
                rotationCount = 0
                context = self.docNum
                self.indexWord(word, context, rotationCount)
                continue
            leftIndex = i - self.windowSize
            if leftIndex < 0:
                leftIndex = 0
            
            rightIndex = i + self.windowSize
            if rightIndex > (textLength - 1):
                rightIndex = textLength - 1
            
            for j in range(leftIndex, rightIndex + 1):
                # the distance between the word and its context word.
                distance = j - i
                # this is the word itself
                if distance == 0:
                    continue
                context = tokens[j]
                # if the type of vector is context no permutation should occur so rotation count is zero
                rotationCount = 0
                if self.vectorType == self.CONTEXT_VECTOR_TYPE:
                    # let rotationCount remain zero
                    pass
                # if the vector type is directional rotation count should be either +1 or -1 according to direction
                elif self.vectorType == self.DIRECTIONAL_VECTOR_TYPE:
                    if distance > 0:
                        rotationCount = +1
                    else:
                        rotationCount = -1
                # if the vector type is order the number of rotations equals the difference between the word and the context word
                elif self.vectorType == self.ORDER_VECTOR_TYPE:
                    rotationCount = distance
                
                self.indexWord(word, context, rotationCount)
                       
        logger.debug("indexing finished.")
    
    
        
    def createContextRandomIndex(self, context, checkForExistingContext=True, sparse=True):
        '''
        create a random index for a context. If checkForExistingContext is True it will check for existing contexts
        but this reduces the performance. The better idea is that this method be called by checkForExistingContext = False and 
        the check for existing context be done before the method is called so you should never call this method with
        checkForExistingContext = False without a former check for existing contexts.
        '''
        if checkForExistingContext:
            if context in self.contexts:
                return
        nonZeroItems = self.getNonZeroItems()
        nonZeroData = nonZeroItems.pop()
        nonZeroColumns = nonZeroItems.pop()
        if sparse:
            nonZeroRows = [0] * self.nonZero
            contextMatrix = csr_matrix((nonZeroData, (nonZeroRows, nonZeroColumns)), shape=(1, self.dimension), dtype=int8)
            self.contexts [context] = contextMatrix
            return contextMatrix
        else:
            arr = sp.zeros(self.dimension, dtype=int)
            for i in range(self.nonZero):
                arr[nonZeroColumns[i]] = nonZeroData[i]
            self.contexts[context] = arr
            return arr
        
        
    
    def indexWord(self, word, context, rotationCount):
        '''
        This method given a word and its context and direction retrieves the corresponsing vectors if they exist. If either the word's vector or its context vector
        does not exist the vectors will be initialized by a zero sparse matrix.
        direction is either +1 or -1 and is used in permutations so that the produced vector can be used as a directional vector using permutations.
        '''
        
        if context not in self.contexts:
            logger.debug(str(context) + " not found in contexts.")
            contextVector = self.createContextRandomIndex(context, False, sparse=False)
        else:
            contextVector = self.contexts[context]
        
        # perform the permutations according to rotationCount
        if rotationCount != 0:
            indices = contextVector.indices
            contextVector.indices = (indices + rotationCount) % contextVector.shape[1]

        # add the word's context vector to word's vector
        self.words[word] = self.words.get(word, self.initContextVector(sparse=False)) + contextVector

        
    def cosineSimilarity(self, first, second):
        '''
        This function given too vectors (2-dimensional matrices with just one row) returns
        the cosine similarity of them. The matrices should both have the same (1, n) shape
        or else a dimension mismatch exception will be raised.
        '''
        dotProduct = first.multiply(second).sum()
        sizefirst = sqrt(first.multiply(first).sum())
        sizesecond = sqrt(second.multiply(second).sum())
        return dotProduct / (sizefirst * sizesecond)
    
    
    def getClosestNeighbor(self, word):
        '''
        This function given a word searches all the word space to find the closest
        word to this word in word space. If the word is not in the word space 
        en error will be raised
        '''
        if word not in self.words:
            logger.error("word not in lexicon.")
            raise()
        wordVector = self.words[word]
        similarityLength = 0
        closestNeighbor = None
        for neighbor, neighborVector in self.words.iteritems():
            if neighbor == word:
                continue
            similarity = abs(self.cosineSimilarity(wordVector, neighborVector))
            if similarity > similarityLength:
                similarityLength = similarity
                closestNeighbor = neighbor
        return closestNeighbor, similarityLength
    
    def buildModel(self, dataProvider):
        while(dataProvider.hasNext()):
            data = dataProvider.getNext()
            text = data[1]
            dataProvider.vectorize(text)
    
    def save(self, fileName='random-index.pkl'):
        with open(fileName, 'wb') as file:
            pickle.dump(self.words, file)
    
    def load(self, fileName='random-index.pkl'):
        with open(fileName, 'rb') as file:
            self.words = pickle.load(file)
            return self.words


        
class DocRandomIndexVectorizer():
    
    def __init__(self, queue, dimension=2048, nonZero=7):
        self.q = queue
        self.docNum = -1
        self.dimension = dimension
        self.nonZero = nonZero
        self.contexts = {}
        # start vectorizer threads

    def createContextRandomIndex(self, context, sparse=True):
        '''
        create a random index for a context. If checkForExistingContext is True it will check for existing contexts
        but this reduces the performance. The better idea is that this method be called by checkForExistingContext = False and 
        the check for existing context be done before the method is called so you should never call this method with
        checkForExistingContext = False without a former check for existing contexts.
        '''
        nonZeroItems = self.getNonZeroItems()
        nonZeroData = nonZeroItems.pop()
        nonZeroColumns = nonZeroItems.pop()
        if sparse:
            nonZeroRows = [0] * self.nonZero
            contextMatrix = csr_matrix((nonZeroData, (nonZeroRows, nonZeroColumns)), shape=(1, self.dimension), dtype=int8)
            self.contexts [context] = contextMatrix
            return contextMatrix
        else:
            arr = sp.zeros(self.dimension, dtype=int)
            for i in range(self.nonZero):
                arr[nonZeroColumns[i]] = nonZeroData[i]
            self.contexts[context] = arr
            return arr
        
    def getNonZeroItems(self):
        '''
        This function creates nonZero +1 and -1 integers and nonZero indices between 0 and dimensions - 1.
        This is the random index for each context type.
        for example for dimensions = 300 and nonZero = 10 it builds the following lists:
        [34, 65, 4, 79, 120, 72, 270, 160, 55, 164]
        [-1, -1, +1, -1, +1, +1, +1, -1, -1, +1]
        The sparse vector has 300 columns of which 10 are nonZero with uniform distribution of nonZero items with values +1 and -1.
        '''
        nonZeroItems = []
        nonZeroData = []
        nonZeroIndices = []
        
        nonZeroIndices = random_integers(0, self.dimension - 1, self.nonZero)
        values = random_integers(0, 1, self.nonZero)
        
        for value in values:
            if value == 1:
                nonZeroData.append(1)
            else:
                nonZeroData.append(-1)
        del values
        nonZeroItems = [nonZeroIndices, nonZeroData]
        return nonZeroItems    
    def vectorize(self, text):
        '''
        add the text to docQueue
        '''
        self.docNum += 1
        if self.docNum % 100000 == 0:
            print 'finished %d' % self.docNum
        contextVector = self.createContextRandomIndex(self.docNum, sparse=True)
        self.q.put((contextVector, text))

        
class DocVectorizerThread(threading.Thread):
    '''
    This is a vectorizer thread. It picks up a text from a queue and 
    adds its words/contexts to the random indexing model.
    '''
    def __init__(self, name, queue, words, lock, normalizer, tokenizer, dimension):
        """
        creates a vectorizer thread
        """
        super(DocVectorizerThread, self).__init__()
        self.name = name
        self.finished = False
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.dimension = dimension
        self.q = queue
        self.words = words
        self.lock = lock
    def initContextVector(self, sparse=True):
        if sparse:
            return csr_matrix((1, self.dimension), dtype=int8)
        else:
            return sp.zeros(self.dimension, dtype=int)
        
    def prepareTokens(self, text):
        '''
        This method uses tokenizer and normalizer to
        prepare a list of normalized tokens from a given piece of text. 
        '''
        self.tokenizer.setText(text)
        self.tokenizer.tokenize()
        tokens = self.tokenizer.getAllTokens()
        textLength = len(tokens)
        # normalize all tokens
        for tokenIndex in range(0, textLength):
            unNormalToken = tokens[tokenIndex]
            tokens[tokenIndex] = self.normalizer.normalize(unNormalToken)
        return tokens, textLength 
    
    def run(self):
        """
        get a text from queue process it and add the results to random index model
        """

        while True:

            # get a document/text from queue
            contextVector, text = self.q.get()

            # do the work
            logger.debug(self.name + " indexing started...")
            tokens, textLength = self.prepareTokens(text)

            # extract tokens and their context in order to vectorize them.
            for token in tokens:
                self.lock.acquire()
                self.words[token] = self.words.get(token, self.initContextVector(sparse=True)) + contextVector
                self.lock.release()
                
            logger.debug(self.name + " indexing finished.")
                 
                
            self.q.task_done()
        logger.info('vectorizing finished for thread ' + self.name)
        self.outofwork = True
      
class IndependentDocVectorizerThread(threading.Thread):
    '''
    This is a vectorizer thread. It picks up a text from a queue and 
    adds its words/contexts to the random indexing model.
    '''
    def __init__(self, name, vectorizer):
        """
        creates a vectorizer thread
        """
        super(IndependentDocVectorizerThread, self).__init__()
        self.name = name
        self.vectorizer = vectorizer
        self.finished = False
        self.normalizer = copy.deepcopy(self.vectorizer.normalizer)
        self.tokenizer = copy.deepcopy(self.vectorizer.tokenizer)
        self.words = {}
    def initContextVector(self, sparse=True):
        if sparse:
            return csr_matrix((1, self.vectorizer.dimension), dtype=int8)
        else:
            return sp.zeros(self.vectorizer.dimension, dtype=int)
        
    def prepareTokens(self, text):
        '''
        This method uses tokenizer and normalizer to
        prepare a list of normalized tokens from a given piece of text. 
        '''
        self.tokenizer.setText(text)
        self.tokenizer.tokenize()
        tokens = self.tokenizer.getAllTokens()
        textLength = len(tokens)
        # normalize all tokens
        for tokenIndex in range(0, textLength):
            unNormalToken = tokens[tokenIndex]
            tokens[tokenIndex] = self.normalizer.normalize(unNormalToken)
        return tokens, textLength 
    
    def run(self):
        """
        get a text from queue process it and add the results to random index model
        """

        while not self.vectorizer.finished or not self.vectorizer.docQueue.empty():

            # get a document/text from queue
            contextVector, text = self.vectorizer.docQueue.get()

            # do the work
            logger.debug(self.name + " indexing started...")
            tokens, textLength = self.prepareTokens(text)

            # extract tokens and their context in order to vectorize them.
            for token in tokens:
                self.words[token] = self.words.get(token, self.initContextVector(sparse=False)) + contextVector
                
            logger.debug(self.name + " indexing finished.")
                 
                
            self.vectorizer.docQueue.task_done()
        logger.info('vectorizing finished for thread ' + self.name)
        
        print "start pickling..."
        with open(self.name + '.pkl', 'wb') as pfile:
            pickle.dump(self.words, pfile)
        print "pickling finished."
        self.finished = True
      

      
class FastRandomIndexVectorizer(RandomIndexVectorizer):
    
    def __init__(self, tokenizer, normalizer, dimension=300, nonZero=7, windowSize=2, vectorType='directional', numThreads=10, queueMaxSize=20000):
        super(FastRandomIndexVectorizer, self).__init__(tokenizer, normalizer, dimension, nonZero, windowSize, vectorType)
        self.queueMaxSize = queueMaxSize
        self.docQueue = Queue.Queue(self.queueMaxSize)
        self.numThreads = numThreads
        self.threads = []
        self.docNumLock = threading.Lock()
        self.modelLock = threading.Lock()
        # start vectorizer threads
        for i in range(0, self.numThreads):
            t = VectorizerThread(str(i), self)
            t.setDaemon(True)
            self.threads.append(t)
            t.start()
    def vectorize(self, text):
        '''
        add the text to docQueue
        '''
        self.docQueue.put((self.docNum, text))     
class VectorizerThread(threading.Thread):
    '''
    This is a vectorizer thread. It picks up a text from a queue and 
    adds its words/contexts to the random indexing model.
    '''
    def __init__(self, name, vectorizer):
        """
        creates a vectorizer thread
        """
        super(VectorizerThread, self).__init__()
        self.name = name
        self.vectorizer = vectorizer

    

    def run(self):
        """
        get a text from queue process it and add the results to random index model
        """
        while not self.vectorizer.finished:
            # get a document/text from queue
            text = self.vectorizer.docQueue.get()
            # do the work
            logger.debug("indexing started...")
            tokens, textLength = self.vectorizer.prepareTokens(text)
            self.vectorizer.docNumLock.acquire()
            self.vectorizer.docNum += 1
            localDocNum = self.vectorizer.docNum
            self.vectorizer.docNumLock.release()
        
            # extract tokens and their context in order to vectorize them.
            for i in range(0, textLength):
                word = tokens[i]
                if self.vectorizer.vectorType == self.vectorizer.DOCUMENT_VECTOR_TYPE:
                    rotationCount = 0
                    context = localDocNum
                    self.vectorizer.modelLock.acquire()
                    self.vectorizer.indexWord(word, context, rotationCount)
                    self.vectorizer.modelLock.release()
                    continue
                leftIndex = i - self.vectorizer.windowSize
                if leftIndex < 0:
                    leftIndex = 0
        
                rightIndex = i + self.vectorizer.windowSize
                if rightIndex > (textLength - 1):
                    rightIndex = textLength - 1
        
                for j in range(leftIndex, rightIndex + 1):
                    # the distance between the word and its context word.
                    distance = j - i
                    # this is the word itself
                    if distance == 0:
                        continue
                    context = tokens[j]
                    # if the type of vector is context no permutation should occur so rotation count is zero
                    rotationCount = 0
                    if self.vectorizer.vectorType == self.vectorizer.CONTEXT_VECTOR_TYPE:
                        # let rotationCount remain zero
                        pass
                    # if the vector type is directional rotation count should be either +1 or -1 according to direction
                    elif self.vectorizer.vectorType == self.vectorizer.DIRECTIONAL_VECTOR_TYPE:
                        if distance > 0:
                            rotationCount = +1
                        else:
                            rotationCount = -1
                    # if the vector type is order the number of rotations equals the difference between the word and the context word
                    elif self.vectorizer.vectorType == self.vectorizer.ORDER_VECTOR_TYPE:
                        rotationCount = distance
                    self.vectorizer.modelLock.acquire()
                    self.vectorizer.indexWord(word, context, rotationCount)
                    self.vectorizer.modelLock.release()
                       
            logger.debug("indexing finished.")
    
            self.vectorizer.docQueue.task_done()
      
