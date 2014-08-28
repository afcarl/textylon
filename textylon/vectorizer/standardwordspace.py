'''
Created on Dec 16, 2013

@author: af
'''
from collections import Counter
import logging
from scipy import isnan

from abstractvectorizer import AbstractVectorizer
from com.textylon.util.matrix import Matrix
from com.textylon.util.vocabulary import Vocabulary


logging.basicConfig(format='%(asctime)s %(module)s %(funcName)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StandartVectorizer(AbstractVectorizer):
    '''
    This class implements the standard word space model
    '''
    dictionary = None
    windowSize = None
    minTermFreq = None
    maxTermFreq = None
    minContextFreq = None
    maxContextFreq = None
    terms = None
    contexts = None
    dictionaryCompleted = False
    coocMatrix = None
    totalTokens = None
    
    def __init__(self, tokenizer, normalizer, windowSize, minTermFreq, maxTermFreq, minContextFreq, maxContextFreq):
        '''
        Constructor
        '''
        
        # set the normalizer and tokenizer in the parent class
        super(StandartVectorizer, self).__init__(tokenizer, normalizer)
        
        self.dictionary = Counter()
        self.minTermFreq = minTermFreq
        self.maxTermFreq = maxTermFreq
        self.minContextFreq = minContextFreq
        self.maxContextFreq = maxContextFreq
        self.terms = Vocabulary()
        self.contexts = Vocabulary()
        self.windowSize = windowSize
        
    
    # collect vocabulary and count frequencies
    def addToDictionary(self, text):
        logger.debug("Adding text tokens to dictionary...")
        tokens, textLength = self.prepareTokens(text)
        self.totalTokens = textLength
        for token in tokens:
            logger.debug(token)
            self.dictionary[token] += 1
        logger.debug("Added " + str(textLength) + " tokens to dictionary.")
        
        
    def buildVocabulary(self):
        '''
        terms and contexts vocabularies are build and the terms
        which are out of minimum and maximum frequency threshold
        are filtered out.
        '''
        logger.info("Filtering terms and contexts started...")
        if not self.dictionaryCompleted:
            logger.error("Before calling this method the dictionary should be build completely and self.dictionaryCompleted should be True.")
            raise 
        termId = 0
        contextId = 0
        for w, freq in self.dictionary.most_common():
            if(freq > self.minTermFreq and freq < self.maxTermFreq):
                self.terms.add(w, termId)
                termId += 1
            if(freq > self.minContextFreq and freq < self.maxContextFreq):
                self.contexts.add(w, contextId)
                contextId += 1
        self.coocMatrix = Matrix(termId, contextId)
        logger.info("Filtering terms and contexts finished. Number of terms = " + str(termId) + " and the number of contexts is " + str(contextId))
    
    
    def addToModel(self, text):
        '''
        add cooccurrences within windowSize in text to the coocMatrix.
        This method should be called when the dictionary is completely built.
        '''
        logger.info("Adding text to model started...")
        if not self.dictionaryCompleted:
            logger.error("Before calling this method the dictionary should be build completely and self.dictionaryCompleted should be True.")
            raise 
        tokens, textLength = self.prepareTokens(text)
        logger.info("indexing " + str(textLength) + " words.")
        # extract tokens and their context in order to vectorize them.
        for i in range(0, textLength):
            if i % 10000 == 0:
                logger.info("vectorizing " + str (100 * float(i) / float(textLength)) + "%")
            term = tokens[i] 
            # check if this term exists in terms
            termId = self.terms.getId(term)
            if termId != -1:           
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
                    contextId = self.contexts.getId(context)
                    if contextId != -1:
                        logger.debug("indexed " + term + " " + context)
                        self.coocMatrix.incrementValue(termId, contextId, 1)

    def getCosineSimilarity(self, word1, word2):
        termId1 = self.terms.getId(word1)
        termId2 = self.terms.getId(word2)
        if termId1 == -1:
            return word1 + " not found in the terms."
        if termId2 == -1:
            return word2 + " not found in the terms."
        self.coocMatrix.getRowCosineSimilarity(termId1, termId2)
    
    def getClosestNeighbor(self, word):
        return self.getNNearestNeighbors(word, n=1)
    
    def getNNearestNeighbors(self, word, n=None):
        wordId = self.terms.getId(word)
        if wordId == -1:
            return None
        result = {}
        wordVector = self.coocMatrix.getRow(wordId)
        for w, id in self.terms.words.iteritems():
            # if it is the word itself do nothing             
            if id == wordId:
                continue
            wVector = self.coocMatrix.getRow(id)
            sim = self.coocMatrix.getCosineSimilarityByRow(wordVector, wVector)
            # if sim is not a number or is zero continue             
            if not sim:
                continue
            result[w] = sim
        sorted_result = sorted(result.iteritems(), key=lambda(k, v): v, reverse=True)
        if n == None:
            n = len(sorted_result)
            
        return sorted_result[0: n]
        
         
    def buildModel(self, dataProvider):
        
        # build frequency dictionary from corpus
        while(dataProvider.hasNext()):
            data = dataProvider.getNext()
            text = data[1]
            self.addToDictionary(text)
        logger.debug(self.dictionary)
        # here reset means that we should read the entire corpus again
        dataProvider.reset()
        self.dictionaryCompleted = True
        
        # build vocabulary
        self.buildVocabulary()
        logger.debug(self.terms.words)
        logger.debug(self.contexts.words)
        # add each text file in the corpus to model
        while(dataProvider.hasNext()):
            data = dataProvider.getNext()
            text = data[1]
            self.addToModel(text)
    
        
        
        
                               
