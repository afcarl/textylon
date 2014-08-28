'''
Created on Dec 2, 2013

@author: Afshin Rahimi
'''
import logging


logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AbstractVectorizer(object):
    '''
    This class using a tokenizer and a normalizer generates word space models.
    '''
    tokenizer = None
    normalizer = None

    def __init__(self, abstractTokenizer, abstractNormalizer):
        '''
        Constructor
        '''
        self.tokenizer = abstractTokenizer
        self.normalizer = abstractNormalizer
        
    def vectorize(self):
        '''
        This method generates a vector model using its data provider, tokenizer and normalizer.
        '''
    
    def prepareTokens(self, text):
        '''
        This method uses tokenizer and normalizer to
        prepare a list of normalized tokens from a given piece of text. 
        '''
        self.tokenizer.setText(text)
        logger.debug("tokenizing started...")
        self.tokenizer.tokenize()
        logger.debug("tokenizing finished.")
        tokens = self.tokenizer.getAllTokens()
        textLength = len(tokens)
        logger.debug("Token length: " + str(textLength))
        # normalize all tokens
        logger.debug("normalizing started...")
        for tokenIndex in range(0, textLength):
            unNormalToken = tokens[tokenIndex]
            tokens[tokenIndex] = self.normalizer.normalize(unNormalToken)
        logger.debug("normalizing finished.")
        return tokens, textLength 
