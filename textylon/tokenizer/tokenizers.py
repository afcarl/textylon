'''
Created on Mar 14, 2014

@author: af
'''
import re


class AbstractTokenizer(object):
    '''
    this is an abstract class for tokenizers
    '''
    tokens = []
    __text = None


    def setText(self, text):
        '''
        This function sets the text of this tokenizer and clears the token list. This function should be called
        whenever we need to use a constructor for a new piece of text.
        '''
        self.__text = text
        self.tokens = []
    def getText(self):
        return self.__text
    
    def __init__(self):
        '''
        this class is not meant to be instantiated
        '''
    def hasToken(self):
        '''
        Returns True if there is another token
        '''
    def nextToken(self):
        '''
        Returns the next token and removes it from tokens list if available else
        raises error.
        tokenize method should be called so that tokens are extracted at the first place.
        '''
    def tokenize(self):
        '''
        Tokenizes a piece of given text
        '''
    def getAllTokens(self):
        '''
        Returns a list of all tokens.
        The tokenize method should be called so that tokens are extracted at the first place.
        '''





class SimpleTokenizer(AbstractTokenizer):
    '''
    This class given a text and a list of delimiters tokenizes
    the text, removes stop words (optional) and returns the list of tokens.
    '''
    delimiters = None
    tokens = []
    stopWords = []
    vocabulary = None

    def __init__(self, stopWords, vocabulary=None):
        '''
        SimpleTokenizer's Constructor
        '''
        super(SimpleTokenizer, self).__init__()
        self.stopWords = stopWords
        self.vocabulary = vocabulary
        
    def hasToken(self):
        '''
        Returns True if there is any token available
        '''
        if len(self.tokens) > 0:
            return True
        else:
            return False
        
    def nextToken(self):
        '''
        Returns the next token if any.
        if there is no token it raises an error.
        tokenize method should be called once so that tokens are extracted from text.
        '''
        return self.tokens.pop()
    def tokenize(self):
        '''
        Removes stop words in self.text using self.stopWords list, tokenizes self.text using delimiters 
        in self.delimiters and populates self.tokens.
        This method should be called once. If there is any token in tokens it does nothing.
        if vocabulary is not None tokenizer will ignore all out of vocabulary words.
        '''
        if len(self.tokens) > 0:
            return
        elif self.getText() is not None and self.getText != '':
            tokens = self.getText().split()
            for token in tokens:
                if token in self.stopWords:
                    pass
                elif self.vocabulary is not None and token not in self.vocabulary:
                    pass
                else:
                    self.tokens.append(token)
            del tokens

            
    def getAllTokens(self):
        return self.tokens
    
if __name__ == '__main__':
    stops = ['to', 'and']
    vocab = ['i', 'going', 'to', 'find', 'way']
    tokenizer = SimpleTokenizer(stops, vocab)
    tokenizer.setText('i am going to find a way and do the work.')
    tokenizer.tokenize()
    tokens = tokenizer.getAllTokens()
    print tokens
        
