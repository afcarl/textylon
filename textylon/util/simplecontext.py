'''
Created on Dec 2, 2013

@author: Afshin Rahimi
'''

class SimpleContext(object):
    '''
    This is a class representing a context in word space vector model.
    '''
    words = ''

    def __init__(self, words):
        '''
        Constructor setting the set of words as context
        '''
        self.words = words
    def toString(self):
        result = ''
        for word in self.words:
            result = word + ' ' + result
        return result
    
    def setWords(self, words):
        self.words = words
    
