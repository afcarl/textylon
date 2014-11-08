'''
Created on Dec 16, 2013

@author: af
'''

class Vocabulary(object):
    '''
    This is a vocabulary class which holds words and their id in a dictionary
    '''
    words = None

    def __init__(self):
        '''
        Constructor
        '''
        self.words = {}
    def add(self, word, wordId):
        self.words[word] = wordId
    
    def exists(self, word):
        if word in self.words:
            return True
        else:
            return False
    
    def addIfNotExist(self, word, wordId):
        if self.exists(word):
            return -1
        else:
            self.add(word, wordId)
    
    def getVocabSize(self):
        return len(self.words)
    def getId(self, word):
        try:
            return self.words[word]
        except KeyError:
            return -1    
