'''
Created on Mar 14, 2014

@author: af
'''
import codecs
from os import listdir
from os.path import isfile, join


class AbstractDataProvider(object):
    '''
    This is the abstract base class for providing data to other components of the system.
    This class is iterable.
    '''


    def __init__(self):
        '''
        Constructor for AbstractDataProvider, as this class is abstract it does nothing
        '''
    def hasNext(self):
        '''
        returns true if the iterator has another item
        '''
    def getNext(self):
        '''
        returns next data item
        '''




class SimpleCorpusDataProvider(AbstractDataProvider):
    '''
    This class is an implementation of AbstractDataProvider and provides data given
    the address of a corpus directory. It simply reads text files in that directory
    and returns the content to the caller.
    '''
    corpusDir = None
    fileNames = []

    def readFileNames(self):
        '''
        using self.corpusDir a corpus directory address, assumes all the files in the directory 
        assert text files and extracts their names. 
        This method does NOT support subdirectories.
        '''
        self.fileNames = [ f for f in listdir(self.corpusDir) if isfile(join(self.corpusDir, f)) ]
        
    def __init__(self, corpusDir):
        '''
        Constructor for SimpleCorpusDataProvider
        '''
        super(SimpleCorpusDataProvider, self).__init__()
        self.corpusDir = corpusDir
        self.readFileNames()
    
    def hasNext(self):
        '''
        returns True if there is any other data to provide
        '''
        if len(self.fileNames) > 0 :
            return True
        else:
            return False
    def getNext(self):
        '''
        returns the next data item 2-tuple [file address, content] if it exists.
        if there is no other data item it raises an @IndexError.
        caller should check hasNext() before calling this.
        '''
        fileName = self.fileNames.pop()
        txtFile = open(join(self.corpusDir, fileName), 'r')
        text = txtFile.read()
        txtFile.close()
        return join(self.corpusDir, fileName), text
    
    def reset(self, corpusDir=None):
        if(corpusDir != None):
            self.corpusDir = corpusDir
        
        self.fileNames = []
        self.readFileNames()
        
class OneFileCorpusDataProvider(AbstractDataProvider):
    '''
    This class is an implementation of AbstractDataProvider and provides data given
    the address of a text file. It simply reads the text file assuming each document in one line
    and returns the content to the caller.
    '''


        
    def __init__(self, corpusFileAddress, encoding='utf-8'):
        '''
        Constructor for SimpleCorpusDataProvider
        '''
        super(OneFileCorpusDataProvider, self).__init__()
        self.encoding = encoding
        self.corpusFileAddress = corpusFileAddress
        self.eof = False
        self.file = codecs.open(self.corpusFileAddress, 'r', self.encoding)
    
    def hasNext(self):
        '''
        returns True if there is any other data to provide
        '''
        
        return not self.eof
    
    def getNext(self):
        '''
        returns the next line of corpus file or None if EOF
        '''
        content = self.file.readline()
        if content is None or content == '':
            self.eof = True
            self.file.close()
            return None
        else:
            return content
    
    def reset(self, corpusDir=None):
        if not self.file.closed:
            self.file.close()
        self.eof = False
        self.file = open(self.corpusFileAddress, 'r')
              
    
