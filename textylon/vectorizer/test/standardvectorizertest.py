'''
Created on Dec 18, 2013

@author: af
'''
import logging
import os
import shutil
import unittest
import urllib2

from textylon.dataprovider.dataproviders import SimpleCorpusDataProvider
from textylon.normalizer.normalizers import LowerCaseNormalizer
from textylon.tokenizer.tokenizers import SimpleTokenizer
from textylon.vectorizer.standardwordspace import StandartVectorizer


logging.basicConfig(format='%(asctime)s %(module)s %(funcName)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DOWMLOAD_ONLINE = False
DOWNLOAD_SIZE = 0
NUM_FILES = 1

class Test(unittest.TestCase):


    corpusPath = '/tmp/testcorpus'
    sampleWord = 'content'

    def setUp(self):
        if os.path.exists(self.corpusPath):
            shutil.rmtree(self.corpusPath)
        # create a directory in temp with some content
        
        
        os.mkdir(self.corpusPath)
        if DOWMLOAD_ONLINE:
            logger.info("start downloading " + str(DOWNLOAD_SIZE) + " bytes of text...")
            response = urllib2.urlopen('http://dingo.sbs.arizona.edu/~hammond/ling696f-sp03/browncorpus.txt',)
            if DOWNLOAD_SIZE == 0:
                html = response.read()
            else: 
                html = response.read(DOWNLOAD_SIZE)
            
            logger.info("download finished.")
            f = open(os.path.join(self.corpusPath, 'txtcorpus.txt'), 'w')
            f.write(html)
            f.close()
        else:
            for i in range(0, NUM_FILES):
                f = open(os.path.join(self.corpusPath, 'file' + str(i)), 'w')
                f.write('afshin nina central nina afshin afshin amin central amin afshin')
                f.close()


    def tearDown(self):
        if os.path.exists(self.corpusPath):
            shutil.rmtree(self.corpusPath)

    def testVectorizer(self):
        stopWords = ['this', 'the',
                     'is', 'for', 'am', 'are',
                     'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
                     'a', 'an', 'it', 'not', 'on', 'with', 'he', 'she',
                     'as', 'do', 'at', 'but', 'his', 'her', 'by', 'from',
                     'they', 'you', 'into', 'all']

        delimiters = " "
        stopWords = []
        tokenizer = SimpleTokenizer(stopWords) 
        provider = SimpleCorpusDataProvider(self.corpusPath)
        normalizer = LowerCaseNormalizer()
        vectorizer = StandartVectorizer(tokenizer, normalizer, 1, 0, 2000, 0, 2000)
        vectorizer.buildModel(provider)
        # logger.info(vectorizer.terms.words)
        # vectorizer.coocMatrix.tfidfTransform()

      



       
        logger.info("finding similar words...")
        # for word, id in vectorizer.terms.words.iteritems():
        words = ['afshin', 'nina', 'central', 'amin']
        for word in words: 
            similar = vectorizer.getNNearestNeighbors(word, 10)
            logger.info(word + " is similar to:")
            logger.info(similar)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
