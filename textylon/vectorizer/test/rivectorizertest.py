'''
Created on Dec 10, 2013

@author: af
'''

import logging
import os
import shutil
import unittest
import urllib2

from pycallgraph import Config
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from textylon.dataprovider.dataproviders import SimpleCorpusDataProvider
from textylon.normalizer.normalizers import LowerCaseNormalizer
from textylon.tokenizer.tokenizers import SimpleTokenizer
from textylon.vectorizer.rindexvectors import RandomIndexVectorizer, MultiThreadedRandomIndexVectorizer


config = Config(max_depth=5)
graphviz = GraphvizOutput(output_file='/tmp/stavicta.vectorizer.randomindexvectorizer.png')


logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DOWMLOAD_ONLINE = True
DOWNLOAD_SIZE = 200000
NUM_FILES = 1

class RandomIndexVectorizerTest(unittest.TestCase):
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
                f.write('my sample sentence')
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
                     'they', 'you', 'into']

        delimiters = " "
        tokenizer = SimpleTokenizer(stopWords) 
        provider = SimpleCorpusDataProvider(self.corpusPath)
        normalizer = LowerCaseNormalizer()
        vectorizer = RandomIndexVectorizer(tokenizer, normalizer, 300, 8, 4, 'context')
        # vectorizer = MultiThreadedRandomIndexVectorizer(tokenizer, normalizer, 300, 8, 4, 'context',10, 20000)
        while(provider.hasNext()):
            data = provider.getNext()
            text = data[1]
            vectorizer.vectorize(text)



       
        
        for word in vectorizer.words:
            similar = vectorizer.getClosestNeighbor(word)
            print word, similar[0], similar[1]
                
        
     
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
