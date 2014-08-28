'''
Created on Dec 10, 2013

@author: af
'''

import os
import shutil
import unittest

from com.textylon.dataprovider.simplecorpusdataprovider import \
    SimpleCorpusDataProvider
from com.textylon.tokenizer.simpletokenizer import SimpleTokenizer


class Test(unittest.TestCase):
    corpusPath = '/tmp/testcorpus'

    def setUp(self):
        if os.path.exists(self.corpusPath):
            shutil.rmtree(self.corpusPath)
        # create a directory in temp with some content
        os.mkdir(self.corpusPath)
        for i in range(0, 10):
            f = open(os.path.join(self.corpusPath, 'file' + str(i)), 'w')
            f.write('Beautiful, is; better*than\nugly\n of    course' + str(i))
            f.close()


    def tearDown(self):
        if os.path.exists(self.corpusPath):
            shutil.rmtree(self.corpusPath)


    def testTokenizer(self):
        stopWords = ['is', 'the', 'for', 'this']
        delimiters = ' '
        tokenizer = SimpleTokenizer(stopWords) 
        provider = SimpleCorpusDataProvider(self.corpusPath)
        i = 0
        while(provider.hasNext()):
            data = provider.getNext()
            filename = data[0]
            nextData = data[1]
            tokenizer.setText(nextData)
            tokenizer.tokenize()
            tokens = tokenizer.getAllTokens()
            
            # print filename
            for token in tokens:
                print token, i
                i += 1
        self.assertEqual(i, 60, "total number of tokens is 40 but tokenizer found " + str(i))
        


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
