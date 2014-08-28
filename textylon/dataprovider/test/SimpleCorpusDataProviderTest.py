'''
Created on Dec 10, 2013

@author: af
'''


import io
import os
import shutil
import unittest

from textylon.dataprovider.SimpleCorpusDataProvider import SimpleCorpusDataProvider


class SimpleCorpusDataProviderTest(unittest.TestCase):
    corpusPath = '/tmp/testcorpus'

    def setUp(self):
        if os.path.exists(self.corpusPath):
            shutil.rmtree(self.corpusPath)
        # create a directory in temp with some content
        os.mkdir(self.corpusPath)
        for i in range(0, 10):
            f = open(os.path.join(self.corpusPath, 'file' + str(i)), 'w')
            f.write('this is the content for file number ' + str(i))
            f.close()


    def tearDown(self):
        if os.path.exists(self.corpusPath):
            shutil.rmtree(self.corpusPath)


    def testGetNext(self):
        provider = SimpleCorpusDataProvider(self.corpusPath)
        i = 0
        while(provider.hasNext()):
            nextData = provider.getNext()
            print nextData[0], nextData[1]
            i += 1
        self.assertEqual(i, 10, "the number of written files were 10 while the number of read files is " + str(i))
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
