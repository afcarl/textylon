'''
Created on Mar 14, 2014
This script uses random indexing to convert a textual dump of wikipedia into random indexes for each word.
It requires the wikipedia text to have each document in one line.
@author: af
'''

import Queue
import codecs
import logging
import pickle
import threading
from time import strftime, gmtime, sleep

from textylon.dataprovider.dataproviders import OneFileCorpusDataProvider
from textylon.normalizer.normalizers import PassNormalizer
from textylon.tokenizer.tokenizers import SimpleTokenizer
from textylon.vectorizer.rindexvectors import DocRandomIndexVectorizer, RandomIndexVectorizer, DocVectorizerThread


logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
hdlr = logging.FileHandler('log.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

corpusFileAddress = "/mnt/data/afshin/enwiki/wikitext-20140203.txt"
stopWords = ['this', 'the',
              'is', 'for', 'am', 'are',
              'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
              'a', 'an', 'it', 'not', 'on', 'with', 'he', 'she',
              'as', 'do', 'at', 'but', 'his', 'her', 'by', 'from',
              'they', 'you', 'into']


content = None
with codecs.open('stopwords.txt', 'r', 'utf-8') as f:
    content = f.read()
with codecs.open('replabtrainingall.txt', 'r', 'utf-8') as vocfile:
    vocContent = f.read()
vocabulary = vocContent.split()
stopWords = content.split()
tokenizer = SimpleTokenizer(stopWords, vocabulary) 
provider = OneFileCorpusDataProvider(corpusFileAddress, encoding='utf-8')
normalizer = PassNormalizer()
maxQueueSize = 20000
q = Queue.Queue(maxQueueSize)
numThreads = 10
threads = []
words = {}
dimension = 2048
lock = threading.Lock()
for i in range(0, numThreads):
    t = DocVectorizerThread(str(i), q, words, lock, normalizer, tokenizer, dimension)
    t.setDaemon(True)
    threads.append(t)
    t.start()

vectorizer = DocRandomIndexVectorizer(q, dimension=2048, nonZero=7)
# vectorizer = MultiThreadedRandomIndexVectorizer(tokenizer, normalizer, 300, 8, 4, 'context',10, 20000)
while(provider.hasNext()):
    text = provider.getNext()
    vectorizer.vectorize(text)
    if vectorizer.docNum % 100000 == 0:
        print "vectorized %d documents." % vectorizer.docNum
        logger.info("queue size is " + str(q.qsize()))
logger.info("waiting for working threads to finish their job...")
q.join()

logger.info("all worker threads finished their job.")
print "pickling random index.." + strftime("%H:%M:%S", gmtime())
with open('randomindex.pkl', 'wb') as pfile:
    pickle.dump(words, pfile)
print "pickling finished." + strftime("%H:%M:%S", gmtime())
'''
for word in vectorizer.words:
    similar = vectorizer.getClosestNeighbor(word)
    print word, similar[0], similar[1]
'''
