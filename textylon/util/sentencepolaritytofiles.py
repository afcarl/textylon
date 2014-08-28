'''
Created on Feb 18, 2014
This class is intended to convert sentence polarity dataset available here:
https://www.cs.cornell.edu/people/pabo/movie-review-data/
into a format which is acceptible by our classification benchmarking program.
it reads the positive and negative sentences and produces one file for each sentence in a k-fold cross validation directory structure.
@author: af
'''
import os
import shutil


SENTENCE_POLARITY_DATASET_DIR = '/home/af/Downloads/rt-polaritydata/rt-polaritydata'
POS_FILE = os.path.join(SENTENCE_POLARITY_DATASET_DIR, 'rt-polarity.pos')
NEG_FILE = os.path.join(SENTENCE_POLARITY_DATASET_DIR, 'rt-polarity.neg')
NUM_FOLDS = 10

posSents = open(POS_FILE, 'r').readlines()
negSents = open(NEG_FILE, 'r').readlines()




 
os.mkdir(os.path.join(SENTENCE_POLARITY_DATASET_DIR, 'pos'))
os.mkdir(os.path.join(SENTENCE_POLARITY_DATASET_DIR, 'neg'))
for i in range(0, len(posSents)):
    f = open(os.path.join(SENTENCE_POLARITY_DATASET_DIR, 'pos/' + str(i).strip() + '.pos'), 'w')
    f.write(posSents[i])
    f.close()
for i in range(0, len(negSents)):
    f = open(os.path.join(SENTENCE_POLARITY_DATASET_DIR, 'neg/' + str(i).strip() + '.neg'), 'w')
    f.write(negSents[i])
    f.close()
