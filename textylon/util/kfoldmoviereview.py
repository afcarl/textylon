'''
Created on Feb 18, 2014
This class converts movie review dataset into a 10-fold that can be used in our classification benchmarking tool
for cross-validation.
@author: af
'''
import os


MOVIE_REVIEW_DATASET = '/home/af/Downloads/movie_review_kfold/review_polarity/txt_sentoken'
POS_DIR = os.path.join(MOVIE_REVIEW_DATASET, 'pos')
NEG_DIR = os.path.join(MOVIE_REVIEW_DATASET, 'neg')
KFOLD = 10
for i in range(0, KFOLD):
    os.mkdir(os.path.join(MOVIE_REVIEW_DATASET, str(i))) 
