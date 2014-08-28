#################################
# Distributional Semantics
#
# July 2013
#
# Revisions:
# ----------
# October 2013: added class structure
# November 2013: use dictionaries instead of arrays for index and context vocabularies, use Counter for frequencies
#
# Copyright (C) Magnus Sahlgren
#################################

from collections import Counter
import logging
import math
import operator
from time import gmtime, strftime

from dsm_classes import Vocabulary, Matrix
import numpy as np
import numpy.random as nprnd
import scipy.spatial as sp


logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
hdlr = logging.FileHandler('log.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

# collect vocabulary and count frequencies
def count_freqs(infile):
    global Words
    Words = Counter()
    print "Started: " + strftime("%H:%M:%S", gmtime())
    inp = open(infile, "r")
    for line in inp.readlines():
        for wrd in line.split():
            Words[wrd] += 1
    inp.close()
    print "Finished: " + strftime("%H:%M:%S", gmtime())
    print "Token count: " + str(sum(Words.values()))

# count cooccurrence frequencies from infile within win
# frequency threhsolds for both index words and context words
# dsm = distributional semantic model
def dsm(infile, win, index_minf, index_maxf, ctx_minf, ctx_maxf):
    global IndVoc
    IndVoc = Vocabulary()
    global CtxVoc
    CtxVoc = Vocabulary()
    global CoocMat
    CoocMat = Matrix()
    print "Started: " + strftime("%H:%M:%S", gmtime())
    inp = open(infile, "r")
    update_vocabulary(index_minf, index_maxf, ctx_minf, ctx_maxf)
    print IndVoc.getlength(), CtxVoc.getlength()
    CoocMat.makematrix(IndVoc.getlength(), CtxVoc.getlength())
    line_nr = 0
    for line in inp.readlines():
        cnt = 0
        wrdlst = line.split()
        for wrd in wrdlst:
            if IndVoc.lookup(wrd):
            # count co-occurrences to the left
                ctx = 1
                while ctx <= win:
                    if (cnt - ctx) >= 0:
                        c = wrdlst[cnt - ctx]
                        update_counts(c, wrd, ctx_minf, ctx_maxf)
                        ctx += 1
                    else:
                        ctx = win + 1
           # count co-occurrences to the right
                ctx = 1
                while ctx <= win:
                    if (cnt + ctx) < len(wrdlst):
                        c = wrdlst[cnt + ctx]
                        update_counts(c, wrd, ctx_minf, ctx_maxf)
                        ctx += 1
                    else:
                        ctx = win + 1
            cnt += 1
        line_nr += 1
    inp.close()
    print "Finished: " + strftime("%H:%M:%S", gmtime())

# check if the word should be indexed and used as ctx word
def update_vocabulary(index_minf, index_maxf, ctx_minf, ctx_maxf):
    i_cnt = 0
    c_cnt = 0
    for w in Words.most_common():
        q = w[1]
        if (q > index_minf) and (q < index_maxf):
            IndVoc.set(w[0], i_cnt)
            i_cnt += 1
        if (q > ctx_minf) and (q < ctx_maxf):
            CtxVoc.set(w[0], c_cnt)
            c_cnt += 1

# update cooccurrence counts
def update_counts(w, wrd, minf, maxf):
    if CtxVoc.lookup(w):
        CoocMat.update(IndVoc.getindex(wrd), CtxVoc.getindex(w), 1)

# TODO: implement direction-sensitive dsm (aka HAL)

######
# Misc
######

# clean up vocabularies and co-occurrence matrix
def clear_ctx():
    IndVoc.delete()
    CtxVoc.delete()
    CoocMat.delete()

# clean up frequency counters
def clear_freq():
    Words.clear()

############
# Evaluation
############

# toefl test
def toefl(testfile):
    inp = open(testfile, "r")
    corr = 0
    tot = 0
    unknown_target = []
    unknown_answer = []
    incorrect = []
    for line in inp.readlines():
        flag = False
        target, correct, alt2, alt3, alt4 = line.replace("(", "").split()
        if IndVoc.lookup(target):
            targetvec = CoocMat.matrix.getrow(IndVoc.getindex(target)).todense()
            tot += 1
            if IndVoc.lookup(correct):
                correctvec = CoocMat.matrix.getrow(IndVoc.getindex(correct)).todense()
                sim = 1 - sp.distance.cosine(targetvec, correctvec)
                if sim > 0.0:
                    flag = True
                else:
                    incorrect.append(target)
                for i in (alt2, alt3, alt4):
                    if IndVoc.lookup(i):
                        i_vec = CoocMat.matrix.getrow(IndVoc.getindex(i)).todense()
                        i_sim = 1 - sp.distance.cosine(targetvec, i_vec)
                        if i_sim > sim:
                            if not target in incorrect:
                                incorrect.append(target)
                            flag = False
                if flag:
                    corr += 1
            else:
                unknown_answer.append(correct)
        else:
            unknown_target.append(target)
    inp.close()
    print "TOEFL synonym score: " + str(float(corr) / float(tot)) + " (" + str(corr) + "/" + str(tot) + ")"
    print "Incorrect: " + str(incorrect)
    print "Unknown targets: " + str(unknown_target)
    print "Unknown answers: " + str(unknown_answer)
    
    logger.info("TOEFL synonym score: " + str(float(corr) / float(tot)) + " (" + str(corr) + "/" + str(tot) + ")")
    logger.info("Incorrect: " + str(incorrect))
    logger.info("Unknown targets: " + str(unknown_target))
    logger.info("Unknown answers: " + str(unknown_answer))

# toefl test
# A is a numpy matrix
def toefl_mat(testfile, A):
    inp = open(testfile, "r")
    flag = False
    corr = 0
    tot = 0
    unknown_target = []
    unknown_answer = []
    incorrect = []
    for line in inp.readlines():
        target, correct, alt2, alt3, alt4 = line.replace("(", "").split()
        if IndVoc.lookup(target):
            targetvec = A[IndVoc.getindex(target), :]
            tot += 1
            if IndVoc.lookup(correct):
                correctvec = A[IndVoc.getindex(correct), :]
                sim = 1 - sp.distance.cosine(targetvec, correctvec)
                if sim > 0.0:
                    flag = True
                for i in (alt2, alt3, alt4):
                    if IndVoc.lookup(i):
                        i_vec = A[IndVoc.getindex(i), :]
                        i_sim = 1 - sp.distance.cosine(targetvec, i_vec)
                        if i_sim > sim:
                            if not target in incorrect:
                                incorrect.append(target)
                            flag = False
                if flag:
                    corr += 1
            else:
                unknown_answer.append(correct)
        else:
            unknown_target.append(target)
    inp.close()
    print "TOEFL synonym score: " + str(float(corr) / float(tot))
    print "Incorrect: " + str(incorrect)
    print "Unknown targets: " + str(unknown_target)
    print "Unknown answers: " + str(unknown_answer)
    logger.info("TOEFL synonym score: " + str(float(corr) / float(tot)) + " (" + str(corr) + "/" + str(tot) + ")")
    logger.info("Incorrect: " + str(incorrect))
    logger.info("Unknown targets: " + str(unknown_target))
    logger.info("Unknown answers: " + str(unknown_answer))

# find the nr nearest neighbors to word using cosine similarity
# TODO: optimization
def nns(word, nr):
    res = {}
    if IndVoc.lookup(word):
        w_vec = CoocMat.matrix.getrow(IndVoc.getindex(word)).todense()
        for k in IndVoc.hsh:
            k_vec = CoocMat.matrix.getrow(IndVoc.getindex(k)).todense()
            sim = 1 - sp.distance.cosine(w_vec, k_vec)
            if (not math.isnan(sim)) and (not math.isinf(sim)):
                res[k] = sim
    sorted_res = sorted(res.iteritems(), key=lambda(k, v): v, reverse=True)
    print word, sorted_res[0:nr]
    # for r in sorted_res[1:nr]: # 1 to avoid including word
        # print r[0] + ' ' + str(r[1][0][0])
def loadExternalRepresentation(textFile):
    """
    this function loads an external representation IndVoc and CoocMatrix
    the format of the file like this:
    first line is
    vocabulary_count dimensionality
    the other lines are of the follwoin format
    word value value .... value
    and the number of values is the dimensionality in the first line
    """
    global IndVoc
    IndVoc = Vocabulary()
    global CtxVoc
    CtxVoc = Vocabulary()
    global CoocMat
    CoocMat = Matrix()
    
    f = open(textFile, 'r')
    # the first line is the vocabulary size and the representation dimensionality
    lines = f.readlines()
    theFirstTime = True
    reprDict = {}
    vocabSize = 0
    dimensionality = 0
    for line in lines:
        if theFirstTime:
            theFirstTime = False
            fs = line.split()
            vocabSize = int(fs[0])
            dimensionality = int(fs[1])
            CoocMat.makematrix(vocabSize, dimensionality)
            continue
        fields = line.split()
        word = fields[0]
        l = fields[1:] 
        vector = np.array(map(float, l))
        i = 0
        for v in np.nditer(vector):
            CoocMat.update(IndVoc.getlength(), i, v)
        IndVoc.set(word, IndVoc.getlength())

    
    

loadExternalRepresentation('/home/af/word2vec/w5bow.txt') 
toefl('/home/af/Downloads/toefl.tsts')

"""
infile = 'tweets.txt'
count_freqs(infile)        


for maxFreq in (5000, 10000, 20000, 50000, 100000, 151000):
    for windowSize in range (1, 10):
        dsm(infile, windowSize, 0, maxFreq, 0, maxFreq)
        print  0, maxFreq, windowSize
        logger.info("minFreq = " + str(0) +  " maxFreq = " + str(maxFreq) + " windowSize =  " + str(windowSize))
        originalCoocMat = CoocMat
        for dimension in range(100, 1100, 100):
            CoocMat = CoocMat.t_svd(dimension)
            toefl_mat('toefl.tsts', CoocMat)
            CoocMat = originalCoocMat
            #CoocMat.tfidf()
            #logger.info("tfidf")
            #toefl('/home/af/Downloads/toefl.tsts')
        clear_ctx()
"""
