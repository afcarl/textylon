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

import codecs
from collections import Counter
import logging
import math, pickle
import operator
from sklearn.feature_extraction.text import TfidfTransformer
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
class DSM(object):
    
    def __init__(self, encoding='utf-8'):
        self.Words = Counter()
        self.IndVoc = Vocabulary()
        self.CtxVoc = Vocabulary()
        self.CoocMat = Matrix()
        self.numLines = 0
        self.encoding = encoding
    # collect vocabulary and count frequencies
    def count_freqs(self, infile):

        print "Started: " + strftime("%H:%M:%S", gmtime())
        with codecs.open(infile, "r", self.encoding) as inp:
            for line in inp:
                self.numLines += 1
                for wrd in line.split():
                    self.Words[wrd] += 1
        print "Finished: " + strftime("%H:%M:%S", gmtime())
        print "Token count: " + str(sum(self.Words.values()))
    
    # count cooccurrence frequencies from infile within win
    # frequency threhsolds for both index words and context words
    # dsm = distributional semantic model
    def dsm(self, infile, win, index_minf, index_maxf, ctx_minf, ctx_maxf):



        print "Started: " + strftime("%H:%M:%S", gmtime())
        with codecs.open(infile, "r", self.encoding) as inp:
            self.update_vocabulary(index_minf, index_maxf, ctx_minf, ctx_maxf)
            # print self.IndVoc.getlength(), self.CtxVoc.getlength()
            self.CoocMat.makematrix(self.IndVoc.getlength(), self.CtxVoc.getlength())
            line_nr = 0
            for line in inp:
                cnt = 0
                wrdlst = line.split()
                for wrd in wrdlst:
                    if self.IndVoc.lookup(wrd):
                    # count co-occurrences to the left
                        ctx = 1
                        while ctx <= win:
                            if (cnt - ctx) >= 0:
                                c = wrdlst[cnt - ctx]
                                self.update_counts(c, wrd, ctx_minf, ctx_maxf)
                                ctx += 1
                            else:
                                ctx = win + 1
                   # count co-occurrences to the right
                        ctx = 1
                        while ctx <= win:
                            if (cnt + ctx) < len(wrdlst):
                                c = wrdlst[cnt + ctx]
                                self.update_counts(c, wrd, ctx_minf, ctx_maxf)
                                ctx += 1
                            else:
                                ctx = win + 1
                    cnt += 1
                line_nr += 1
        print "Finished: " + strftime("%H:%M:%S", gmtime())
    
    # check if the word should be indexed and used as ctx word
    def update_vocabulary(self, index_minf, index_maxf, ctx_minf, ctx_maxf):
        i_cnt = 0
        c_cnt = 0
        for w in self.Words.most_common():
            q = w[1]
            if (q > index_minf) and (q < index_maxf):
                self.IndVoc.set(w[0], i_cnt)
                i_cnt += 1
            if (q > ctx_minf) and (q < ctx_maxf):
                self.CtxVoc.set(w[0], c_cnt)
                c_cnt += 1
    
    # update cooccurrence counts
    def update_counts(self, w, wrd, minf, maxf):
        if self.CtxVoc.lookup(w):
            self.CoocMat.update(self.IndVoc.getindex(wrd), self.CtxVoc.getindex(w), 1)
    
    # TODO: implement direction-sensitive dsm (aka HAL)
    
    ######
    # Misc
    ######
    
    # clean up vocabularies and co-occurrence matrix
    def clear_ctx(self):
        self.IndVoc.delete()
        self.CtxVoc.delete()
        self.CoocMat.delete()
    
    # clean up frequency counters
    def clear_freq(self):
        self.Words.clear()
    
    ############
    # Evaluation
    ############
    
    # toefl test
    def toefl(self, testfile):
        inp = open(testfile, "r")
        corr = 0
        tot = 0
        unknown_target = []
        unknown_answer = []
        incorrect = []
        for line in inp.readlines():
            flag = False
            target, correct, alt2, alt3, alt4 = line.replace("(", "").split()
            if self.IndVoc.lookup(target):
                targetvec = self.CoocMat.matrix.getrow(self.IndVoc.getindex(target)).todense()
                tot += 1
                if self.IndVoc.lookup(correct):
                    correctvec = self.CoocMat.matrix.getrow(self.IndVoc.getindex(correct)).todense()
                    sim = 1 - sp.distance.cosine(targetvec, correctvec)
                    if sim > 0.0:
                        flag = True
                    else:
                        incorrect.append(target)
                    for i in (alt2, alt3, alt4):
                        if self.IndVoc.lookup(i):
                            i_vec = self.CoocMat.matrix.getrow(self.IndVoc.getindex(i)).todense()
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
    def toefl_mat(self, testfile, A):
        inp = open(testfile, "r")
        flag = False
        corr = 0
        tot = 0
        unknown_target = []
        unknown_answer = []
        incorrect = []
        for line in inp.readlines():
            target, correct, alt2, alt3, alt4 = line.replace("(", "").split()
            if self.IndVoc.lookup(target):
                targetvec = A[self.IndVoc.getindex(target), :]
                tot += 1
                if self.IndVoc.lookup(correct):
                    correctvec = A[self.IndVoc.getindex(correct), :]
                    sim = 1 - sp.distance.cosine(targetvec, correctvec)
                    if sim > 0.0:
                        flag = True
                    for i in (alt2, alt3, alt4):
                        if self.IndVoc.lookup(i):
                            i_vec = A[self.IndVoc.getindex(i), :]
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
    def nns(self, word, nr):
        res = {}
        if self.IndVoc.lookup(word):
            w_vec = self.CoocMat.matrix.getrow(self.IndVoc.getindex(word)).todense()
            for k in self.IndVoc.hsh:
                k_vec = self.CoocMat.matrix.getrow(self.IndVoc.getindex(k)).todense()
                sim = 1 - sp.distance.cosine(w_vec, k_vec)
                if (not math.isnan(sim)) and (not math.isinf(sim)):
                    res[k] = sim
        sorted_res = sorted(res.iteritems(), key=lambda(k, v): v, reverse=True)
        # print word, sorted_res[0:nr]
        return sorted_res[0:nr]
        # for r in sorted_res[1:nr]: # 1 to avoid including word
            # print r[0] + ' ' + str(r[1][0][0])
    def loadExternalRepresentation(self, textFile):
        """
        this function loads an external representation self.IndVoc and CoocMatrix
        the format of the file like this:
        first line is
        vocabulary_count dimensionality
        the other lines are of the follwoin format
        word value value .... value
        and the number of values is the dimensionality in the first line
        """


        
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
                self.CoocMat.makematrix(vocabSize, dimensionality)
                continue
            fields = line.split()
            word = fields[0]
            l = fields[1:] 
            vector = np.array(map(float, l))
            i = 0
            for v in np.nditer(vector):
                self.CoocMat.update(self.IndVoc.getlength(), i, v)
            self.IndVoc.set(word, self.IndVoc.getlength())
    def dumpVocabAndCoocMatrix(self):
        print "start pickling dsm model..." + strftime("%H:%M:%S", gmtime())
        pickle.dump((self.IndVoc, self.CoocMat, self.CtxVoc), open("dsm.pkl", "wb"))
        print "Finished: " + strftime("%H:%M:%S", gmtime())
    def loadVocabAndCoocMatrix(self):
        print "start depickling dsm model..." + strftime("%H:%M:%S", gmtime())
        self.IndVoc, self.CoocMat, self.CtxVoc = pickle.load(open("dsm.pkl", "rb"))
        print "Finished: " + strftime("%H:%M:%S", gmtime())
    def tfidf(self):
        transformer = TfidfTransformer()
        self.CoocMat.matrix = transformer.fit_transform(self.CoocMat.matrix)      
 
'''
This class produces DSM models using word-document cooccurrence matrix.
It uses one text file containing each document in one line.
'''
class DSMDocument(object):
    
    def __init__(self, encoding='utf-8'):
        self.Words = Counter()
        self.IndVoc = Vocabulary()
        self.CoocMat = Matrix()
        self.numLines = 0
        self.encoding = encoding
    # collect vocabulary and count frequencies
    def count_freqs(self, infile):

        print "Started: " + strftime("%H:%M:%S", gmtime())
        with codecs.open(infile, "r", self.encoding) as inp:
            for line in inp:
                self.numLines += 1
                for wrd in line.split():
                    self.Words[wrd] += 1
        print "Finished: " + strftime("%H:%M:%S", gmtime())
        print "Token count: " + str(sum(self.Words.values()))
        print "%d is the number of lines/documents." % self.numLines
    
    # count cooccurrence frequencies from infile within win
    # frequency threhsolds for both index words and context words
    # dsm = distributional semantic model
    def dsm(self, infile, index_minf=0, index_maxf=1000000):



        print "Started: " + strftime("%H:%M:%S", gmtime())
        with codecs.open(infile, "r", self.encoding) as inp:
            self.update_vocabulary(index_minf, index_maxf)
            print self.IndVoc.getlength()
            self.CoocMat.makematrix(self.IndVoc.getlength(), self.numLines)
            doc_nr = 0
            for line in inp:
                cnt = 0
                wrdlst = line.split()
                for wrd in wrdlst:
                    if self.IndVoc.lookup(wrd):
                        self.update_counts(wrd, doc_nr)
                        
                doc_nr += 1
        print self.CoocMat.matrix.shape
        print "Finished: " + strftime("%H:%M:%S", gmtime())
    
    # check if the word should be indexed and used as ctx word
    def update_vocabulary(self, index_minf=0, index_maxf=1000000):
        i_cnt = 0
        for w in self.Words.most_common():
            q = w[1]
            if (q > index_minf) and (q < index_maxf):
                self.IndVoc.set(w[0], i_cnt)
                i_cnt += 1
    
    # update cooccurrence counts
    def update_counts(self, wrd, doc_number):
        self.CoocMat.update(self.IndVoc.getindex(wrd), doc_number, 1)
    
    # TODO: implement direction-sensitive dsm (aka HAL)
    
    ######
    # Misc
    ######
    
    # clean up vocabularies and co-occurrence matrix
    def clear_ctx(self):
        self.IndVoc.delete()
        self.CoocMat.delete()
    
    # clean up frequency counters
    def clear_freq(self):
        self.Words.clear()
    
    ############
    # Evaluation
    ############
    
    # toefl test
    def toefl(self, testfile):
        inp = open(testfile, "r")
        corr = 0
        tot = 0
        unknown_target = []
        unknown_answer = []
        incorrect = []
        for line in inp.readlines():
            flag = False
            target, correct, alt2, alt3, alt4 = line.replace("(", "").split()
            if self.IndVoc.lookup(target):
                targetvec = self.CoocMat.matrix.getrow(self.IndVoc.getindex(target)).todense()
                tot += 1
                if self.IndVoc.lookup(correct):
                    correctvec = self.CoocMat.matrix.getrow(self.IndVoc.getindex(correct)).todense()
                    sim = 1 - sp.distance.cosine(targetvec, correctvec)
                    if sim > 0.0:
                        flag = True
                    else:
                        incorrect.append(target)
                    for i in (alt2, alt3, alt4):
                        if self.IndVoc.lookup(i):
                            i_vec = self.CoocMat.matrix.getrow(self.IndVoc.getindex(i)).todense()
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
    def toefl_mat(self, testfile, A):
        inp = open(testfile, "r")
        flag = False
        corr = 0
        tot = 0
        unknown_target = []
        unknown_answer = []
        incorrect = []
        for line in inp.readlines():
            target, correct, alt2, alt3, alt4 = line.replace("(", "").split()
            if self.IndVoc.lookup(target):
                targetvec = A[self.IndVoc.getindex(target), :]
                tot += 1
                if self.IndVoc.lookup(correct):
                    correctvec = A[self.IndVoc.getindex(correct), :]
                    sim = 1 - sp.distance.cosine(targetvec, correctvec)
                    if sim > 0.0:
                        flag = True
                    for i in (alt2, alt3, alt4):
                        if self.IndVoc.lookup(i):
                            i_vec = A[self.IndVoc.getindex(i), :]
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
    def nns(self, word, nr):
        res = {}
        if self.IndVoc.lookup(word):
            w_vec = self.CoocMat.matrix.getrow(self.IndVoc.getindex(word)).todense()
            for k in self.IndVoc.hsh:
                k_vec = self.CoocMat.matrix.getrow(self.IndVoc.getindex(k)).todense()
                sim = 1 - sp.distance.cosine(w_vec, k_vec)
                if (not math.isnan(sim)) and (not math.isinf(sim)):
                    res[k] = sim
        sorted_res = sorted(res.iteritems(), key=lambda(k, v): v, reverse=True)
        print word, sorted_res[0:nr]
        # for r in sorted_res[1:nr]: # 1 to avoid including word
            # print r[0] + ' ' + str(r[1][0][0])
    def loadExternalRepresentation(self, textFile):
        """
        this function loads an external representation self.IndVoc and CoocMatrix
        the format of the file like this:
        first line is
        vocabulary_count dimensionality
        the other lines are of the follwoin format
        word value value .... value
        and the number of values is the dimensionality in the first line
        """


        
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
                self.CoocMat.makematrix(vocabSize, dimensionality)
                continue
            fields = line.split()
            word = fields[0]
            l = fields[1:] 
            vector = np.array(map(float, l))
            i = 0
            for v in np.nditer(vector):
                self.CoocMat.update(self.IndVoc.getlength(), i, v)
            self.IndVoc.set(word, self.IndVoc.getlength())
    
    def save(self, filename='dsm.pkl'):
        print "start pickling dsm model..." + strftime("%H:%M:%S", gmtime())
        pickle.dump((self.IndVoc, self.CoocMat), open(filename, "wb"))
        print "Finished: " + strftime("%H:%M:%S", gmtime())
    def load(self, filename='dsm.pkl'):
        print "start depickling dsm model..." + strftime("%H:%M:%S", gmtime())
        self.IndVoc, self.CoocMat = pickle.load(open(filename, "rb"))
        print "Finished: " + strftime("%H:%M:%S", gmtime())  
    def tfidf(self):
        transformer = TfidfTransformer()
        self.CoocMat.matrix = transformer.fit_transform(self.CoocMat.matrix)      
        
if __name__ == "__main__":
    infile = '/home/af/Downloads/rt-polaritydata/rt-polaritydata/allsents.txt'
    dsm = DSM('utf-8')
    dsm.count_freqs(infile)
    dsm.dsm(infile, 2, 5, 10000000, 5, 10000000)
    dsm.dumpVocabAndCoocMatrix()
    dsm.CtxVoc = None
    dsm.IndVoc = None
    dsm.CoocMat = None
    dsm.loadVocabAndCoocMatrix()
    print len(dsm.CtxVoc.hsh)
    print len(dsm.IndVoc.hsh)
    print dsm.CoocMat.matrix.shape  
