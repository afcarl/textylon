'''
Created on Dec 20, 2013

@author: af
'''
import codecs
from collections import Counter
import string
from time import gmtime, strftime


exclude = set(string.punctuation)
table = string.maketrans("", "")

# collect vocabulary and count frequencies
def count_freqs(infile, outfile):
    Words = Counter()
    stops = []
    with codecs.open('/home/af/stopwords.txt', 'r', 'utf-8') as stopFile:
        stops = stopFile.read().split()
    print "Started: " + strftime("%H:%M:%S", gmtime())
    with codecs.open(infile, "r", 'utf-8') as inp:
        lineread = 0
        for line in inp:
            # line = string.translate(line, table, string.punctuation).lower()
            lineread += 1
            if lineread % 100000 == 0:
                print "%d lines read" % lineread
                print "%d words found." % len(Words)
            for wrd in line.split():
                if wrd not in stops and len(wrd) < 20 and len(wrd) > 2:
                    Words[wrd] += 1
    
    print "Finished: " + strftime("%H:%M:%S", gmtime())
    # print "Token count: " + str(sum(Words.values()))
    print 'Writing to output file'
    with codecs.open(outfile, 'w', 'utf-8') as out:
        for word, freq in Words.most_common():
            out.write(word + '\t' + str(freq) + '\n')
    print "Finished writing freqs to file"
if __name__ == '__main__':
    count_freqs('/home/af/Documents/alltweets.txt', '/home/af/Documents/alltweetscount.txt')
