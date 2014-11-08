import codecs
from collections import Counter
import re
import os
import urllib2
import nltk
import sys
from bs4 import BeautifulSoup
from goose import Goose
import string
import unicodedata
home = '/home/af/Downloads/alta/'
option = 10

if option == 1:
    with codecs.open('train.txt', 'r', 'utf-8') as inf:
        with codecs.open('t.txt', 'w', 'utf-8') as outf:
            for line in inf:
                fields = line.split('|||')
                # print fields[0], fields[1], fields[2]
                if len(fields) == 3:
                    outf.write(fields[2].strip() + '\n')
                elif len(fields) == 4:
                    outf.write(fields[2].strip() + ' ' + fields[3].strip() + '\n')
                else:
                    print line
if option == 2:
    with codecs.open('t.xml', 'r', 'utf-8') as inf:
        with codecs.open('locations.txt', 'w', 'utf-8') as outf:
            for line in inf:
                found = re.findall('<Location>(.*?)</Location>', line)        
                found = [x.lower() for x in found]
                dic = Counter(found)
                for d in dic:
                    for i in range(0, dic[d]):
                        if i == 0:
                            outLoc = d
                        else:
                            outLoc = d + str(i + 1).strip()
                        outf.write(outLoc + ' ')
                outf.write('\n')

if option == 3:
    locations = None
    tweetIds = []
    with codecs.open('locations.txt', 'r', 'utf-8') as locf:
        locations = locf.readlines()
    with codecs.open('train.txt', 'r', 'utf-8') as trainf:
        for line in trainf:
            fields = line.split('|||')
            tweetId = fields[0]
            tweetIds.append(tweetId)
    numLocationSets = len(locations)
    numTweets = len(tweetIds)
    if numLocationSets != numTweets:
        print 'Fatal Error: the number of location sets are not equal to the number of tweets!!!!!!!'
        exit(-1)
    with codecs.open('final_output.txt', 'w', 'utf-8') as outf:
        for i in range(0, numTweets):
            location = locations[i].strip()
            if location == '':
                print 'location is empty'
                location = 'NONE'
            outf.write(tweetIds[i] + ',' + location + '\n') 
    
if option == 4:
    cnt = Counter()
    with codecs.open('AU.txt', 'r', 'utf-8') as inf:
        for line in inf:
            words = line.split()
            for word in words:
                if len(word) > 3:
                    cnt[word] += 1
    print cnt.most_common(200)

if option == 5:
    ids = []
    trainLoc = {}
    trainText = {}
    geoLoc = {}
    falsepos = {}
    falseneg = {}
    truepos = {}
    trueneg = {}
    falseposNum = 0
    falsenegNum = 0
    truenegNum = 0
    trueposNum = 0
    with codecs.open('Train_Data_2.txt', 'r', 'utf-8') as inf:
        for line in inf:
            fields = line.split('|||')
            ids.append(fields[0].strip())
            # print fields[0], fields[1], fields[2]
            ulocs = Counter(fields[1].strip().split())            
            nlocs = []
            for uloc in ulocs:
                nlocs.append(uloc)
                for i in range(2, ulocs[uloc] + 1):                
                    nlocs.append(uloc + str(i).strip())
            trainLoc[fields[0].strip()] = nlocs    
            # trainLoc[fields[0].strip()] = fields[1].strip().split()
            trainText[fields[0].strip()] = fields[2].strip()
    with codecs.open('merge_crf++_stanfordner_None_train.txt', 'r', 'utf-8') as inf:
        for line in inf:
            fields = line.split(',')
            geoLoc[fields[0].strip()] = fields[1].strip().split()
    print len(trainLoc)
    print len(trainText)
    print len(geoLoc)
    print len(ids)
    for id in ids:
        locs = trainLoc[id]
        detected = geoLoc[id]

        falsepos[id] = ""
        falseneg[id] = ""
        truepos[id] = ""
        trueneg[id] = ""

        for l in locs:
            if l not in detected:
                falseneg[id] = falseneg[id] + " " + l
                falsenegNum += 1
            else:
                truepos[id] = truepos[id] + " " + l
                trueposNum += 1
        
        for l in detected:
            if l == 'NONE':
                continue
            if l not in locs:
                falsepos[id] = falsepos[id] + " " + l
                falseposNum += 1        
                
    with codecs.open('analysis.html', 'w', 'utf-8') as outf:
        outf.write('<html><body>')
        outf.write('<h1><font color=green>green = true positive (we detected them correctly) </font><br> <font color=red>red = false negative (we should have detected them as location) </font><br> <font color=blue>blue = false positive (we have wrongly detected them as location)</font><br></h1>')
        for id in ids:
            outf.write(id + "|||" + '<font color=green>' + truepos[id] + "</font>\n") 
            outf.write("<font color=red>" + falseneg[id] + "</font>\n") 
            outf.write("<font color=blue>" + falsepos[id] + "</font>\n")
            outf.write("|||" + trainText[id] + "<br>" + '\n')
            outf.write('</body></html>')
            '''
Created on 10 Sep 2014

@author: af
'''
if option == 6:
    ifname = 'Train_Data_2.txt'
    ofname = 'links.txt'
    ifname = os.path.join(home, ifname)
    ofname = os.path.join(home, ofname)
    with codecs.open(ofname, 'w', 'utf8') as outf:
        with codecs.open(ifname, 'r', 'utf-8') as inf:
            for s in inf:
                t = s[s.find("http://"):]
                t = t[:t.find(" ")].strip()
                if t != '':
                    outf.write(t + '\n')

if option == 7:
    with codecs.open(os.path.join(home, 'trainlinks.txt'), 'r', 'utf-8') as inf:
        with codecs.open(os.path.join(home, 'trainurltext.txt'), 'w', 'utf-8') as outf:
            for url in inf:
                bs = False
                if bs:
                    response = urllib2.urlopen('http://t.co/uRFq9gAZ')
                    html = response.read()
                    soup = BeautifulSoup(html) 
                    print(soup.get_text())
                gooose = True
                if gooose:
                    try:
                        g = Goose()
                        article = g.extract(url=url)
                        text = article.cleaned_text
                        outf.write(text + '\n')
                    except:
                        pass

if option == 8:
    dic = []
    with codecs.open(os.path.join(home, 'aunz.txt'), 'r', 'utf-8') as inf:
        for line in inf:
            # words = line.split()
            # dic.extend(words)
            dic.append(line.strip())
    dic = set(dic)
    with codecs.open(os.path.join(home, 'certainly detect these locations - compiled from geonames of au and nz.txt'), 'w', 'utf-8') as outf:
        for i in dic:
            outf.write(i + '\n')

if option == 9:
    dic = []
    with codecs.open(os.path.join(home, 'aunz.txt'), 'r', 'utf-8') as inf:
        for line in inf:
            words = line.split()
            dic.extend(words)
            # dic.append(line.strip())
    c = Counter(dic)
    sorted = c.most_common()
    with codecs.open(os.path.join(home, 'needs manual harassing sorted by frequency ascending - compiled from geonames of au and nz.txt'), 'w', 'utf-8') as outf:
        for i in sorted:
            word , freq = i
            outf.write(word + '\t' + str(freq).strip() + '\n')
    
    text = ''
    with codecs.open(os.path.join(home, 'Test_Data.txt'), 'r', 'utf-8') as inf:
        text = inf.read()
    words = re.split(" |\||\n|\r|\t|#|/|\(|\)|\@|\'|\"|\!|\?", text)
    words = [w.lower() for w in words]
    words = set(words)
    print words
    
    
    sorted = c.most_common()
    with codecs.open(os.path.join(home, 'just words in test collection - needs manual harassing sorted by frequency ascending - compiled from geonames of au and nz.txt'), 'w', 'utf-8') as outf:
        for i in sorted:
            word , freq = i
            if word.lower() in words:
                outf.write(word + '\t' + str(freq).strip() + '\n')

if option == 10:
    print "Usage: python alta.py test-file result-file dictionary-file outputfile"
    if len(sys.argv) != 5:
        print "Fatal error: wrong number of arguments."
        sys.exit()
    pythonfilename, testfile, resultfile, dictionaryfile, outputfile = sys.argv
    encoding = 'utf-8'
    with codecs.open(testfile, 'r', encoding=encoding) as inf:
        testLines = inf.readlines()
    with codecs.open(resultfile, 'r', encoding=encoding) as inf:
        resultLines = inf.readlines()
    with codecs.open(dictionaryfile, 'r', encoding=encoding) as inf:
        dictionaryLines = inf.readlines()
    
    testDataDic = {}
    resultDic = {}
    newResultDic = {}
    gazetter = [line.lower().strip() for line in dictionaryLines]
    
    for line in testLines:
        fields = line.split('|||')
        # Felix's normalization translate doesn't work with unicode here
        # fields[1] = fields[1].lower().translate(None, "#'\"@") 
        if len(fields) < 2:
            continue
        fields[1] = fields[1].lower()
        fields[1] = fields[1].replace('#', '')
        fields[1] = fields[1].replace('\"', '')
        fields[1] = fields[1].replace('@', '')
        testDataDic[fields[0]] = ' '.join(re.split("\W+", fields[1]))
    
    for line in resultLines:
        fields = line.split(',')
        resultDic[fields[0]] = fields[1].split()
    newItemCount = 0
    for tweetID in resultDic:
        tweet_text = testDataDic[tweetID]
        detected_locations = resultDic[tweetID] 
        # order matters here
        for dicItem in gazetter:
            # check if it is in tweet_text
            if dicItem in tweet_text:
                # check if items are in results, if not add each token
                tokens = dicItem.split()
                for token in tokens:
                    if token in detected_locations:
                        continue
                    else:
                        textWords = tweet_text.split()
                        for word in textWords:
                            if token in word:
                                if word in detected_locations:
                                    pass
                                else:
                                    detected_locations.append(word)
                                    newItemCount += 1  
                            else:
                                pass

            else:
                continue
        newResultDic[tweetID] = detected_locations
    
    print "Detected " + str(newItemCount) + " new locations."
    
    # write the newResultDic to the outputfile
    print "writing the final results in " + outputfile
    with codecs.open(outputfile, 'w', encoding=encoding) as outf:
        for tweetID in sorted(newResultDic):
            locations = newResultDic[tweetID]
            outf.write(tweetID + ',')
            locStr = ' '.join(locations)
            outf.write(locStr + '\n')
    print "new result file is ready."
