import codecs
from collections import Counter
import re

option = 5

if option==1:
    with codecs.open('train.txt', 'r', 'utf-8') as inf:
        with codecs.open('t.txt', 'w', 'utf-8') as outf:
            for line in inf:
                fields  = line.split('|||')
                #print fields[0], fields[1], fields[2]
                if len(fields) == 3:
                    outf.write(fields[2].strip() + '\n')
                elif len(fields) == 4:
                    outf.write(fields[2].strip() + ' ' + fields[3].strip() + '\n')
                else:
                    print line
if option==2:
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
                            outLoc = d+ str(i+1).strip()
                        outf.write(outLoc + ' ')
                outf.write('\n')

if option==3:
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
    
if option==4:
    cnt = Counter()
    with codecs.open('AU.txt', 'r', 'utf-8') as inf:
        for line in inf:
            words = line.split()
            for word in words:
                if len(word) > 3:
                    cnt[word] += 1
    print cnt.most_common(200)

if option==5:
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
            fields  = line.split('|||')
            ids.append(fields[0].strip())
            #print fields[0], fields[1], fields[2]
            ulocs = Counter(fields[1].strip().split())            
            nlocs = []
            for uloc in ulocs:
                nlocs.append(uloc)
                for i in range(2, ulocs[uloc] + 1):                
                    nlocs.append(uloc+str(i).strip())
            trainLoc[fields[0].strip()] = nlocs    
            #trainLoc[fields[0].strip()] = fields[1].strip().split()
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
