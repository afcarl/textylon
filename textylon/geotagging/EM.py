'''
Created on 22 Sep 2014

@author: af
'''
'''
1 kd-tree using training samples
2 soft classify test (or perhaps train too) and compute p(region | user) then average coordinates with regards to the weighted classes Sigma p(region | user) * p(coordinates | region)
over all regions
3 measure error
4 kd-tree using training and test samples
5 go to 2
'''

import os
home = '/home/af/Downloads/GeoText.2010-10-12/'
kdtreesource = os.path.join(home, 'code/KdTree.java')
kdtreeclass = os.path.join(home, 'code/KdTree')
bucketSize = 640
partitionMethod = 'halfway'
os.system("javac " + kdtreesource)
os.system("java --classpath " + os.path.join(home, 'code/') + ' ' + kdtreeclass + " " + home + " " + str(bucketSize).strip() + " " + partitionMethod)
