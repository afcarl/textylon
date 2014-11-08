'''
Created on 29 Aug 2014

@author: af
'''

from lxml import etree

doc = etree.parse('/home/af/Downloads/pt.stackoverflow.com/Posts.xml')

print doc

for  elt in doc.getiterator():
    # print elt.tag
    for childelt in elt.getiterator():
        print childelt
    break
