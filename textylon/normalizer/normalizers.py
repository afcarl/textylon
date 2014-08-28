'''
Created on Nov 29, 2013

@author: Afshin Rahimi
'''

class AbstractNormalizer(object):
    '''
    This is an abstract normalizer base class for various normalizers
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
    def normalize(self, token):
        '''
        normalizes a token and returns the normalized token      
        '''
    def multiNormalize(self, token, normalizers):
        '''
        Given a list of normalizers it applies all of them on token with the same order
        and then returns to normalized token.
        '''
        for normalizer in normalizers:
            token = normalizer.normalize(token)
        return token

class LowerCaseNormalizer(AbstractNormalizer):
    '''
    This is a simple normalizer calls that just lower cases a token.
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(LowerCaseNormalizer, self).__init__()
        
    def normalize(self, token):
        if token != None:
            return token.lower()
        else:
            return None

class PassNormalizer(AbstractNormalizer):
    '''this class does nothing
    '''
    def __init__(self):
        super(PassNormalizer, self).__init__()
    def normalize(self, token):
        return token
        
