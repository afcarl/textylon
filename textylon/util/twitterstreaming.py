'''
Created on 11 Sep 2014

@author: af
'''
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import codecs





# Go to http://dev.twitter.com and create an app.
# The consumer key and secret will be generated for you after
consumer_key = 'COVV87zJN2wHRfAz7zB5p2QPQ'
consumer_secret = 'HhNmLOsIui0rG04XltDSHdmEBNf9IAtkZeW17U7pFYAuKf8qiv'

# After the step above, you will be redirected to your app's page.
# Create an access token under the the "Your access token" section
access_token = '2205031009-a7FMWRzzTi5wooSMFkYUqyiq1aGREBSMCyBX2vw'
access_token_secret = 'TfPQ5V8X9BOwjWQU7UBTJHTR8kYJzyhM1em8I4YGIcZxh'
tfile = codecs.open('/home/af/tweets.txt', 'a+', 'utf-8')

class StdOutListener(StreamListener):
    """ A listener handles tweets are the received from the stream.
    This is a basic listener that just prints received tweets to stdout.

    """
    def on_data(self, data):
        forbidden =  ['\"country_code\":\"SA\"']
        for forb in forbidden:
            if forb in data:
                return True
        
        data = data.decode( 'unicode-escape' )
        tfile.write(data)
        tfile.flush()
        return True

    def on_error(self, status):
        print status

if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    
    stream = Stream(auth, l)
    stream.filter(locations=[51,29,57,37, 45, 36,48, 39, 53,27, 61, 37, 49, 30, 61, 37], languages=['fa'])
