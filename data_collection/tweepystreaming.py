#!/usr/bin/env python
# coding: utf-8
#pip install tweepy

# AWS resource
import boto3
import re

#   import tweepy things
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
# import others
# import pandas as pd
# import numpy as np
import json
import time
import datetime
import sys

# twitter keys
consumer_key = 'YOUR CONSUMER KEY'
consumer_secret = 'YOUR CONSUMER SECRET'
access_key = 'YOUR ACCESS KEY'
access_secret = 'YOUR ACCESS SECRET'


'''
Collecting tweet data through streaming API is quite different from doing so in REST API. 
The streaming API is used to passively obtain incoming tweets data in real time, 
whereas the REST API is used to actively request to obtain data in past one week. 
Therefore, the streaming API is very useful for obtaining a high volume of tweets in real-time/ for real-time tweet analysis.

# https://docs.tweepy.org/en/v3.10.0/streaming_how_to.html
# https://developer.twitter.com/en/docs/tutorials/consuming-streaming-data
The streaming api is quite different from the REST api because the REST api is used to pull data from twitter 
but the streaming api pushes messages to a persistent session. 
This allows the streaming api to download more data in real time than could be done using the REST API.

Tweepy implements the StreamListener class. 
Thereby when using the streaming API , users only need to inherit this class and override its on_status function 
to implement the a Stream object and connect to the Twitter API. 
The on_status method will be called when pulling the new data.

'''



# override tweepy.StreamListener to add logic to on_status
# This function creates a class inheriting from Tweepy StreamListener
# This function is partly from Risser (2020): https://towardsdatascience.com/how-to-create-a-dataset-with-twitter-and-cloud-computing-fcd82837d313
# This function is partly from Taskinoor (2015): https://stackoverflow.com/questions/27900451/convert-tweepy-status-object-into-json
# This function is partly from AWS (2021): https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/firehose.html#Firehose.Client.put_record
# This function is partly from Tweepy (2021): https://docs.tweepy.org/en/latest/extended_tweets.html
class TwitterStreamListener(StreamListener):

    # override on_data function
    def on_data(self, data):
        #print(status.text)
        
        # load Json data into python dictionary
        tweet = json.loads(data)
        
        try:            
            # !!!! If status is a Retweet, it will not have an extended_tweet attribute, and status.text could be truncated.
            # get tweet full text
            if 'retweeted_status' in tweet.keys():
                # retweeted and extended
                if 'extended_tweet' in tweet['retweeted_status']:
                    tweet_text = tweet['retweeted_status']['extended_tweet']['full_text']
                    is_retweet = True
                    ori_tweet_id = tweet['retweeted_status']['id']
                    ori_user_id = tweet['retweeted_status']['user']['id']
                # retweetd but not extended
                else:
                    tweet_text = tweet['retweeted_status']['text']
                    is_retweet = True
                    ori_tweet_id = tweet['retweeted_status']['id']
                    ori_user_id = tweet['retweeted_status']['user']['id']
            # not retweet
            else:
                # extended tweet
                if 'extended_tweet' in tweet.keys():
                    tweet_text = tweet['extended_tweet']['full_text']
                    is_retweet = False
                    ori_tweet_id = 'ORIGINAL'
                    ori_user_id = 'ORIGINAL'
                # normal tweet    
                else:
                    tweet_text = tweet['text']
                    is_retweet = False
                    ori_tweet_id = 'ORIGINAL'
                    ori_user_id = 'ORIGINAL'
                    
            # get quoted info
            if tweet['is_quote_status']:
                is_quoted = True
                quoted_id = tweet['quoted_status']['id']
                quoted_author_id = tweet['quoted_status']['user']['id']                
            else:
                is_quoted = False
                quoted_id = 'ORIGINAL'
                quoted_author_id = 'ORIGINAL'
                
            # get reply info
            is_reply = str(bool(tweet['in_reply_to_status_id']))
            reply_to_id = str(tweet['in_reply_to_status_id'])
                        
            # data record
            # keep every field in str, expected str instance
            record_lst = [str(tweet['id']),
                          #str(tweet['user']['screen_name']),
                          str(tweet['user']['id']),
                          str(tweet['user']['location']),
                          str(tweet['user']['followers_count']),
                          str(tweet['created_at']),
                          str(is_retweet),
                          str(ori_tweet_id),
                          str(ori_user_id),
                          tweet_text.replace('\n',' ').replace('\r',' '),
                          str(tweet['favorite_count']),
                          str(tweet['retweet_count']), #reply_count and quote_count requires premium version
                          str(is_quoted),
                          str(quoted_id),
                          str(quoted_author_id),
                          is_reply,
                          reply_to_id,
                          '\n', # AWS Firehose will decode the records in each file based on the character ('\n')
                         ]
            
            # formating message 
            record = '\t'.join(record_lst)
            print(record)
            
            # deliver message to the S3 through the Firehose delivery streams
            client.put_record(
                DeliveryStreamName = 'tweepystreaming',
                Record={
                    'Data': record
                }
            )            
                       
        except (AttributeError, Exception) as e:
            print (e)
        
        return True
        
    def on_error(self,status):
        print (status)


        
    

# This fucntion is partly from Tweepy (2021): https://docs.tweepy.org/en/latest/stream.html#tweepy.Stream.filter
# This fucntion is partly from Tweepy (2021): https://docs.tweepy.org/en/latest/streaming.html
# This fucntion is partly from Tweepy (2021): https://docs.tweepy.org/en/latest/stream.html
if __name__ == '__main__':
    # create overode StreamListener class 
    Tweet_listener = TwitterStreamListener()
    
    # Oauth 1.0 get authenication
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    # access auth
    auth.set_access_token(access_key, access_secret)
    # connect to twitter api
    api = tweepy.API(auth)
    
    # Verify connection
    # This function is from Tweepy (2021): https://docs.tweepy.org/en/latest/auth_tutorial.html
    try:
        redirect_url = auth.get_authorization_url()
    except tweepy.TweepError:
        print('Error! Failed to get request token.')
    
    
    # set Firehose data delivery stream
    client = boto3.client('firehose',
                          region_name = 'us-east-1',
                          aws_access_key_id = 'YOUR_AWS_ACCESS_KEY',
                          aws_secret_access_key = 'YOUR_AWS_ACCESS_KEY_SECRET'
                          ) 
    # Kinesis name
    
    # terms to be tracked, A OR B
    terms = ['Huawei','P50']
    
    while True:
        try:
            print('Twitter streaming...')
            # create tweet streaming
            tweet_stream = tweepy.Stream(auth = api.auth, listener = Tweet_listener, tweet_mode='extended')
            # set streaming filter
            # track for all tweets containing words in terms list (OR operation)
            tweet_stream.filter(track=terms, languages=['en'], stall_warnings=True)

        except Exception as e:
            print(e)
            print('Disconnected...')
            time.sleep(5)
            continue   




