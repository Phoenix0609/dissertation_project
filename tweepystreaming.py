#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install tweepy


# In[3]:


# Define IAM role
import boto3
import re
# from sagemaker import get_execution_role
# role = get_execution_role()
# import sagemaker
# from sagemaker.predictor import csv_serializer
#   import tweepy things
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
# import others
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import json
import time
import datetime
import sys


# In[4]:


# twitter keys
consumer_key = 'KrZINiWKxyGH540GC5QDfRvhS'
consumer_secret = 'TV92ydK18ry71hbdENEIbauKMglgMwf71bZqLsf8knN45yHzKt'

access_key = '1323281849364480006-pJQvWemDB7pdIn2eOqWd6gymIQwA4N'
access_secret = 'x2Q8SeEuR7lOnYKl64MnRFKMmktgVEPjfG20aJJdLgMoC'


# In[5]:


#OAuth 2身份验证 创建一个AppAuthHandler实例  仅公共信息的只读访问权限
#https://docs.tweepy.org/en/latest/auth_tutorial.html

# Oauth 1.0
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)


# In[6]:


# access auth
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


# In[7]:


# https://docs.tweepy.org/en/latest/auth_tutorial.html
# 验证链接
try:
    redirect_url = auth.get_authorization_url()
except tweepy.TweepError:
    print('Error! Failed to get request token.')


# In[ ]:


# https://www.cnblogs.com/zhuminghui/p/12918272.html
# tweepy 中文，搜stream

'''
stream 和 Rest api 不同，前者是长链接被动获取实时获取数据，后者则是主动请求以获得数据。
tweepy 实现了 StreamListener 类，我们想要实现 stream 的功能只要继承这个类并重写它的 on_status 方法即可，on_status 方法会在得到数据时被调用

https://developer.twitter.com/en/docs/tutorials/consuming-streaming-data

'''


# In[ ]:


# extended 的处理 要看
# https://docs.tweepy.org/en/latest/extended_tweets.html

# 目前是没有一个retweet的
# 搜不到 retweet,搜不到python的 


# In[8]:


# 创建类
# https://docs.tweepy.org/en/v3.10.0/streaming_how_to.html#a-few-more-pointers

#override tweepy.StreamListener to add logic to on_status

# Convert Tweepy Status to JSON
# https://stackoverflow.com/questions/27900451/convert-tweepy-status-object-into-json

class TwitterStreamListener(StreamListener):

    # 注意只能 on_data, on_status 出来的是status 会报错
    # status 是一个对象，里面包含了该条推文的所有字段，比如推文内容、点赞数、评论数、作者id、作者昵称、作者粉丝数等等
    
    # override on_data function
    def on_data(self, data):
        #print(status.text)
        
        # https://www.geeksforgeeks.org/json-loads-in-python/
        # load Json data into python dictionary
        
        tweet = json.loads(data)
        
        #'extended tweets'
        # keys() returns keys in a dictiorary
        
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
                          '\n', # 会基于换行符 ('\n') 分析每个文件中的记录
                         ]
            
            # formating message 
            record = '\t'.join(record_lst)
            print(record)
            
            # deliver message to the S3 through the Firehose delivery streams
            # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/firehose.html#Firehose.Client.put_record
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


# In[10]:


if __name__ == '__main__':
    # 创建流
    Tweet_listener = TwitterStreamListener()

    # https://docs.tweepy.org/en/latest/stream.html
    # https://docs.tweepy.org/en/latest/streaming.html
    # Stream 类 
    # https://docs.tweepy.org/en/latest/stream.html#tweepy.Stream
    # filter的用法
    # https://docs.tweepy.org/en/latest/stream.html#tweepy.Stream.filter

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    
    
    # set Firehose data delivery stream
    client = boto3.client('firehose',
                          region_name = 'us-east-1',
                          aws_access_key_id = 'AKIA426II2GSWONAECM6',
                          aws_secret_access_key = '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'
                          ) # 所以必须加session_token？
    # Kinesis name
    # delivery_stream = 'tweepystreaming'
    
    # 启动stream  --- 支持异步，参数is_async，推荐使用异步形式
    
    #tweet_stream.filter(track=terms, is_async=True)
    
    terms = ['Huawei','P50','HarmonyOS']
    # 这样搜会有很多不属于华为P50的手机型号，比如P40什么的
    
    while True:
        try:
            print('Twitter streaming...')
            # 身份验证，绑定监听流媒体
            tweet_stream = tweepy.Stream(auth = api.auth, listener = Tweet_listener, tweet_mode='extended')
            # 应该是先listen 然后再filter 所以一开始的是杂乱的
            
            # 确定track的规则 想track 包含 python或Java的推文 而不是两个都包含的
            tweet_stream.filter(track=terms, languages=['en'], stall_warnings=True)
            #track 后面必须加list, list里面包含关键词
            #包含两个关键词呢 track 里track的本来就是 A或B
            #https://docs.tweepy.org/en/latest/stream.html#tweepy.Stream.filter
            
        except Exception as e:
            print(e)
            print('Disconnected...')
            time.sleep(5)
            continue   


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




