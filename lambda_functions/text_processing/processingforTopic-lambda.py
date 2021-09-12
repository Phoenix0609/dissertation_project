# S3 resource
import boto3

# NLP things
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import gensim

# import others
import pandas as pd
import numpy as np

import io
from io import StringIO
import string
import re

import os
import json
import time
from datetime import datetime, timedelta
import sys
import urllib.parse
import csv


print('Loading function')

#downloading nltk files
#store in Lambda's ephemeral storage at the location /tmp
nltk.data.path.append("/tmp")
nltk.download("punkt", download_dir = "/tmp")
nltk.download("stopwords", download_dir = "/tmp")

##################################################
#     Function for retriving and restoring data
##################################################
# This function retrives tweet files created in the last hour and merge them together into a dataframe
# This function is partly from Harrison (2021): https://gist.github.com/onelharrison/20689c9d11da48e0bd28f40d50fbb194
# This function is partly from Garnaat (2015): https://stackoverflow.com/questions/30249069/listing-contents-of-a-bucket-with-boto3
# This function is partly from Selah (2015): https://stackoverflow.com/questions/18016037/pandas-parsererror-eof-character-when-reading-multiple-csv-files-to-hdf5
def retrive_tweet_files(used_time, bucket_name):
    # create bucket resource
    bucket = s3_resource.Bucket(bucket_name)
    
    # search for files created in the last hour (with certain prefix)
    files = []
    for bucket_object in bucket.objects.filter(Prefix = used_time):
        files.append(bucket_object.key)
    
    # check if there is sufficient data
    if len(files) == 0:
        print("NO SUFFICIENT DATA")
        return pd.DataFrame()
        
    # getting the files contents
    t = 0
    for file_key in files:
        # get individual response
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        print(f"Status - {status}") # status == 200 means success
        
        # convert to pandas dataframes and concat them together  
        if t == 0:
            df = pd.read_csv(response.get("Body"), dtype='str', sep='\t', header=None)    
        else:
            # using quoting=csv.QUOTE_NONE to ignore single quote mark in a string
            # jump bad lines
            next_df = pd.read_csv(response.get("Body"), dtype='str', sep='\t', header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False)
            df = pd.concat([df, next_df], axis = 0, join = 'outer')
        t += 1
        
    return df
    
# Function is for fliterring irrlevant contents and only  keep tweets containing 'huawei' AND 'p50'
# Despite that tweets have been filtered in the streaming stage, a extra filter is set here to ensure accuracy
# This function is from Flyingmeatball (2016): https://stackoverflow.com/questions/37011734/pandas-dataframe-str-contains-and-operation
# Notice that the word pair should be exactly two words
def tweets_filter(data, words_pair):
    # case to false means ignoring case
    data = data[(data['text'].str.contains(words_pair[0], case=False)) & (data['text'].str.contains(words_pair[1], case=False))]
    return data

# Funtion for restoring processed data to the S3
# This function is partly from Stefan (2016): https://stackoverflow.com/questions/38154040/save-dataframe-to-csv-directly-to-s3-python
def save_processed_tweets(df, dest_bucket, last_hour):
    # Create s3 client resource
    s3_client = boto3.client('s3')
    
    # get current time
    # different format compared with when reading the data
    # so redefine now_time and used_time
    now_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    used_time = last_hour.strftime('%Y%m%d_%H_oclock')
        
    # convert df to csv
    #from io import StringIO
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
        
    # put it to the destination bucket
    response = s3_client.put_object(Bucket=dest_bucket,
                                    Key="data_of_{}_created_at_{}.csv".format(used_time,now_time),
                                    Body=csv_buffer.getvalue())
    # check if it is done
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if status == 200:
        print(f"Successful S3 put_object response. Status - {status}")
    else:
        print(f"Unsuccessful S3 put_object response. Status - {status}")

#########################################
##  TEXT CLEANING FUNCTIONS
#########################################

# Function for deleting emoji
# This function is from Adam (2018): https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
def deleteEmojis(text):    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U00010000-\U0010ffff"
                              "]+", flags=re.UNICODE)    
    return emoji_pattern.sub(r' ',text)

# Function for deleting default tags or labels in the tweets like 'VIDEO:' and 'AUDIO:'
# This function is partly from Bica (2010): https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
def remove_tweet_marks(tweet):
    tweet = re.sub('VIDEO:', '', tweet)  # remove 'VIDEO:' from start of tweet
    tweet = re.sub('AUDIO:', '', tweet)  # remove 'AUDIO:' from start of tweet
    tweet = re.sub('&amp', '', tweet)
    tweet = re.sub('RT @', '', tweet) # keep one space
    return tweet

# Function for expanding contractions
# This function is from Dubois (2017): https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# Function for tweet text cleaning
# This function is partly from Zx81 (2014): https://stackoverflow.com/questions/24399820/expression-to-remove-url-links-from-twitter-tweet/24399874
# This function is partly from Oneporter (2014): https://stackoverflow.com/questions/817122/delete-digits-in-python-regex
def text_cleaning(data):
    # delete emoji
    data = data.map(lambda text: deleteEmojis(text))
    # deleting the URL
    data = data.map(lambda text: re.sub(r"http\S+", "", text))
    # deleting 'VIDEO' and 'AUDIO'
    data = data.map(lambda text: remove_tweet_marks(text))
    # convert the relevant column to lowercase
    data = data.str.lower()
    # expending contractions
    data = data.map(lambda text: decontracted(text))
    # delete punctuations
    data = data.map(lambda text: re.sub(r'[,\.!?:;@#&*$¥+~•₹€£=—\-\–\\→\⇢\<\>\|\“\”\’\{\}\'\"\`\[\]\(\)_\-\%\/]', ' ', text))
    # remove all single characters
    data = data.map(lambda text: re.sub(r'\s+[a-zA-Z]\s+', ' ', text))
    # remove digits
    # notice that in this case some product names or terms may contain numbers, e.g. "P50","4g","5g"
    # Thus only remove those digits that are not part of another word
    data = data.map(lambda text: re.sub(r'\b\d+\b', ' ', text))
    # deleting surplus spacings
    data = data.map(lambda text:  re.sub(r'\s+', ' ', text))
    
    # 删除过短的记录 delete short sentence?? # 可以结合word token，计算长度？
    return data

#########################################
##  TEXT PROCESSING FUNCTIONS
#########################################

# This function is for tokenising sentences in the corpus
def tokenising_corpus(data):
    # Transform df into list
    words = data.tolist()
    # tokenising each sentence
    word_tokens = []
    for tweet in words:
        word_tokens.append(word_tokenize(tweet))
        
    return word_tokens

# This function allows user to remove stopwords
# and also allow to specify and remove some irrelevant words in this case (such sentiments) for tuning the model
# This will only apply to the first LDA model for get more clear topics
def custom_words_remover(word_lst, text_tokens):
    # create a new list with specified words removed 
    processed_tokens = []
    for token in text_tokens:
        processed_tokens.append([w for w in token if not w in word_lst])
        
    return processed_tokens

# This function is for stemming the words in the corpus
def stemming_words(text_tokens):
    #from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    # stemming the tokens
    stemmed = []
    for token in text_tokens:
        stemmed.append([ps.stem(word) for word in token])
    
    return stemmed
    
# Function for making biagram
# This funciton is from Prabhakaran (2018): https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
def make_bigrams(data,min_count,thres):
    # Build the bigram model with min_count=10
    # higher threshold fewer phrases.
    bigram = gensim.models.Phrases(data, min_count=min_count, threshold=thres)
    
    # Faster way to get a sentence clubbed as a bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    
    return [bigram_mod[doc] for doc in data]
    

#########################################
##             EXECUTION
#########################################    

# Create a S3 resource and client objects using boto3
s3_resource = boto3.resource('s3') # for getting bucket info
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    
    # getting bucket info
    bucket_name = "rawtweet"
    #bucket = s3_resource.Bucket(bucket_name)
    
    # calculate time
    now_time = datetime.now().strftime('%Y/%m/%d/%H')
    last_hour = datetime.now() - timedelta(hours = 1)
    used_time = last_hour.strftime('%Y/%m/%d/%H')
    
    try:
        #  Retrive tweet files created in the last hour and merge them together into a dataframe
        df = retrive_tweet_files(used_time, bucket_name)
        # if no sufficient data, break the lambda function
        if len(df) == 0:
            return "NO SUFFICIENT DATA"
            
        '''
        “errorMessage”: Task timed out after 3.00 seconds
        need to increase task time in the configuration
        '''
            
        # infering the schema
        df.columns = ['tweet_id',
                    'user_id',
                    'location',
                    'user_followers_count',
                    'time',
                    'is_retweet',
                    'ori_tweet_id',
                    'ori_user_id',
                    'text',
                    'favorite_count',
                    'retweet_count',
                    'is_quoted',
                    'quoted_id',
                    'quoted_author_id',
                    'is_reply',
                    'reply_to_id',
                    'NAN']
                    
        # Drop the NAN row
        df.drop('NAN', axis=1, inplace=True)
        
        # Delete duplications with the same tweet id
        df = df.drop_duplicates('tweet_id')
        
        # Delete NA in review id and text
        df = df.dropna(axis=0, subset=['tweet_id','text'])
        
        # filterring tweets based on keywords
        # the filtering should be performed after dropping NA
        # set word pair
        words_pair = ['huawei','p50']
        # keeping tweets containing only 'huawei' AND 'p50'
        
        df = tweets_filter(df, words_pair)
        
        # reset the dataframe index
        df = df.reset_index()
        # drop old index column
        df = df.drop(['index'], axis=1)
        
        # check if the df is null
        # if no sufficient data, break the lambda function
        if len(df) == 0:
            return "NO SUFFICIENT DATA"
        
        #########################################
        ##  BASIC CLEANING AND TEXT PROCESSING
        #########################################
        # processing time
        # define date format
        DateFormat = '%a %b %d %H:%M:%S %z %Y'
        # apply the dateformat and set errors to NA
        df['time'] = pd.to_datetime(df['time'], format=DateFormat, errors = 'coerce')

        df['date'] = df['time'].dt.date
        df['hour'] = df['time'].dt.hour        
        
        
        # text cleaning
        df['text'] = text_cleaning(df['text'])
        # Tokenising sentences in the corpus
        word_tokens = tokenising_corpus(df['text'])
        # keep the copies
        df['text_tokens'] = word_tokens
        # get the length of each text
        df['text_len'] = df['text_tokens'].map(lambda x: len(x))
        
        ###################################
        #      Deleting stop words
        ###################################

        # create a list of stopwords
        stops = set(stopwords.words("english"))
        # remove stopwords
        filtered_tokens = custom_words_remover(stops, word_tokens)
        
        # Tweets are filtered based on 'huawei' and 'p50', 
        # which means that every text in the current data is about 'huawei p50' and contains'huawei' and'p50'. 
        # Therefore, all close expressions of huawei p50 can be deleted to improve the accuracy
        # Add p50 words as new stopword list
        newstopwords = ['huawei','p50','huaweip50','huaweip50pro','p50pro','pro','huaweip50series','series']
        # delete new added stopwords from the text tokens
        filtered_tokens = custom_words_remover(newstopwords, filtered_tokens)

        ###################################
        #           Stemming
        ###################################
        # Use defined function to stem the words
        stemmed = stemming_words(filtered_tokens)

        df['processed_tokens'] = stemmed

        # Phrase Modeling: Making Bigrams
        # Build the bigram model with min_count=10
        # higher threshold fewer phrases
        df['processed_tokens'] = make_bigrams(data= df['processed_tokens'],min_count= 10,thres= 100)
        
        
        ###########################################
        ##  PUT TO BUCKET FOR TOPIC MODELLING
        ###########################################
        
        # destination bucket
        dest_bucket = 'data-ready-for-topic-modelling' 
        save_processed_tweets(df, dest_bucket, last_hour)
        
                
        return T
        
    except Exception as e:
        print(e)
        #print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(file_key, bucket_name))
        #raise e        
        
        

    








