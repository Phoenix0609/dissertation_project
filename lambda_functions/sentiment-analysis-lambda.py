import boto3

# sentiment analysis toolkits
from textblob import TextBlob
import pickle

# import others
import pandas as pd
import numpy as np

import io
from io import StringIO
import string

import os
import json
import time
from datetime import datetime, timedelta
import sys
import urllib.parse
import csv

print('Loading function')

##################################################
#     Function for retriving and restoring data
##################################################
# This function retrives tweet files created in the last hour and merge them together into a dataframe
# This function is partly from Harrison (2021): https://gist.github.com/onelharrison/20689c9d11da48e0bd28f40d50fbb194
# This function is partly from Garnaat (2015): https://stackoverflow.com/questions/30249069/listing-contents-of-a-bucket-with-boto3
# This function is partly from Selah (2015): https://stackoverflow.com/questions/18016037/pandas-parsererror-eof-character-when-reading-multiple-csv-files-to-hdf5
# Notice that header is in the dataframe, and also no need for sep='\t'
# Already remove duplications, so no needs for keeping dtype='str' (which is for keeping tweet_id from being contracted)
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
                df = pd.read_csv(response.get("Body"))
        else:
            # do not set quoting here, since the data like processed words lists are quoted with ""
            # jump bad lines
            next_df = pd.read_csv(response.get("Body"), error_bad_lines=False)
            df = pd.concat([df, next_df], axis = 0, join = 'outer')
        t + 1
    return df
    
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
                                    Key="sentiment_analysis_data_of_{}_processed_at_{}.csv".format(used_time,now_time),
                                    Body=csv_buffer.getvalue())
    # check if it is done
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if status == 200:
        print(f"Successful S3 put_object response. Status - {status}")
    else:
        print(f"Unsuccessful S3 put_object response. Status - {status}")
    
        
##########################################################
#     Function for sentiment analysis
########################################################## 
# This function is for mapping the sentimental polarity and subjectivity scores of tweets 
# into a range from -5 to +5, with a step of 1
# Faster way using np to process numbers
def scoring(a):
    # faster way using np
    s = np.where(a >= 0.8, 5,
                 np.where(a >= 0.6, 4,
                          np.where(a >= 0.4, 3,
                                   np.where(a >= 0.2, 2, 
                                            np.where(a > 0, 1, 
                                                     np.where(a == 0, 0, 
                                                              np.where(a >= -0.2, -1, 
                                                                       np.where(a >= -0.4, -2, 
                                                                                np.where(a >= -0.6, -3, 
                                                                                         np.where(a >= -0.8, -4, -5))))))))))
    return s
    
# This function is for getting sentimental polarity and subjectivity scores of tweets
def get_senti_subj_scores(data):
    # get sentiment score
    data['polarity'] = data['text'].apply(lambda text: scoring(TextBlob(str(text)).sentiment.polarity))
    # get subjectivity score
    data['subjectivity'] = data['text'].apply(lambda text: scoring(TextBlob(str(text)).sentiment.subjectivity))
    return data

#########################################
##             EXECUTION
#########################################    

# Create a S3 resource and client objects using boto3
s3_resource = boto3.resource('s3') # for getting bucket info
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    
    # read data
    # getting bucket info
    bucket_name = "data-ready-for-sentiment-analysis"
    
    # calculate time
    now_time = datetime.now().strftime('%Y/%m/%d/%H')
    last_hour = datetime.now() - timedelta(hours = 1)
    used_time = last_hour.strftime('data_of_%Y%m%d_%H_oclock')
    
    
    try:
        # Retrive processed data files created in 15 mins ago and merge them together into a dataframe
        # Notice that header is in the dataframe
        df = retrive_tweet_files(used_time, bucket_name)
        # if no sufficient data, break the lambda function
        if len(df) == 0:
            return "NO SUFFICIENT DATA"
        
        '''
        “errorMessage”: Task timed out after 3.00 seconds
        need to increase task time in the configuration
        '''
        # drop irrelevant contents
        df.drop('Unnamed: 0',axis=1, inplace = True)
        
        
        #########################################################
        ###     Sentiment analysis
        #########################################################

        # get sentimental polarity and subjectivity scores of tweets
        df = get_senti_subj_scores(df)
        
        
        
        ###########################################
        ##  SAVE TOPIC RESULTS
        ###########################################
        
        # destination bucket
        dest_bucket = 'data-labelled-sentiments' 
        save_processed_tweets(df, dest_bucket, last_hour)
        
        
        
        # os.listdir(download_path)
        # guided_lda.print_topics(num_words=10)
        # df.columns.values
        return True
    
    except Exception as e:
        print(e)
        #print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(file_key, bucket_name))
        #raise e        
        
