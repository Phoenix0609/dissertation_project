# AWS resource
import boto3

# Mysql support
import pymysql
import pymysql.cursors

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
import pickle
import sys
import urllib.parse
import csv

print('Loading function')

##################################################
#     Function for retriving data
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
        t += 1
    return df

#########################################################
#     Function for writing data to the rds MySQL
#########################################################
# This function is for writing the data records from the dataframe to aws mysql database one by one
# This function is from Doe (2019): https://stackoverflow.com/questions/58232218/how-to-insert-a-pandas-dataframe-into-mysql-using-pymysql
def write_to_rds_mysql(data):
    # build connection to rds mysql
    try:
        connection = pymysql.connect(host=RDS_HOST,
                                     user=USER,
                                     passwd=PASSWORD,
                                     db=DB_NAME,
                                     connect_timeout=10,
                                     charset='utf8mb4')
    except:
        print('Fail to connect RDS')
        sys.exit()
    # create cursor
    cursor = connection.cursor()
    # creating column list for insertion
    cols = "`,`".join([str(i) for i in data.columns.tolist()])
    # insert df records one by one
    for i,row in data.iterrows():
        # SQL insert command for excution
        sql = "INSERT INTO `tweets` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
        try:
            # excute the insert command
            cursor.execute(sql, tuple(row))
            # commit changes and updates through database connection
            connection.commit()
            print('Successfully write records:    ', tuple(row))
        except Exception as e:
            # print error info
            print("--ERROR--", e)
            continue
    # close connection to the database    
    connection.close()



#########################################
##             EXECUTION
#########################################

# Setting RDS info globally
REGION = 'us-east-1'
RDS_HOST = "database-1.ct58zutwy87d.us-east-1.rds.amazonaws.com"
USER = "admin"
PASSWORD = "zfx199869"
DB_NAME = "twitter"
PORT = 3306

# Create a S3 resource and client objects using boto3
s3_resource = boto3.resource('s3') # for getting bucket info
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    
    # read data
    # getting bucket info
    bucket_name_topic = 'data-labelled-topics'
    bucket_name_senti = 'data-labelled-sentiments'
    # get time info
    now_time = datetime.now().strftime('%Y/%m/%d/%H')
    last_hour = datetime.now() - timedelta(hours = 1)
    used_time_topic = last_hour.strftime('topic_modelling_data_of_%Y%m%d_%H_oclock')
    used_time_senti = last_hour.strftime('sentiment_analysis_data_of_%Y%m%d_%H_oclock')
    
    
    try:
        # Seperately retrive data labelled topics and sentiments created in 15 mins ago 
        # Notice that header is in the dataframe
        df_topic = retrive_tweet_files(used_time_topic, bucket_name_topic)
        df_senti = retrive_tweet_files(used_time_senti, bucket_name_senti)
        # if no sufficient data, break the lambda function
        if len(df_topic) == 0:
            print("NO SUFFICIENT TOPIC DATA")
            return "NO SUFFICIENT DATA"
        if len(df_senti) == 0:
            print("NO SUFFICIENT SENTIMENT DATA")
            return "NO SUFFICIENT DATA"
        
        # select id and label columns from df_senti
        df_senti = df_senti[['tweet_id','polarity','subjectivity']]
        # merge the topic data and sentiment data
        df = df_topic.merge(df_senti, how='left', on='tweet_id')
        
        # remove irrelevant columns
        df.drop(['Unnamed: 0','text_tokens','processed_tokens','time'], axis=1, inplace=True)
        
        # As all these data are well processed, there is no need for deleting NA in id or text and duplications
        # the Mysql DB does not have NaN type, thus need to convert all NaN in the df to None
        # this function is from Hayden (2013): https://stackoverflow.com/questions/14162723/replacing-pandas-or-numpy-nan-with-a-none-to-use-with-mysqldb
        df = df.astype(object).where(pd.notnull(df), None)
        
        # rename columns to fit the database scheme
        df.rename(columns={"text":"tweet_text"}, inplace=True)
        
        
        #########################################################
        ###        Writing data to the rds MySQL
        #########################################################
        # inserting data records to RDS one by one
        write_to_rds_mysql(df)
        
        '''
        df = df.astype(object).where(pd.notnull(df), None)
        for i,row in df.iterrows():
            print(tuple(row))
        '''
        
        
        # os.listdir(download_path)
        # guided_lda.print_topics(num_words=10)
        # df.columns.values
        return True
    
    except Exception as e:
        print(e)
        #print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(file_key, bucket_name))
        #raise e        
        
