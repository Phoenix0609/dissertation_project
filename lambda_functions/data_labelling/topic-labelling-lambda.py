import boto3

# NLP things
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

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
import pickle
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
        t += 1
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
                                    Key="topic_modelling_data_of_{}_processed_at_{}.csv".format(used_time,now_time),
                                    Body=csv_buffer.getvalue())
    # check if it is done
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if status == 200:
        print(f"Successful S3 put_object response. Status - {status}")
    else:
        print(f"Unsuccessful S3 put_object response. Status - {status}")
    
##########################################################
#     Functions for loading the production model
##########################################################    
# Function for download files including the model component list and eat matrix
# This function is from Kindjacket (2019): https://stackoverflow.com/questions/48964181/how-to-load-a-pickle-file-from-s3-to-use-in-aws-lambda
def file_download_helper(file_key, bucket_name):
    s3_resource = boto3.resource('s3')
    # load pkl file
    file = pickle.loads(s3_resource.Bucket(bucket_name).Object(file_key).get()['Body'].read())
    print('Successfully load: ',file_key)
    return file
    
# Function for downloading model to S3
# This function is from Sophros (2020): https://stackoverflow.com/questions/61638940/save-a-gensim-lda-model-to-s3
def model_download_helper(download_path, file_lst, bucket_name):
    # create s3 resource
    s3_resource = boto3.resource('s3')
    # download model components
    for file_name in file_lst:
        full_path = download_path + file_name
        s3_resource.Bucket(bucket_name).download_file(file_name, full_path)
        print('Successfully download: ', file_name)
        
##########################################################
#     Function for topic modelling
########################################################## 
# This function update the dictionary with words in new corpus
# and also calculate original and extended length for update eta matrix
# The id of the original word will remain the same, 
# and new words will get ids starting from the last number in the original dictionary
def dictionary_updater(dictionary, data):
    # get original length
    ori_len = len(dictionary.token2id)
    # extend the vocabulary 
    dictionary.add_documents(data)
    # calculate extended length
    new_len = len(dictionary.token2id)
    
    return dictionary, ori_len, new_len

# This functions is for creating the documents-topic matrix
# which can show the individual document's probabilities for each topic
# This function is from Wang (2019): https://stackoverflow.com/questions/56408849/after-applying-gensim-lda-topic-modeling-how-to-get-documents-with-highest-prob
def create_doc_topic_matrix(model, corpus, num_topics):
    # Create a dictionary, with topic ID as the key, and the value is a list of tuples (docID, probability of this particular topic for the doc) 
    topic_dict = {i: [] for i in range(num_topics)}
    
    # Remember to set the minimum_probability=0 in the model or can't get probabilities of one under each topic
    # Loop over all the documents to group the probability of each topic
    for doc_id in range(len(corpus)):
        # generate topic vectors for unseen documents
        topic_vector = model[corpus[doc_id]]
        for topic_id, prob in topic_vector: 
            topic_dict[topic_id].append(prob)
    
    # Create documents-topic matrix
    doc_topic = pd.DataFrame.from_dict(topic_dict)
    
    return doc_topic
    
# This function is for getting most prevalent topics of each document in the doc_topic_matrix
# This function is partly from Alexander (2015): https://stackoverflow.com/questions/34518634/finding-highest-values-in-each-row-in-a-data-frame-for-python
def get_most_prevalent_topics(data, doc_topic_matrix,top_n_topics,threshold):
    
    # replace all values in the matrix with a probability less than threshold
    # threshold is supposed to be (1/num_topics) * num_top_topics_selected
    doc_topic_matrix[doc_topic_matrix < threshold] = np.nan
    # get the top n topics of each document
    # if the topics value is less than the threshold, just return NA
    top_topics = doc_topic_matrix.apply(lambda s: pd.Series(s.nlargest(top_n_topics).index), axis=1)
    
    # get number of columns in the top topic matrix
    topic_len = len(top_topics.columns)
    # check if number of columns in the top topic matrix equals to the input top topic numbers
    # if not, expend the matrix and fill with NaN
    if topic_len != top_n_topics:
        for t in range(topic_len, top_n_topics):
            # add new columns, Note the new column name is INT type
            top_topics[t] = np.nan
            
    # generate column names
    name_lst = []
    for i in range(1, top_n_topics+1):
        name_lst.append('top_{}'.format(i))
    # rename columns
    top_topics.columns = name_lst

    # Convert the float to string 
    # use 'Int64' to ignore NA and decimal
    top_topics = top_topics.astype('Int64').astype('str')
    
    # concat the top_topics to the main dataframe
    joined_df = pd.concat([data, top_topics], axis = 1, join = 'outer')
    
    return joined_df    
    
    

#########################################
##             EXECUTION
#########################################    

# Create a S3 resource and client objects using boto3
s3_resource = boto3.resource('s3') # for getting bucket info
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    
    # read data
    # getting bucket info
    bucket_name = "data-ready-for-topic-modelling"
    
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
        
        # note that df['processed_tokens'] is now read as a string rather than a list object
        # so need to convert string to list object
        df['processed_tokens'] = df['processed_tokens'].apply(lambda lst: lst.strip("[]").replace("'","").split(", "))
        
        
        ##############################################################
        ###      Download model component list
        ##############################################################
        
        bucket_name = "guided-lda-on-pre-collected-data"
        model_component_list_key = "components.pkl"
        components = file_download_helper(model_component_list_key, bucket_name)
        
        
        ##############################################################
        ###      Download production guided lda model and dictionary
        ##############################################################
        bucket_name = "guided-lda-on-pre-collected-data"
        # '/tmp' can only read files, for writing operations, use '/tmp/'
        # This function is partly from Joonas (2017): https://stackoverflow.com/questions/39383465/python-read-only-file-system-error-with-s3-and-lambda-when-opening-a-file-for-re
        download_path = '/tmp/'
        # download the model (with all its components) to the local folder (tmp)
        model_download_helper(download_path, components, bucket_name)
        # load the production model 
        guided_lda = gensim.models.LdaMulticore.load(download_path + 'guided_lda_for_production.model')
        
        ##########################################################
        #       Updating word dictionary
        ##########################################################
        # get the pre-generated dictionary
        id2word = guided_lda.id2word
        
        # Update the dictionary with words in new corpus
        # The id of the original word will remain the same, 
        # and new words will get ids starting from the last number in the original dictionary
        #id2word, ori_len, new_len = dictionary_updater(dictionary= id2word, data= df['processed_tokens'])
        
        '''
        https://stackoverflow.com/questions/22196248/gensim-lda-model-calling-update-on-a-corpus-with-unseen-words
        必须使用相同的字典（单词及其整数 id 之间的映射）进行训练、更新和推理。，因此就算是update也没有用
        you must use the same dictionary (mapping between words and their integer ids) for both training, UPDATES and inference.
        Which means you can update the model with new documents, but not with new word types.
        new words will be added to the model in the continuous training part
        '''
        
        #########################################################
        ###     Create bag of words (bow) corpus
        #########################################################

        # Create corpus that contains all documents
        texts = df['processed_tokens']

        # Create bag of word for each document in the corpus 
        # each bow contains the id of each word in that single document and its number of occurrences in that document 
        # (term id, term document Frequency)
        # NOTE: the corpus will only contain word_ids that are in the dictionary. 
        # As the dictionary was not updated with new words, the corpus will not contain new words
        corpus = [id2word.doc2bow(text) for text in texts]

        # get the number of topics of the model
        num_topics = len(guided_lda.get_topics()[:,0])
        
        #########################################################
        ###     Get the topics of each tweets
        #########################################################
        
        
        # Creating the documents-topic matrix
        # which can show the individual document's probabilities for each topic
        doc_topic = create_doc_topic_matrix(model = guided_lda,
                                            corpus = corpus,
                                            num_topics = num_topics)
        #print(doc_topic.head())
        
        
        # getting most prevalent topics of each document in the doc_topic_matrix
        # threshold is supposed to be (1/num_topics) * num_top_topics_selected
        df = get_most_prevalent_topics(data= df,
                                    doc_topic_matrix= doc_topic,
                                    top_n_topics= 3,
                                    threshold = 0.1)
        
        
        
        ###########################################
        ##  SAVE TOPIC RESULTS
        ###########################################
        
        # destination bucket
        dest_bucket = 'data-labelled-topics' 
        save_processed_tweets(df, dest_bucket, last_hour)
        
        
        
        # os.listdir(download_path)
        # guided_lda.print_topics(num_words=10)
        # df.columns.values
        return True
    
    except Exception as e:
        print(e)
        #print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(file_key, bucket_name))
        #raise e        
        
