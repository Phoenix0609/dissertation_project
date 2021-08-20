#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install spacy
# pip install --upgrade gensim
# pip install tqdm
### pip install guidedlda
#pip install pyldavis
# install pickle


# In[1]:


# Define IAM role
import boto3

# NLP things
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# visualisation
import pyLDAvis.gensim 
import pyLDAvis

# import others
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import io
from io import StringIO
import string
import re

import os
import json
import time
from datetime import datetime, timedelta
import pickle
from pprint import pprint
import sys
import urllib.parse
import csv


# In[2]:


#import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
# use English stopwords
stops = set(stopwords.words("english"))


# In[ ]:





# In[3]:


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


# In[4]:


# This function is for tokenising sentences in the corpus
def tokenising_corpus(data):
    # Transform df into list
    words = data.tolist()
    # tokenising each sentence
    word_tokens = []
    for tweet in words:
        word_tokens.append(word_tokenize(tweet))
        
    return word_tokens


# In[5]:


# This function allows user to remove stopwords
# and also allow to specify and remove some irrelevant words in this case (such sentiments) for tuning the model
# This will only apply to the first LDA model for get more clear topics
def custom_words_remover(word_lst, text_tokens):
    # create a new list with specified words removed 
    processed_tokens = []
    for token in text_tokens:
        processed_tokens.append([w for w in token if not w in word_lst])
        
    return processed_tokens


# In[6]:


# This function is for stemming the words in the corpus
def stemming_words(text_tokens):
    #from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    # stemming the tokens
    stemmed = []
    for token in text_tokens:
        stemmed.append([ps.stem(word) for word in token])
    
    return stemmed


# In[7]:


# Function for making biagram
# This funciton is from Prabhakaran (2018): https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
def make_bigrams(data,min_count,thres):
    # Build the bigram model with min_count=10
    # higher threshold fewer phrases.
    bigram = gensim.models.Phrases(data, min_count=min_count, threshold=thres)
    
    # Faster way to get a sentence clubbed as a bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    
    return [bigram_mod[doc] for doc in data]


# In[8]:


# Compute the perplexity and coherence score of the model
# This function is from Kapadia (2019):https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
def model_benchmarking(data, model, dictionary, corpus):
    # Compute Model Perplexity
    p = model.log_perplexity(corpus)
    print('\nPerplexity: ', p)

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model = model, 
                                         texts = data, 
                                         dictionary = dictionary, 
                                         coherence = 'c_v')

    coherence_lda = coherence_model_lda.get_coherence()

    print('\nCoherence Score: ', coherence_lda)


# In[9]:


# Function is for fliterring tweets containing only 'huawei' AND 'p50'
# Despite that tweets have been filtered in the streaming stage, a extra filter is set here to ensure accuracy
# This function is from Flyingmeatball (2016): https://stackoverflow.com/questions/37011734/pandas-dataframe-str-contains-and-operation
# Notice that the word pair should be exactly two words
def tweets_filter(data, words_pair):
    # case to false means ignoring case
    data = data[(data['text'].str.contains(words_pair[0], case=False)) & (data['text'].str.contains(words_pair[1], case=False))]
    return data


# In[10]:


# Function for download files including the model component list and eat matrix
# This function is from Kindjacket (2019): https://stackoverflow.com/questions/48964181/how-to-load-a-pickle-file-from-s3-to-use-in-aws-lambda
def file_download_helper(file_key, bucket_name):
    s3_resource = boto3.resource('s3')
    # load pkl file
    file = pickle.loads(s3_resource.Bucket(bucket_name).Object(file_key).get()['Body'].read())
    print('Successfully load: ',file_key)
    return file


# In[11]:


# Function for downloading model to S3
# This function is from Sophros (2020): https://stackoverflow.com/questions/61638940/save-a-gensim-lda-model-to-s3
def model_download_helper(download_path, file_lst, bucket_name):
    
    s3_resource = boto3.resource('s3')
    # download model components
    for file_name in file_lst:
        full_path = download_path + file_name
        s3_resource.Bucket(bucket_name).download_file(file_name, full_path)
        print('Successfully download: ', file_name)


# In[12]:


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


# In[13]:


# This function is for updating the eta matrix to fit the new number of topics
# Notice that the number of new topics should be more than 9 (the num of rows in the eta matrix)
# which is the number of pre-defined topics with prior knowledge
def eta_matrix_updater(num_topics, ori_len, new_len, matrix):
    
    extended_len = new_len - ori_len
    ori_num_topics = matrix.shape[0]
    
    # if the number of new topics equals to the number of pre-defined topics
    if num_topics != ori_num_topics:
        matrix_new_added_1 = np.full(((num_topics-ori_num_topics), ori_len), 0.001)
        matrix_temp = np.r_[ matrix, matrix_new_added_1] # add rows
        # extend the matrix 
        matrix_new_added_2 = np.full((num_topics, extended_len), 0.001)
        matrix = np.c_[ matrix_temp, matrix_new_added_2]
    # if the number of new topics is more than to the number of pre-defined topics
    else:
        matrix_new_added = np.full((num_topics, extended_len), 0.001)
        # extend the matrix 
        matrix = np.c_[ matrix, matrix_new_added]    
    
    return matrix


# In[14]:


# Compute the perplexity and coherence score of the model
# This function is from Kapadia (2019):https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
def model_benchmarking(data, model, dictionary, corpus):
    # Compute Model Perplexity
    p = model.log_perplexity(corpus)
    print('\nPerplexity: ', p)

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model = model, 
                                         texts = data, 
                                         dictionary = dictionary, 
                                         coherence = 'c_v')

    coherence_lda = coherence_model_lda.get_coherence()

    print('\nCoherence Score: ', coherence_lda)


# In[15]:


# Supporting function of the model tuning: build individual lda model and compute its coherence
# This function is from Kapadia (2019):https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
from gensim.models import CoherenceModel

def compute_coherence_values_basic(data,corpus,dictionary,k,matrix):
    # build individual lda model
    # assuming that each document tends to cover multiple topics, thus the alpha should be symmetric
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           alpha = 'symmetric',
                                           eta = matrix,
                                           random_state=5,
                                           passes=30)
    
    #p = lda_model.log_perplexity(corpus)
    # build coherence model
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                         texts=data, 
                                         dictionary=dictionary, 
                                         coherence='c_v')
    # get coherence score
    return coherence_model_lda.get_coherence()


# In[16]:


# This function search the optimal hyperparameter settings for the lda model
# Similar to the grid search 
# It can take a long time to run
# This function is from Kapadia (2019):https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
def tuning_guided_lda_model(data,corpus,dictiornay, min_topics, max_topics, step_size):
    # tqdm is a progress bar for visualising the cost time
    import tqdm

    grid = {}
    grid['Validation_Set'] = {}
    
    # Set topics range
    topics_range = range(min_topics, max_topics, step_size)
    
    # Use 75% of original corpus as the validation sets
    num_of_docs = len(corpus)
    corpus_sets = [ gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)),
                    corpus]

    corpus_title = ['75% Corpus', '100% Corpus']
    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Coherence': []}
    
    # calculate iterating times
    t = 0
    for i in range(len(corpus_sets)):
        for k in topics_range:
            t += 1
                
    print('iteration times: ',t)
    
    # Can take a long time to run
    pbar = tqdm.tqdm(total=t)
    
    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # update eta_matrix
            eta_matrix = eta_matrix_updater( k, ori_len, new_len, eta_matrix_ori)
            # Compute the coherence for each model
            cv = compute_coherence_values_basic(data=data,
                                                corpus=corpus_sets[i],
                                                dictionary=dictiornay,
                                                k=k,
                                                matrix=eta_matrix)                
            # Save the model results
            model_results['Validation_Set'].append(corpus_title[i])
            model_results['Topics'].append(k)
            model_results['Coherence'].append(cv)
            print('pass')
                
            pbar.update(1)
                
    pbar.close()
    
    return model_results


# In[17]:


# This functions is for creating the documents-topic matrix
# which can show the individual document's probabilities for each topic
# This function is from Wang (2019): https://stackoverflow.com/questions/56408849/after-applying-gensim-lda-topic-modeling-how-to-get-documents-with-highest-prob

def create_doc_topic_matrix(model, corpus, num_topics):
    # Create a dictionary, with topic ID as the key, and the value is a list of tuples (docID, probability of this particular topic for the doc) 
    topic_dict = {i: [] for i in range(num_topics)}
    
    # Remember to set the minimum_probability=0 in the model or can't get probabilities of one under each topic
    # Loop over all the documents to group the probability of each topic
    for doc_id in range(len(corpus)):
        topic_vector = model[corpus[doc_id]]
        for topic_id, prob in topic_vector: 
            topic_dict[topic_id].append(prob)
    
    # Create documents-topic matrix
    doc_topic = pd.DataFrame.from_dict(topic_dict)
    
    return doc_topic


# In[73]:


# Function for creating eta matrix for training the guided lda model
# the eta matrix can be used as a prior belief on word probability
# can be use to assign probabilities for each word-topic combination
def create_eta_matrix(num_topics,top_n,lda_model,id2word):
    # get dictionary length
    dic_len = len(id2word.token2id)
    # initialising eta matrix with 0.001
    eta_matrix = np.full((num_topics, dic_len), 0.001)
    
    # update the eta_matrix
    # add the confidence to top_n words based on the model output probabilities
    # hierarchical assignment: assign top 10 words with extra 0.15 and assign the top 10-20 words with 0.1
    for topic_i in range(num_topics):
        top_words = lda_model.get_topic_terms(topicid=topic_i,topn=top_n)
        #count = 0
        for pair in top_words:
            #print(pair[0],pair[1],'\n')
            if top_words.index(pair) < 10:
                eta_matrix[topic_i][pair[0]] = pair[1] + 0.10
            else:
                eta_matrix[topic_i][pair[0]] = pair[1] + 0.05
                
    return eta_matrix


# In[18]:


# save the model to model_path
def save_lda_model(model, model_name, save_path):
    # save the model to model_path
    model.save(save_path+'{}.model'.format(model_name))
    # get list of componenets
    components = [file for file in os.listdir(model_path) if file.startswith(model_name)]
    
    return components


# In[19]:


# Function for uploading eta matrix and list of component to S3
# This function is from Shabani (2018): https://stackoverflow.com/questions/49120069/writing-a-pickle-file-to-an-s3-bucket-in-aws
def file_upload_helper(file, file_name, bucket_name):
    # create S3 resource
    s3_resource = boto3.resource('s3')
    
    # covert the file to pkl
    obj_pkl = pickle.dumps(file)
    obj_key = '{}.pkl'.format(file_name)
    
    s3_resource.Object(bucket_name, obj_key).put(Body=obj_pkl)
    print('Success')


# In[20]:


# Function for uploading model to S3
# This function is from Sophros (2020): https://stackoverflow.com/questions/61638940/save-a-gensim-lda-model-to-s3
def model_upload_helper(file_lst, local_path, bucket_name):
    for file_name in file_lst:
        # get file path
        file_path = local_path + file_name        
        # create s3 resource
        s3_resource = boto3.resource('s3')
        # upload file
        s3_resource.meta.client.upload_file(file_path, bucket_name, file_name)
        print('successfully upload ' + file_name)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


# read pre-event data
bucket_name = "proxy-data-and-pre-collected-data-for-training"
data_location = "s3://proxy-data-and-pre-collected-data-for-training/tweets_20210727.csv"


# In[22]:


df = pd.read_csv(data_location, dtype = 'str')
print(df.shape)


# In[23]:


# remove duplications with the same tweet id
df = df.drop_duplicates('tweet_id')
print(df.shape)


# In[24]:


# keep useful contents
df = df[['tweet_id','user_id','text']]


# In[25]:


# read post-event data
bucket_name = "proxy-data-and-pre-collected-data-for-training"
data_location_2 = "s3://proxy-data-and-pre-collected-data-for-training/Streaming_data_from_20210729_to_20200805.csv"


# In[26]:


df_2 = pd.read_csv(data_location_2, dtype = 'str')
print(df_2.shape)


# In[27]:


# drop irrelevant contents
df_2.drop('Unnamed: 0',axis=1, inplace = True)


# In[28]:


# Delete duplications with the same tweet id
df_2 = df_2.drop_duplicates('tweet_id')
print(df_2.shape)


# In[29]:


# keep useful contents
df_2 = df_2[['tweet_id','user_id','tweet_text']]


# In[30]:


# Rename columns
df_2.rename(columns = { 'tweet_text': 'text'}, inplace=True)


# In[31]:


# concat the two dataframes
df = pd.concat([df, df_2], axis = 0)
print(df.shape)


# In[32]:


# Check NA
print(df.isnull().sum())


# In[33]:


# Delete NA in review id and text
df = df.dropna(axis=0, subset=['tweet_id','text'])


# In[34]:


print(df.shape)


# In[35]:


# filterring tweets based on keywords
# set word pair
words_pair = ['huawei','p50']
# keeping tweets containing only 'huawei' AND 'p50'
df = tweets_filter(df, words_pair)


# In[36]:


print(df.shape)


# In[37]:


# reset the dataframe index
df = df.reset_index()
# drop old index column
df = df.drop(['index'], axis=1)


# In[ ]:





# In[ ]:





# In[38]:


#########################################
##  BASIC CLEANING AND TEXT PROCESSING
#########################################

# text cleaning
df['text'] = text_cleaning(df['text'])


# In[39]:


# Tokenising sentences in the corpus
word_tokens = tokenising_corpus(df['text'])

# keep the copies
df['text_tokens'] = word_tokens

# get the length of each text
df['text_len'] = df['text_tokens'].map(lambda x: len(x))


# In[40]:


df.head()


# In[41]:


###################################
#      Deleting stop words
###################################

# create a list of stopwords
stops = set(stopwords.words("english"))
# remove stopwords
filtered_tokens = custom_words_remover(stops, word_tokens)


# In[42]:


# Tweets are filtered based on 'huawei' and 'p50', 
# which means that every text in the current data is about 'huawei p50' and contains'huawei' and'p50'. 
# Therefore, all close expressions of huawei p50 can be deleted to improve the accuracy

# Add p50 words as new stopword list
newstopwords = ['huawei','p50','huaweip50','huaweip50pro','p50pro','pro','huaweip50series','series']

# delete new added stopwords from the text tokens
filtered_tokens = custom_words_remover(newstopwords, filtered_tokens)


# In[ ]:





# In[ ]:





# In[43]:


###################################
#           Stemming
###################################
# Use defined function to stem the words
stemmed = stemming_words(filtered_tokens)

df['processed_tokens'] = stemmed


# In[44]:


# Phrase Modeling: Making Bigrams
# Build the bigram model with min_count=10
# higher threshold fewer phrases.
df['processed_tokens'] = make_bigrams(data= df['processed_tokens'],min_count= 10,thres= 100)


# In[ ]:





# In[45]:


# check data
df.head()


# In[46]:


df.shape


# In[ ]:





# In[ ]:





# In[47]:


##############################################################
###      Download model component list and eta matrix
##############################################################
# location info
bucket_name = "lda-model-on-proxy-data"

model_component_list_key = "components.pkl"
eta_matrix_key = "eta_matrix.pkl"


# In[48]:


# Download model component list and eta matrix
components = file_download_helper(model_component_list_key, bucket_name)
eta_matrix_ori = file_download_helper(eta_matrix_key, bucket_name)


# In[49]:


# check if it works well
print(eta_matrix_ori.shape)
print(eta_matrix_ori[0][72])


# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


##############################################################
###      Download proxy lda model and dictionarys
##############################################################


# In[50]:


# 下载模型

# 1.直接从S3 加载
# https://stackoverflow.com/questions/50477192/loading-word2vec-binary-model-from-s3-into-gensim-fails

# 2.模型打包和下载都用pickel 不用download到本地
# https://stackoverflow.com/questions/50655405/gensim-pickle-or-not

# 3.适用于lambda, 下载到临时本地，然后模型加载
# https://github.com/RaRe-Technologies/gensim/issues/1851


# In[51]:


# download file to local folder
# location info
bucket_name = "lda-model-on-proxy-data"
download_path = '/home/ec2-user/SageMaker/LDAdownloads/'
print(os.listdir(download_path))


# In[52]:


# download the model (with all its components) to the local folder
model_download_helper(download_path, components, bucket_name)


# In[53]:


# check if it works well
print(os.listdir(download_path))
print(download_path)


# In[54]:


# load the model 
proxy_lda_model = gensim.models.LdaMulticore.load(download_path + 'lda_model_on_proxy_data.model')
proxy_lda_model


# In[55]:


##########################################################
#       Updating word dictionary and eta matrix
##########################################################


# In[56]:


# get the pre-generated dictionary
id2word = proxy_lda_model.id2word


# In[57]:


# Update the dictionary with words in new corpus
# The id of the original word will remain the same, 
# and new words will get ids starting from the last number in the original dictionary
id2word, ori_len, new_len = dictionary_updater(dictionary= id2word, data= df['processed_tokens'])


# In[58]:


# check if it works well
print(ori_len)
print(new_len)


# In[ ]:





# In[76]:


# 检验
## 记住这里初始的eta matrix叫eta_matrix_ori
#num_topics = 15
#eta_matrix_test = eta_matrix_updater(num_topics, ori_len, new_len, eta_matrix_ori)


# In[ ]:





# In[59]:


#########################################################
#          Create bag of words (bow) corpus
#########################################################

# Create corpus that contains all documents
texts = df['processed_tokens']

# Create bag of word for each document in the corpus 
# each bow contains the id of each word in that single document and its number of occurrences in that document 
# (term id, term document Frequency)
corpus = [id2word.doc2bow(text) for text in texts]


# In[ ]:





# In[ ]:





# In[ ]:





# In[71]:


#guided_lda = proxy_lda_model


# In[ ]:





# In[ ]:





# In[79]:


##########################################################################
###              Building the basic guided LDA model
##########################################################################
# define num of topics
num_topics = 10
# update eta_matrix
eta_matrix = eta_matrix_updater(num_topics, ori_len, new_len, eta_matrix_ori)


# In[81]:


# Build guided LDA model
# Remember to set the minimum_probability=0 in the model or can't get probabilities of a word under each topic
guided_lda = gensim.models.LdaMulticore(corpus = corpus,
                                        id2word = id2word,
                                        num_topics = num_topics,
                                        passes = 30,
                                        random_state=5,
                                        eta = eta_matrix,
                                        minimum_probability=0)


# In[82]:


# Compute the perplexity and coherence score of the model
model_benchmarking(df['processed_tokens'], guided_lda, id2word, corpus)


# In[88]:


##########################################################################
##        Tuning the guided LDA model on the pre collected data
##########################################################################
# Use defined functions to tune the guided lda model and find optimal hyperparameter settings
# Notice that the number of new topics should be more than 9 (the num of rows in the eta matrix)
# It can take a long time to run
model_results = tuning_guided_lda_model(data = df['processed_tokens'],
                                        corpus = corpus,
                                        dictiornay = id2word,
                                        min_topics = 9, 
                                        max_topics = 20,
                                        step_size = 1)

# Convert to the dataframe and save to the csv files
model_results_df = pd.DataFrame.from_dict(model_results)
model_results_df.to_csv("guided_lda_on_pre_collected_data_tuning_results.csv")


# In[92]:


##########################################################################
##          Building the guided LDA model (with optimal parameter)
##########################################################################
# Optimal model after tuning: 
# Hyperparameters: num_topics = 12, alpha = 'symmetric', passes = 30

# set num of topics
num_topics = 12
# update eta_matrix
eta_matrix = eta_matrix_updater(num_topics, ori_len, new_len, eta_matrix_ori)


# In[94]:


# Build guided LDA model
# Remember to set the minimum_probability=0 in the model or can't get probabilities of a word under each topic
guided_lda = gensim.models.LdaMulticore(corpus = corpus,
                                        id2word = id2word,
                                        num_topics = num_topics,
                                        passes = 30,
                                        alpha = 'symmetric',
                                        eta = eta_matrix,
                                        random_state=5,
                                        minimum_probability=0)
# Perplexity:  -5.88497913815473
# Coherence Score:  0.4357691561491484


# In[95]:


# Compute the perplexity and coherence score of the model
model_benchmarking(df['processed_tokens'], guided_lda, id2word, corpus)


# In[96]:


# from pprint import pprint
# print the top 20 keywords under each topic
pprint(guided_lda.print_topics(num_words=20))


# In[99]:


# Visualisation
# Visualize the topics 
# lambda = 0.6 can be ideal
pyLDAvis.enable_notebook()
LDAvis_prepared = pyLDAvis.gensim.prepare(guided_lda, corpus, id2word)
LDAvis_prepared


# In[100]:


##########################################################
##   Check topics and contents under each topic
##########################################################
# create test df
test = df
texts = test['processed_tokens']
# create new corpus
corpus_new = [guided_lda.id2word.doc2bow(text) for text in texts]


# In[103]:


# Creating the documents-topic matrix
# which can show the individual document's probabilities for each topic
doc_topic = create_doc_topic_matrix(model = guided_lda,
                                    corpus = corpus_new,
                                    num_topics = num_topics)
print(doc_topic.head())


# In[104]:


# Concat documents-topic matrix with the review dataframe
joined_df = pd.concat([df, doc_topic], axis = 1, join = 'outer')


# In[105]:


# Select the 20 comments that are most relevant to topic n
# Notice that the column name is INT value in this case
joined_df.sort_values(by = 10,ascending=False)['text'].iloc[0:19].tolist()


# In[106]:


# show the top 20 words under each topic
guided_lda.show_topic(topicid = 0, topn = 20)


# In[ ]:





# In[76]:





# In[ ]:





# In[77]:


#################################
#    Create eta matrix
#################################
# Creating eta matrix with top 20 words under each topic
# the eta matrix can be used to train the guided lda model as a prior belief on word probability 
# can be use to assign probabilities for each word-topic combination
eta_matrix_production_v1 = create_eta_matrix(num_topics,20,guided_lda,id2word)


# In[111]:


#################################
#    Save the model
#################################
#homepath = '/home/ec2-user/SageMaker/'
homepath = os.getcwd()
#homepath = '/home/ec2-user/SageMaker/'
model_path = homepath + 'guided_lda_for_production/'
print(model_path)


# In[112]:


# save the model to model_path and get list of componenets
components = save_lda_model(guided_lda, 'guided_lda_for_production', model_path)


# In[116]:


# Check if it works well
print(os.listdir(model_path),'\n')
print(components)


# In[79]:


#################################
#    Upload to S3
#################################
bucket_name = 'guided-lda-on-pre-collected-data'
#homepath = '/home/ec2-user/SageMaker/'
#model_path = '/home/ec2-user/SageMaker/guided_lda_for_production/'


# In[80]:


# Upload list of components and eta matrix for production to S3
file_upload_helper(file = components, file_name ='components', bucket_name='guided-lda-on-pre-collected-data')
file_upload_helper(file = eta_matrix_production_v1, file_name ='eta_matrix_production_v1', bucket_name='guided-lda-on-pre-collected-data')


# In[ ]:


# Upload model to S3
model_upload_helper(components, model_path, bucket_name)


# In[ ]:





# In[ ]:





# In[ ]:




