#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install --upgrade gensim
#pip install tqdm
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


# import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
# use English stopwords
stops = set(stopwords.words("english"))


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


# In[ ]:


# This function is for tokenising sentences in the corpus
def tokenising_corpus(data):
    # Transform df into list
    words = data.tolist()
    # tokenising each sentence
    word_tokens = []
    for tweet in words:
        word_tokens.append(word_tokenize(tweet))
        
    return word_tokens


# In[24]:


# This function allows user to remove stopwords
# and also allow to specify and remove some irrelevant words in this case (such sentiments) for tuning the model
# This will only apply to the first LDA model for get more clear topics
def custom_words_remover(word_lst, text_tokens):
    # create a new list with specified words removed 
    processed_tokens = []
    for token in text_tokens:
        processed_tokens.append([w for w in token if not w in word_lst])
        
    return processed_tokens


# In[ ]:


# This function is for stemming the words in the corpus
def stemming_words(text_tokens):
    #from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    # stemming the tokens
    stemmed = []
    for token in text_tokens:
        stemmed.append([ps.stem(word) for word in token])
    
    return stemmed


# In[5]:


# Function for making biagram
# This funciton is from Prabhakaran (2018): https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
def make_bigrams(data,min_count,thres):
    # Build the bigram model with min_count=10
    # higher threshold fewer phrases.
    bigram = gensim.models.Phrases(data, min_count=min_count, threshold=thres)
    
    # Faster way to get a sentence clubbed as a bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    
    return [bigram_mod[doc] for doc in data]


# In[42]:


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


# In[ ]:


# Supporting function of the model tuning: build individual lda model and compute its coherence
# This function is from Kapadia (2019):https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
from gensim.models import CoherenceModel

def compute_coherence_values_basic(data,corpus,dictionary,k,alpha):
    # build individual lda model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           alpha = alpha,
                                           random_state=5,
                                           passes=10)
    
    #p = lda_model.log_perplexity(corpus)
    # build coherence model
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                         texts=data, 
                                         dictionary=dictionary, 
                                         coherence='c_v')
    # get coherence score
    return coherence_model_lda.get_coherence()


# In[ ]:


# This function search the optimal hyperparameter settings for the lda model
# Similar to the grid search 
# It can take a long time to run
# This function is from Kapadia (2019):https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
def tuning_lda_model(data,corpus,dictiornay, min_topics, max_topics, step_size):
    # tqdm is a progress bar for visualising the cost time
    import tqdm

    grid = {}
    grid['Validation_Set'] = {}
    
    # Set topics range
    topics_range = range(min_topics, max_topics, step_size)
    
    # Alpha parameter
    alpha = [0.01, 0.1, 0.3, 0.6, 1]
    alpha.append('symmetric')
    alpha.append('asymmetric')
    
    # Use 75% of original corpus as the validation sets
    num_of_docs = len(corpus)
    corpus_sets = [ gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)),
                    corpus]

    corpus_title = ['75% Corpus', '100% Corpus']
    model_results = {'Validation_Set': [],
                     'Alpha':[],
                     'Topics': [],
                     'Coherence': []}
    
    # calculate iterating times
    t = 0
    for i in range(len(corpus_sets)):
        for a in alpha:
            for k in topics_range:
                #print(i,' ',a,' ',k)
                t += 1
                
    print('iteration times: ',t)
    
    # Can take a long time to run
    pbar = tqdm.tqdm(total=t)
    
    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through alpha values
        for a in alpha:
            # iterate through number of topics
            for k in topics_range:
                # Compute the coherence for each model
                cv = compute_coherence_values_basic(data=data,
                                                    corpus=corpus_sets[i],
                                                    dictionary=dictiornay,
                                                    k=k,
                                                    alpha=a)                
                # Save the model results
                model_results['Validation_Set'].append(corpus_title[i])
                model_results['Alpha'].append(a)
                model_results['Topics'].append(k)
                model_results['Coherence'].append(cv)
                print('pass')
                
                pbar.update(1)
                
    pbar.close()
    
    return model_results


# In[ ]:


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


# In[88]:


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


# In[86]:


# Function is for deleting abandoned topic from the eta matrix
# This function is from Deshpande (2012): https://stackoverflow.com/questions/3877491/deleting-rows-in-numpy-array
def abandon_topic(topic_id, matrix):
    matrix = np.delete(matrix, (topic_id), axis=0)
    return matrix


# In[ ]:


# save the model to model_path
def save_lda_model(model, model_name, save_path):
    # save the model to model_path
    model.save(save_path+'{}.model'.format(model_name))
    # get list of componenets
    components = [file for file in os.listdir(model_path) if file.startswith(model_name)]
    
    return components


# In[ ]:


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


# In[ ]:


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





# In[6]:


# read the data
bucket_name = "proxy-data-and-pre-collected-data-for-training"
file_key_text = "20191226-reviews.csv"
file_key_brand = "20191226-items.csv"

data_location_text = "s3://proxy-data-and-pre-collected-data-for-training/20191226-reviews.csv"
data_location_brand = "s3://proxy-data-and-pre-collected-data-for-training/20191226-items.csv"


# In[7]:


df_text = pd.read_csv(data_location_text)
df_brand = pd.read_csv(data_location_brand)


# In[8]:


df_text.head()


# In[9]:


# merge the two dataset
df_brand = df_brand[['asin','brand']]
df = df_text.merge(df_brand, how='left', on='asin')
#drop irrelevant contents
df.drop(['asin','name','date','verified'], axis=1, inplace=True)
df.head()


# In[10]:


# Initial exploratory data analysis
df.groupby(['brand']).count()


# In[11]:


# Check NA
print(df.isnull().sum())


# In[12]:


# Delete NA in review title and body
df = df.dropna(axis=0, subset=['title','body'])


# In[13]:


# reset the dataframe index
df = df.reset_index()
# drop old index column
df = df.drop(['index'], axis=1)


# In[14]:


# Combine the review title and body into full text for analysis
df['text'] = df['title'] + ' ' +df['body']


# In[15]:


df.head()


# In[16]:


#########################################
##  BASIC CLEANING AND TEXT PROCESSING
#########################################

# text cleaning
df['text'] = text_cleaning(df['text'])


# In[19]:


# Tokenising sentences in the corpus
word_tokens = tokenising_corpus(df['text'])

# keep the copies
df['text_tokens'] = word_tokens

# get the length of each text
df['text_len'] = df['text_tokens'].map(lambda x: len(x))


# In[25]:


###################################
#      Deleting stop words
###################################

# create a list of stopwords
stops = set(stopwords.words("english"))
# remove stopwords
filtered_tokens = custom_words_remover(stops, word_tokens)


# In[ ]:


###################################
#           Stemming
###################################
stemmed = stemming_words(filtered_tokens)


# In[20]:


'''
When keeping the emotional words, the LDA model tends to classify topics based on positive and negative sentiments, 
rather than based on single functions. 
For example, the negative comments about the price and the negative comments about the battery
will be put under one topic. In this way, reviews about one single function might be scattered on several topics. 
However, the focus of the first LDA model should be extract hot words about specific cellphone features or functions 
and generate the corresponding eta matrix.  
Therefore, when tuning the model, irrelevant words that express sentiments should be removed from the texts.

"star" will also be deleted, 
because in this case "star" is typically used to describe the sentimental polarity of customers (5 star = best while 1star = worst)

Another example is that in the first model, the topic "screen" might associate with strongly positive attitudes, 
but when it comes to the second model, most people might actually complain the srceen of the Huawei P50, 
and associate the negetive word with the topic "screen"
'''


# In[30]:


# Use defined function to specify some irrelevant words in this case (such sentiments) for tuning the model
# This will only apply to the first LDA model for get more clear topics

# Add emotional words as new stopword list
newstopwords = ['like', 'good', 'better', 'best', 'bad', 'worse', 'worst', 'happi', 'great', 'really','realli',
                'love', 'lov', 'also', 'awesome','awesom','amaz','lousi','far','well','perfectli','ok',
                'ever','perfect','fun','excelent','excel','excelled','absolut','less','much','more','fewer','fine','finest',
                'exactli','poor','pleas','glad','veri','high','terribl','minim','never','even','thank','gift','star','thank'] 

# delete new added stopwords from the text tokens
df['processed_tokens'] = custom_words_remover(newstopwords, stemmed)


# In[25]:


# Phrase Modeling: Making Bigrams
# Build the bigram model with min_count=10
# higher threshold fewer phrases.
df['processed_tokens'] = make_bigrams(data= df['processed_tokens'],min_count= 10,thres= 100)


# In[34]:


# drop processed columns
df.drop(['title','body'], axis=1, inplace=True)


# In[40]:


df.head()


# In[ ]:





# In[28]:


#########################################################
#  Create word dictionary and bag of words (bow) corpus
#########################################################
# Create id to word Dictionary
# id2word is a dictionary containing the IDs of all input words
id2word = corpora.Dictionary(df['processed_tokens'])

# Create corpus that contains all documents
texts = df['processed_tokens']

# Create bag of word for each document in the corpus 
# each bow contains the id of each word in that single document and its number of occurrences in that document 
# (term id, term document Frequency)
corpus = [id2word.doc2bow(text) for text in texts]


# In[ ]:





# In[43]:


##########################################################################
##            Building the basic LDA model
##########################################################################
# assuming number of topics
num_topics = 10

# Build LDA model
# Remember to set the minimum_probability=0 in the model or can't get probabilities of a word under each topic
lda_model = gensim.models.LdaMulticore(corpus = corpus,
                                       id2word = id2word,
                                       num_topics = num_topics,
                                       passes = 10,
                                       random_state=5, 
                                       minimum_probability=0)


# In[44]:


# Compute the perplexity and coherence score of the model
model_benchmarking(df['processed_tokens'], lda_model, id2word, corpus)


# In[ ]:





# In[ ]:


##########################################################################
##                  Tuning the LDA model on proxy data
##########################################################################
# Use defined functions to tune the lda model and find optimal hyperparameter settings
# It can take a long time to run
model_results = tuning_lda_model(data = df['processed_tokens'],
                                 corpus = corpus,
                                 dictiornay = id2word,
                                 min_topics = 5, 
                                 max_topics = 6,
                                 step_size = 1)

# Convert to the dataframe and save to the csv files
model_results_df = pd.DataFrame.from_dict(model_results)
model_results_df.to_csv("lda_on_proxy_tuning_results.csv")


# In[ ]:





# In[ ]:





# In[79]:


##########################################################################
##          Building the LDA model (with optimal parameter)
##########################################################################

# Optimal model after tuning: 
# Hyperparameters: num_topics = 11, alpha = 'symmetric', passes =10

# Set number of topics
num_topics = 11

# Build LDA model
# Remember to set the minimum_probability=0 in the model or can't get probabilities of a word under each topic
lda_model = gensim.models.LdaMulticore(corpus = corpus,
                                       id2word = id2word,
                                       num_topics = num_topics,
                                       passes = 10,
                                       alpha = 'symmetric',
                                       random_state=5, 
                                       minimum_probability=0)

# Perplexity:  -7.206703803237903
# Coherence Score:  0.5061053071793964


# In[80]:


# Compute the perplexity and coherence score of the model
model_benchmarking(df['processed_tokens'], lda_model, id2word, corpus)


# In[ ]:





# In[32]:


# from pprint import pprint
# print the top 20 keywords under each topic
pprint(lda_model.print_topics(num_words=20))


# In[ ]:





# In[ ]:


# Visualisation
# Visualize the topics 
# lambda = 0.6 can be ideal
pyLDAvis.enable_notebook()
LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
LDAvis_prepared


# In[ ]:





# In[82]:


##########################################################
##   Check topics and contents under each topic
##########################################################
# create test df
test = df
texts = test['processed_tokens']
# create new corpus
corpus_new = [lda_model.id2word.doc2bow(text) for text in texts]


# In[84]:


# Creating the documents-topic matrix
# which can show the individual document's probabilities for each topic
doc_topic = create_doc_topic_matrix(model = lda_model,
                                    corpus = corpus_new,
                                    num_topics = num_topics)
print(doc_topic.head())


# In[103]:


# Concat documents-topic matrix with the review dataframe
joined_df = pd.concat([df, doc_topic], axis = 1, join = 'outer')


# In[88]:


# Select the 20 comments that are most relevant to topic n
# Notice that the column name is INT value in this case
joined_df.sort_values(by = 10,ascending=False)['text'].iloc[0:19].tolist()


# In[76]:


# show the top 20 words under each topic
lda_model.show_topic(topicid = 0, topn = 20)


# In[ ]:





# In[89]:


#################################
#    Create eta matrix
#################################
# Creating eta matrix with top 20 words under each topic
# the eta matrix can be used to train the guided lda model as a prior belief on word probability 
# can be use to assign probabilities for each word-topic combination
eta_matrix = create_eta_matrix(num_topics,20,lda_model,id2word)


# In[90]:


# Check if it works well
print(eta_matrix.shape,'\n')
print(lda_model.get_topic_terms(topicid=5,topn=20),'\n')
print(eta_matrix[5][72],'\n')
print(eta_matrix[5][122],'\n')


# In[39]:


# Deleting abandoned topic from the eta matrix
# topic 3 is reviews in Spanish and topic 4 is talking about cellphone refurbishment
# They can be considered irrelevant in the future analysis, so need to be abandoned
# abandon topic 3: Spanish 
eta_matrix = abandon_topic(topic_id = 3, matrix = eta_matrix)
# abandon topic 4: Refurbishment
# Notice now original topic 4 become the topic 3 in the eta matrix (after deleting the previous topic3)
eta_matrix = abandon_topic(topic_id = 3, matrix = eta_matrix)


# In[43]:


# Check if it works well
#eta_matrix[:,72]
print(eta_matrix.shape,'\n')
print(lda_model.get_topic_terms(3),'\n')
print(lda_model.get_topic_terms(4),'\n')

print(lda_model.get_topic_terms(5),'\n')
print(eta_matrix[3][72],'\n')


# In[ ]:





# In[96]:


#################################
#    Save the model
#################################
#homepath = '/home/ec2-user/SageMaker/'
homepath = os.getcwd()
model_path = homepath + 'LDA_model_on_proxy_data/'
print(model_path)


# In[100]:


# save the model to model_path and get list of componenets
components = save_lda_model(lda_model, 'lda_model_on_proxy_data', model_path)


# In[102]:


# Check if it works well
print(os.listdir(model_path),'\n')
print(components)


# In[ ]:





# In[ ]:


#################################
#    Upload to S3
#################################
bucket_name = 'lda-model-on-proxy-data'
#homepath = '/home/ec2-user/SageMaker/'
#model_path = homepath + 'LDA_model_on_proxy_data/'


# In[229]:


# Upload eta_matrix and list of component to S3
file_upload_helper(file = eta_matrix, file_name ='eta_matrix', bucket_name='lda-model-on-proxy-data')
file_upload_helper(file = components, file_name ='components', bucket_name='lda-model-on-proxy-data')


# In[151]:


# Upload model to S3
model_upload_helper(components, model_path, bucket_name)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




