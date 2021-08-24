# remember to allocate more RAM in the configuration, at least 1024 MB
# 提一下lambda可能不是最优的训练场所 因为有时间限制,所以话题数量不敢设的太高 也不敢用测试集
# Notice that Lambda gets maximum duration up to 15 mins per execution, 
# thus need to carefully design hyperparameter grids and allocate computing resources (like RAM)
import boto3

# LDA models
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# Visualisation
from pyLDAvis import gensim_models
import pyLDAvis
from fpdf import FPDF

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
            # do not set quoting, since the data like processed words lists are quoted with ""
            # jump bad lines
            next_df = pd.read_csv(response.get("Body"), error_bad_lines=False)
            df = pd.concat([df, next_df], axis = 0, join = 'outer')
        t += 1
    return df
    
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
    
# Function for downloading model from S3
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
    # hierarchical assignment: assign top 10 words with extra 0.10 and assign the top 10-20 words with 0.05
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

# This function is for updating the eta matrix to fit the new number of topics
# Notice that the number of new topics should be more than 9 (the num of rows in the eta matrix)
# which is the number of pre-defined topics with prior knowledge
def eta_matrix_updater(num_topics, ori_len, new_len, matrix):
    # calculate extensions of the matrix
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
    

##########################################################
#     Function for model tuning
########################################################## 

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
    
    return coherence_lda

# Supporting function of the model tuning: build individual lda model and compute its coherence
# This function is from Kapadia (2019):https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0

####################### 别忘了改 passe =  30
####################### 别忘了改 passe =  30
####################### 别忘了改 passe =  30

def compute_coherence_values_basic(data,corpus,dictionary,k,matrix):
    # from gensim.models import CoherenceModel
    # build individual lda model
    # assuming that each document tends to cover multiple topics, thus the alpha should be symmetric
    # multiprocessing won't work on AWS Lambda because the execution environment/container is missing
    # Therefore cannot use gensim.models.LdaMulticore, only use gensim.models.LdaModel
    
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           alpha = 'symmetric',
                                           eta = matrix,
                                           random_state=5,
                                           passes=10)                  # 别忘了改10或者20或者30
    
    #p = lda_model.log_perplexity(corpus)
    # build coherence model
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                         texts=data, 
                                         dictionary=dictionary, 
                                         coherence='c_v')
    # get coherence score
    return coherence_model_lda.get_coherence()


# This function search the optimal hyperparameter settings for the lda model
# Similar to the grid search 
# It can take a long time to run
# This function is from Kapadia (2019):https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
def tuning_guided_lda_model(data,corpus,dictiornay, min_topics, max_topics, step_size, ori_matrix, ori_len, new_len):
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
    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # update eta_matrix
            eta_matrix = eta_matrix_updater( k, ori_len, new_len, ori_matrix)
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
    return model_results

# Function for getting the number of topics with the optimal coherence
# The default setting only searches optimal num_topic on the full corpus training set, 
# however, sometimes the coherence score might be higher on the 75% corpus training set, 
# in this case, user can change to "on_full_corpus=False" to search the optimal num_topic across training sets with different sizes
def get_optimal_topic_number(data, on_full_corpus=True):
    if on_full_corpus == True:
        data = data[data['Validation_Set']=='100% Corpus']
    data = data.loc[data['Coherence'].idxmax()]
    return data['Topics']

##################################################################
##   Functions for generating training reports and visulisation
##################################################################
# This function is for generating a PDF report for visualising topics info during the continuous training
# This function is partly from Fabian (2019): https://stackoverflow.com/questions/55628044/writing-a-string-to-pdf-with-pypdf2
# This function is partly from PyFPDF (2021): https://pyfpdf.readthedocs.io/en/latest/reference/multi_cell/index.html
def generate_topics_report(model, num_topics, coherence_score, top_n_words, now_time):
    
    # get top_n hot words words under each topic
    text = model.print_topics(num_words= top_n_words)
    # get timestamps
    report_time = now_time.strftime('%Y/%m/%d  %H:%M:%S')
    timestamp = now_time.strftime('%Y%m%d_%H%M%S')
    
    # create PDF
    pdf = FPDF()
    pdf.add_page() #create new page
    pdf.set_font("Arial", size=12) # font and textsize
    # print basic info
    pdf.cell(200, 5, txt="Model created at: {}".format(report_time), ln=1, align="L")
    pdf.cell(200, 5, txt="Optimal number of topics: {}".format(num_topics), ln=1, align="L")
    pdf.cell(200, 5, txt="Optimal coherence score: {}".format(coherence_score), ln=1, align="L")
    
    # print topics info
    pdf.multi_cell(200, 5, txt="\n", align="L")
    for pair in text:
        pdf.multi_cell(200, 5, txt="Topic number: {}".format(pair[0]), align="L")
        pdf.multi_cell(200, 5, txt="Top 20 hot words and probabilities: {}".format(pair[1]), align="L")
        pdf.multi_cell(200, 5, txt="\n", align="L")
    
    file_name = "model_training_report_at_{}.pdf".format(timestamp)
    # output pdf
    pdf.output(file_name)
    # return file name
    return file_name



#########################################################
## Functions for saving training results to S3
#########################################################
# Function for uploading training reports & visulisations (pdf, csv, html files) to S3
# This function is from Sophros (2020): https://stackoverflow.com/questions/61638940/save-a-gensim-lda-model-to-s3
def result_upload_helper(file_name, local_path, bucket_name):
    # get file path
    file_path = local_path + file_name        
    # create s3 resource
    s3_resource = boto3.resource('s3')
    # upload file
    s3_resource.meta.client.upload_file(file_path, bucket_name, file_name)
    print('successfully upload ' + file_name)

# save the model to model_path
def save_lda_model(model, model_name, save_path):
    # save the model to model_path
    model.save(save_path+'{}.model'.format(model_name))
    # get list of componenets
    components = [file for file in os.listdir(save_path) if file.startswith(model_name)]
    
    return components

# Function for uploading eta matrix and list of components to S3
# This function is from Shabani (2018): https://stackoverflow.com/questions/49120069/writing-a-pickle-file-to-an-s3-bucket-in-aws
def file_upload_helper(file, file_name, bucket_name):
    # create S3 resource
    s3_resource = boto3.resource('s3')
    # covert the file to pkl
    obj_pkl = pickle.dumps(file)
    obj_key = '{}.pkl'.format(file_name)
    s3_resource.Object(bucket_name, obj_key).put(Body=obj_pkl)
    print('Success')

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







#########################################
##             EXECUTION
#########################################    

# Create a S3 resource and client objects using boto3
s3_resource = boto3.resource('s3') # for getting bucket info
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    
    # change the working directory to /tmp
    os.chdir('/tmp')
    
    # read data
    # getting bucket info
    bucket_name = "data-ready-for-topic-modelling"
    
    # collecting data created in this full month
    # calculate time
    now_time = datetime.now()
    last_hour = datetime.now() - timedelta(hours = 1)
    used_time = last_hour.strftime('data_of_%Y%m')
    print(used_time)
    
    
    
    try:
        # Retrive processed data files created in last full month and merge them together into a dataframe
        # Notice that header is in the dataframe
        df = retrive_tweet_files(used_time, bucket_name)
        # if no sufficient data, break the lambda function
        if len(df) == 0:
            print("NO SUFFICIENT DATA") 
        
        # note that df['processed_tokens'] is now read as a string rather than a list object
        # so need to convert string to list object
        df['processed_tokens'] = df['processed_tokens'].apply(lambda lst: lst.strip("[]").replace("'","").split(", "))
        
        # keep useful contents
        df = df[['tweet_id','user_id','text','processed_tokens']]
        
        # Delete NA, drop if any value in the row has a nan
        df = df.dropna(how='any')  
        
        # reset the dataframe index
        df = df.reset_index()
        # drop old index column
        df = df.drop(['index'], axis=1)
        
        
        
        
        
        #########################################################################################
        ###      Download model component list and eta matrix of the production model
        #########################################################################################
        # location info
        bucket_name = "guided-lda-on-pre-collected-data"
        model_component_list_key = "components.pkl"
        eta_matrix_key = "eta_matrix_production_v1.pkl"
        
        # Download model component list and eta matrix
        components = file_download_helper(model_component_list_key, bucket_name)
        eta_matrix_ori = file_download_helper(eta_matrix_key, bucket_name)
        
        
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
        production_model = gensim.models.LdaMulticore.load(download_path + 'guided_lda_for_production.model')
        
        
        ##########################################################
        #       Updating word dictionary
        ##########################################################
        # get the pre-generated dictionary
        id2word = production_model.id2word
        
        # Update the dictionary with words in new corpus
        # The id of the original word will remain the same, 
        # and new words will get ids starting from the last number in the original dictionary
        id2word, ori_len, new_len = dictionary_updater(dictionary= id2word, data= df['processed_tokens'])
        
        #########################################################
        ###     Create bag of words (bow) corpus
        #########################################################
        
        # Create corpus that contains all documents
        texts = df['processed_tokens']

        # Create bag of word for each document in the corpus 
        # each bow contains the id of each word in that single document and its number of occurrences in that document 
        # (term id, term document Frequency)
        # NOTE: the corpus will only contain word_ids that are in the dictionary. 
        # As the dictionary was updated with new words, the corpus will contain new words (and assign default confidence 0.001)
        corpus = [id2word.doc2bow(text) for text in texts]
        
        ######################################################################################
        ##        Tuning the guided LDA model on the data collected in the last month
        ######################################################################################
        #       Find the optimal number of topics that has the highest coherence score
        ######################################################################################
        # Use defined functions to tune the guided lda model and find optimal hyperparameter settings
        # Notice that the number of new topics should be more than  12  (the num of rows in the production eta matrix)
        # It can take a long time to run
        model_results = tuning_guided_lda_model(data = df['processed_tokens'],
                                                corpus = corpus,
                                                dictiornay = id2word,
                                                min_topics = 12, 
                                                max_topics = 25,              #############改
                                                step_size = 1, 
                                                ori_matrix = eta_matrix_ori,
                                                ori_len = ori_len, 
                                                new_len = new_len)
        # Convert to the dataframe and save to the csv files
        # the result df will be saved to S3 later
        model_results_df = pd.DataFrame.from_dict(model_results)
        
        # searches optimal num_topic only on the full corpus training set
        num_topics = get_optimal_topic_number(data= model_results_df, on_full_corpus=True)
        print("optimal num_topics:  ", num_topics)
        
        ##########################################################################
        ##          Building the guided LDA model (with optimal parameter)
        ##########################################################################

        # update eta_matrix
        eta_matrix = eta_matrix_updater(num_topics, ori_len, new_len, eta_matrix_ori)
        # Build new guided LDA model
        # Remember to set the minimum_probability=0 in the model or can't get probabilities of a word under each topic
        guided_lda = gensim.models.LdaModel(corpus = corpus,
                                            id2word = id2word,
                                            num_topics = num_topics,
                                            passes = 20,                       # ########别忘了改
                                            alpha = 'symmetric',
                                            eta = eta_matrix,
                                            random_state=5,
                                            minimum_probability=0)
        # Compute the perplexity and coherence score of the model
        coherence_score = model_benchmarking(df['processed_tokens'], guided_lda, id2word, corpus)
        
        #####################################################
        #    Create the eta matrix for the new model
        #####################################################
        # Creating eta matrix with top 20 words under each topic
        # the eta matrix can be used to train the guided lda model as a prior belief on word probability 
        # can be use to assign probabilities for each word-topic combination
        eta_matrix_updates = create_eta_matrix(num_topics,20,guided_lda,id2word)
        
        #########################################################################
        #      Save the new model, its list of components and eta matrix to S3
        #########################################################################
        
        # define destination bucket
        dest_bucket = 'continuous-training-for-production-models'
        # generate timestamp
        timestamp = now_time.strftime('%Y%m%d_%H%M%S')
        # save the model to local tmp folder
        components_lst = save_lda_model(guided_lda, 'production_updates_{}'.format(timestamp), download_path)
        
        # Upload list of components and new eta matrix to S3
        file_upload_helper(file = components_lst, file_name = 'components_lst_{}'.format(timestamp), bucket_name = dest_bucket)
        file_upload_helper(file = eta_matrix_updates, file_name = 'eta_matrix_updates_{}'.format(timestamp), bucket_name = dest_bucket)
        
        # Upload training model to S3
        model_upload_helper(components_lst, download_path, dest_bucket)
        
        
        #########################################################################
        #      Generating training reports & visulisations, and upload to S3
        #########################################################################        
        
        # generate topics reports for visualisation
        pdf_name = generate_topics_report(model = guided_lda,
                                        num_topics = num_topics,
                                        coherence_score = coherence_score,
                                        top_n_words = 20,
                                        now_time = now_time)
        
        # create model tuning results csv                        
        csv_name = "model_training_report_at_{}.csv".format(timestamp)
        model_results_df.to_csv(csv_name)
        
        # create pyLDAvis graph for visualisation 
        # Visualize the topics 
        # lambda = 0.6 can be ideal
        html_name = "model_training_pyLDAvis_graphs_at_{}.html".format(timestamp)
        # note that different versions of pyLDAvis might have different syntax requirements
        LDAvis_prepared = gensim_models.prepare(guided_lda, corpus, id2word)
        pyLDAvis.save_html(LDAvis_prepared, html_name)
        
        # upload training reports & visulisations (pdf, csv, html files) to S3
        result_upload_helper(file_name = pdf_name, local_path = download_path, bucket_name = dest_bucket)
        result_upload_helper(file_name = csv_name, local_path = download_path, bucket_name = dest_bucket)
        result_upload_helper(file_name = html_name, local_path = download_path, bucket_name = dest_bucket)
        
        
        # os.listdir(download_path)
        # os.listdir(download_path), file_name
        # guided_lda.print_topics(num_words=10)
        # df.columns.values
        return True
    
    except Exception as e:
        print(e)
        #print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(file_key, bucket_name))
        #raise e        
        

