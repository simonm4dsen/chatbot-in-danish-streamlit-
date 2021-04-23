#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re

from sklearn.model_selection import LeaveOneOut


from tokenization import tokenize_df
from tokenization import tokenize

import nltk
stopwords = nltk.corpus.stopwords.words('danish')


# In[34]:


threshold = 0.3


# In[2]:


#encoding = "utf-8"
#encoding = "ISO-8859-1"
def read_file_to_df(path, file, sep=";", encoding = "ISO-8859-1",sheet_name=0):
    file_type = file.split(".")[-1]
    if file_type == "xlsx":
        return pd.read_excel(path+"\\"+file,sheet_name=sheet_name)
    else:
        return pd.read_csv(path+"\\"+file, sep=sep, encoding = encoding) #low_memory=False


# In[3]:

#The input will more often than not be summed log  values (negative) for each class
#therefore we ue softmax
# x: one-demensional array of numbers
def probabilities(x):
	x = list(x)
	return np.exp(x)/sum(np.exp(x))
#    sum_list = sum(list)
#    if sum_list == 0:
#        print("probablities too small!")
#        return [x for x in list]
#    return [x/sum_list for x in list]


# In[4]:


#path = r"C:\Users\simo7\Documents\GitHub\bachelor_project\Data"
#file = "Agent intent mapping.xlsx"

#df = read_file_to_df(path,file)


# In[8]:


#df = tokenize_df(df,column = "training_phrases",
#                 n_grams = 2, recurring_n_grams = True, 
#                 remove_stopwords = True, n_char = 6)


#X = np.array(df["training_phrases"])
#y = np.array(df["intent"])


# X_train: matrix type/ list of lists with tokenized text
# Y_train: list of class/categories
# c_p: Class probabilities - add if you have an initial probability distribution, otherwise get it from the data
# c_p can also be = "equal", which is just a uniform distribution across each class

def naive_bayes_probabilities(X_train, y_train,classes_from_data=True, c_p = "equal"):
    if classes_from_data==True:
    	classes = set(y_train)
    
    word_count = {c:{} for c in classes} #c:{word:count}
    class_count = {c:0 for c in classes}
    
    for idx,tokens in enumerate(X_train):
        class_count[y_train[idx]] += 1
        for word in tokens:
            if word not in word_count[y_train[idx]].keys():
                word_count[y_train[idx]][word] = 1
            else:
                word_count[y_train[idx]][word] += 1
                    
    if c_p == None:
        #calculate probability of Class
        sum_class_count = sum(class_count.values())
        c_p = {c:class_count[c]/sum_class_count for c in classes}
    elif str(c_p) == "equal":
        c_p = {c:1/len(classes) for c in classes}

    #start working towards getting probabilty of word given class
    #computes total number of words/ tokens (non-unique) for each class
    total_words_count = {c:sum(word_count[c].values()) for c in classes}
    
    #computes the number of unique words/tokens across all classes (the entire training set)
    unique_words = set()
    for c in classes:
        unique_words.update(word_count[c].keys())
    unique_words_count = len(unique_words)
    
    
    P_word_given_class = {c:{} for c in classes}
    
    for c in classes:
        for word in word_count[c].keys():
            P_word_given_class[c][word] = (word_count[c][word]+1) / (total_words_count[c]+unique_words_count)
            # the +1 is for laplacian smoothing, since we add the OOV token
    
    # add Out Of Vocabulary (OOV)
    for c in classes:
        P_word_given_class[c]["OOV"] = 1 / (total_words_count[c]+unique_words_count)
    
    #print(word_count)
    #print(P_category)
    #print("---")
    #print(P_word_given_category)
    
    return c_p, P_word_given_class

#

#c_p, P_word_given_class = naive_bayes_probabilities(X, y)


# X_test: list of tokens to be classified. so pre-cleaned
def naive_bayes_predict(X_test, c_p, P_word_given_class, return_dict = False):
    
    #print(sentence)
    
    classes = P_word_given_class.keys()
    
    P_dict = {}
    for c in classes:
        temp_P = np.log(c_p[c])
        
        for word in X_test:
            if word not in P_word_given_class[c].keys():
            	#note: we take the log() of each probability and sum to prevent underflow erros
                temp_P = temp_P + np.log(P_word_given_class[c]["OOV"])
            else:
                temp_P = temp_P + np.log(P_word_given_class[c][word])
        
        P_dict[c] = temp_P
        #print(category)
        #print(temp_P)
    
    #convert into class probabilities where they sum = 1. **consider swapping for softmax**
    class_probabilities = probabilities(P_dict.values())
    
    for idx,key in enumerate(P_dict.keys()):
        P_dict[key] = class_probabilities[idx]
    
    #sort output
    sorted_P_dict = dict(sorted(P_dict.items(), key=lambda item: item[1],reverse=True))
    
    #for i in range(print_top):
    #    print("{:.2f}% {}".format(list(sorted_P_dict.values())[i]*100,list(sorted_P_dict.keys())[i]))
    
    if return_dict:
        # return entire prediction probability distribution
        return sorted_P_dict
    else:
        #return highest_p_class
        return list(sorted_P_dict.keys())[0], sorted_P_dict[list(sorted_P_dict.keys())[0]]


# In[42]:


#return the index of the last (right-most) "needle" (string) in haystack (list)
def find_latest_index(needle,haystack):
    if needle in haystack:
        return - (haystack[::-1].index(needle) +1)
    else:
        return None

#category = "category"
#remove_stopwords = True
#n_grams = 2
#recurring_n_grams = True
#n_char = 6


# In[41]:

#assumes that the input df has column "trainingphrases", which is already tokenized
# it assumes that the "conversation" list is NOT tokenized
def intent_classification(df,conversation):
    X = np.array(df["training_phrases"])
    y = np.array(df["group"])
    
    # conversation_types: list of reciever/sender in the conversation timeline. E.g ["Bot:","You:","Bot:",...]
    conversation_types = [msg[:4] for msg in conversation]
    latest_msg = conversation[-1][4:]
    # if client has typed something
    if conversation_types[-1] == "You:":
        # Train the model on 'group' level- idealy change this so that it loads-pretrained model?
        c_p, P_word_given_class = naive_bayes_probabilities(X, y)
        
        predictions = []
        # make prediction - OBS here it tokenizes the input
        prediction_dict = naive_bayes_predict(tokenize(latest_msg), c_p, P_word_given_class, return_dict = True)
        for c in prediction_dict.keys():
            # OBS! This is where we check which classes are above the set threshold for being presented to the user!
            if prediction_dict[c] >= threshold:
                predictions.append(c)
        
        if len(predictions) == 0:
            return "rephrase question"
        if len(predictions) > 1:
            return predictions
        else:
            
            if predictions[0] == "Unknown":
                return "rephrase question"
            # The data is now only considering intents whithin a specific 'group'
            X = np.array(df[df["group"] == predictions[0]]["training_phrases"])
            y = np.array(df[df["group"] == predictions[0]]["intent"])
            
            # Train model again, this time only based on intents within the defined group
            c_p, P_word_given_class = naive_bayes_probabilities(X, y)
            
            # make prediction, this time, only pick the most probable (return_dict = False)
            # **consider if client should be able to get a multiple-chhoice for the specific intent as well**
            # **this will depend on how many intents within each group we plan to have **
            prediction = naive_bayes_predict(tokenize(latest_msg), c_p, P_word_given_class, return_dict = False)

            return prediction[0]

    # if client has answered a multiple-choice pol regarding 'group'
    elif conversation_types[-1] == "Ans:":
        # find the last time the client typed a message - this forms basis for prediction
        latest_msg_you = conversation[find_latest_index("You:",conversation_types)]
        
        #training data focused to the group the client chose
        X = np.array(df[df["group"] == latest_msg]["training_phrases"])
        y = np.array(df[df["group"] == latest_msg]["intent"])
        
        # train model
        c_p, P_word_given_class = naive_bayes_probabilities(X, y)
    
        prediction = naive_bayes_predict(tokenize(latest_msg_you), c_p, P_word_given_class, return_dict = False)
        
        return prediction[0]
