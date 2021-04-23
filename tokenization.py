#!/usr/bin/env python
# coding: utf-8

# In[82]:
import re
import copy
import pandas as pd

from nltk.corpus import stopwords
stop_words = set(stopwords.words('danish'))

from bpemb import BPEmb
bpemb_da = BPEmb(lang="da", vs=3000)

# In[91]:


def isNaN(num):
    return num != num


# In[2]:


#Tokenizes, removes characters and potentially also stopwords
def clean_string(text, remove_stopwords = True):
    text = text.replace('\\n','')
    
    text = text.lower()
    text  = re.sub('[\W^\d^]+', ' ', text)
    
    text = text.split(' ')
    text = [word for word in text if len(word) > 0]
    
    if remove_stopwords:
        text = [word for word in text if word not in stop_words]
    return text


# In[60]:


#Tokens is a list structure of words. 
#Recurring: if E.g. n_gram = 4, and recurring = True, it will also return n_gram = 3, 2, 1.
def n_gram_tokens(tokens, n_grams = 2, recurring = False):
    if len(tokens) == 0:
        return tokens
    
    tokens_copy = tokens.copy()
    
    if n_grams < 1 or len(tokens_copy)+2 < n_grams:
        return []
    
    if recurring:
        output = n_gram_tokens(tokens_copy, n_grams = n_grams - 1, recurring = True)
    else:
        output = []
    
    if n_grams == 1:
        return tokens_copy
    
    tokens_copy = ["<START>"] + tokens + ["<STOP>"]
    
    for idx, word in enumerate(tokens_copy):
        if idx + n_grams > len(tokens_copy):
            break
        temp_n_gram = []
        for i in range(n_grams):
            temp_n_gram.append(tokens_copy[i+idx])
        
        output.append("".join(temp_n_gram))
    return output


# In[61]:


#slides a "window" of size n over each word in the text and saves it at each timestep
#happens before
def n_char_tokens(tokens, window_size = 5):
    #ignore = ["<START>","<STOP>"]
    concat_string = " ".join(tokens)
    
    output = []
    for i in range(len(concat_string)- window_size +1):
        output.append(concat_string[i:i+window_size])

    return output


# In[62]:


# NOT COMPLETELY DONE
# might not work that well in danish, due to the nature of how sentences are structured?
def NOT_(tokens, negation = False):

    negation_set = set(["ikke","ej"])
    negation_set_inverse = set(["man","men","dog",".","!","?","selvom","trods","<stop>","<start>"])
    new_tokenized = []
    
    for token in tokens:
        
        if " " in token:
            expanded_ngrams = NOT_(token.split(" "),negation)
            new_tokenized.append(" ".join(expanded_ngrams))
            if expanded_ngrams[-1][:4] == "NOT_":
                negation = True
        
        elif token.lower() in negation_set:
            negation = True
            new_tokenized.append(token)
        elif token.lower() in negation_set_inverse:
            negation = False
            new_tokenized.append(token)
        else:
            if negation == True:
                new_tokenized.append("NOT_" + token)
            else:
                new_tokenized.append(token)

    return new_tokenized


# In[92]:


def tokenize(string, bpemb = bpemb_da, n_grams = 2, recurring_n_grams = True, remove_stopwords = False):
    if isNaN(string):
        return []
    string = clean_string(string, remove_stopwords = remove_stopwords)
    
    string = bpemb.encode(" ".join(string))
    
    final_output = []
    
    final_output += n_gram_tokens(string, n_grams = n_grams, recurring = recurring_n_grams)
    
    return final_output


# In[93]:


#test_string = "Hvornår åbner I efter nedlukningen?"

#print(tokenize(test_string, n_grams = 3, recurring_n_grams = True, remove_stopwords = True, n_char = 6))


# In[94]:


def tokenize_df(df,bpemb = bpemb_da, column = "training_phrases", n_grams = 2, recurring_n_grams = True, remove_stopwords = False):
    df[column] = df[column].apply(lambda x: tokenize(x, bpemb = bpemb, n_grams = n_grams, recurring_n_grams = recurring_n_grams, remove_stopwords = remove_stopwords))
    return df