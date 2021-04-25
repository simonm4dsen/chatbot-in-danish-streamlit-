#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from tokenization import tokenize_df
from tokenization import tokenize

from StreamLit_UI import read_file_to_df

from Our_NER import Our_NER

# Our_NER returns an info_dict:
# info_dict['Phone Number'] = []
# info_dict['Email'] = []
# info_dict['Location'] = []
# info_dict['Name'] = []

# add down the line:
# info_dict['MemberID'] = []

# alternated info_dict in Streamlit:


# In[2]:

path = ""#r"data"
file = "Agent intent mapping.xlsx"

df_training = read_file_to_df(path,file, sheet_name = 0)
df_parameters = read_file_to_df(path,file, sheet_name = 1)
df_actions = read_file_to_df(path,file, sheet_name = 2)

#tokenize training data - set parameters for tokenization here!
df_training = tokenize_df(df_training,column = "training_phrases")



def NER_conversation(conversation,bert_model = bert):
    user_dialog = [msg[5:] for msg in conversation if msg[:4] == "You:"]
    return Our_NER(". ".join(user_dialog),bert_model)


# In[24]:


def check_parameters(intent,df_parameters = df_parameters):
    #if you need to collect parameters, return then as a list
    #otherwise return False
    parameters = df_parameters[df_parameters["Intent"] == intent]["parameters"]
    parameters = parameters.to_string(index=False)[1:]
    if parameters ==  "NaN":
        return False
    else:
        return parameters.split(",")



# In[70]:


# When you are done with collecting all parameters
def give_answer(intent,df_parameters = df_parameters,column = "Answer"):
    answer = df_parameters[df_parameters["Intent"] == intent][column]
    print(df_parameters[df_parameters["Intent"] == intent])
    print(intent)
    print("yolo")
    print(answer)
    return "Bot:" + answer.values[0]


# In[73]:


def parameter_collecter(intent,):#,info_dict, verification_dict,df_parameters = df_parameters,df_actions = df_actions):
    parameters = check_parameters(intent)
    print(parameters)
    if parameters:
        return "Par:," + ",".join(parameters)
    else:
        return give_answer(intent)


# In[ ]:
# type: "ask","verify" or "options"
def give_action(parameter,action,df_actions = df_actions):
    #if you need to collect parameters, return then as a list
    #otherwise return False
    action = df_actions[df_actions["parameter"] == parameter][action]
    action = action.to_string(index=False)[1:]
    if action ==  "NaN":
        return False
    else:
        return action