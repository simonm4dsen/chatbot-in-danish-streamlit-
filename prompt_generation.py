#!/usr/bin/env python
# coding: utf-8

# # Prompt generation
# 

# In[33]:


import pandas as pd
import random


# In[34]:


def read_file_to_df(path, file, sep=";", encoding = "ISO-8859-1",sheet_name = 0):
	file_type = file.split(".")[-1]
	if file_type == "xlsx":
		if path == "":
			return pd.read_excel(file,sheet_name = sheet_name)
		else:
			return pd.read_excel(path+"\\"+file,sheet_name = sheet_name)
	else:
		return pd.read_csv(path+"\\"+file, sep=sep, encoding = encoding) #low_memory=False

path = ""#r"data"
file = "prompt-generation.xlsx"

df = read_file_to_df(path,file)


# In[35]:


er = [x for x in df["er"] if str(x) != 'nan']
fra = [x for x in df["fra"] if str(x) != 'nan']
har = [x for x in df["har"] if str(x) != 'nan']
fordi = [x for x in df["fordi"] if str(x) != 'nan']


# In[36]:


def return_random_prompt():
	prompt = "Du er {} og kommer fra {}. Du har {}. Du skriver til Fitness Worlds chatbot fordi {}.".format(
	random.choice(er),random.choice(fra),random.choice(har),random.choice(fordi))
	return prompt


# In[37]:


#return_random_prompt()


# In[ ]:




