### 1.4 build: added feedback functionality

#How To run
#	1.	Install Streamlit
#	2.	open terminal (Anaconda Prompt), and locate the folder where this script + SessionState.py is
#	3.	run command: streamlit run "StreamLit UI v1.0.py"
#	4.	to close/ restart the script, press Ctrl + C in the terminal

import pandas as pd
import numpy as np
import re

import random

import streamlit as st
import SessionState

import time

import mysql.connector
import datetime

#might make sense to change function name to intent_matching
#from review_classification_v3 import chat_response
#from intent_classification import intent_classification
#from intent_classification import find_latest_index

#from parameter_search import parameter_collecter
#from parameter_search import NER_conversation
#from parameter_search import give_action
#from parameter_search import give_answer

#from prompt_generation import return_random_prompt

#from connect_ITU import connect_ITU_database
#from connect_ITU import write_line_to_table

#from Our_NER import Our_NER

#from tokenization import tokenize_df
#from tokenization import tokenize

from nltk.corpus import stopwords
from bpemb import BPEmb

from danlp.models import load_bert_ner_model

#SessionState saves variables that should NOT reset when Streamlit reruns the script (similar to cache)
# Conversation: List of strings which represent the conversation.
#  The first four characters in each string is a refference to the type of interaction:
#  -"You:" User input utterance
#  -"Bot:" Response from Bot that is visible to the user
#  -"Pol:" User is presented with one or more options as to what the user meant
#  -"Ver:" User has verified an option given through the Pol
# key: whenever the user submits an utterance, then key += 1
# utterances: list of strings containing all user utterances/ submissions
# intent: if the bot is currently collecting parameters (etc email, name,...) this = the intent. Otherwise None
# intent
ss = SessionState.get(conversation=[],key=0,utterances=[],intent=None,potential_parameters={},verified_info_dict={},feedback_given=0,scenario="")

#-----------------------------------------------------------------
### Connecting and writing to the ITU database ###
@st.cache
def connect_ITU_database(host,database,user,password):
	mydb = mysql.connector.connect(host=host,
	                               user=user,
	                               password=password,
	                               charset='utf8',
	                               database=database)
	mycursor = mydb.cursor()
	return mycursor,mydb

mycursor,mydb = connect_ITU_database(st.secrets["DB_HOST"],
	st.secrets["DB_NAME"],
	st.secrets["DB_USERNAME"],
	st.secrets["DB_PASSWORD"])

def write_line_to_table(mycursor,mydb,table_name,values,columns = ["id","conversation", "usefull", "comment","time"]):
	values = [None]+values
	ct = datetime.datetime.now()
	values += [ct]

	if len(columns) != len(values):
		print("columns and values are not the same length! Returning None")
		print(values)
		print(columns)
		return None
	
	sql = "INSERT INTO "+table_name+" ("+ ", ".join(columns) +") VALUES ("+  ", ".join(["%s"]*len(values))  +")"
	val = values
	mycursor.execute(sql, val)

	mydb.commit()

#-----------------------------------------------------------------
### Tokenization function

@st.cache
def load_bpemb():
	bpemb_da = BPEmb(lang="da", vs=3000)
	return bpemb_da

bpemb_da = load_bpemb()

try:
	stop_words = set(stopwords.words('danish'))
except:
	import nltk
	nltk.download('stopwords')
	from nltk.corpus import stopwords
	stop_words = set(stopwords.words('danish'))
	print("downloading stopwords")

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

#Recurring: if E.g. n_gram = 4, and recurring = True, it will also return n_gram = 3, 2, 1. E.g.
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

#tokenize a string given various parameters
def tokenize(string, bpemb = bpemb_da, n_grams = 2, recurring_n_grams = True, remove_stopwords = False):
    if isNaN(string):
        return []
    string = clean_string(string, remove_stopwords = remove_stopwords)
    string = bpemb.encode(" ".join(string))
    
    final_output = []
    final_output += n_gram_tokens(string, n_grams = n_grams, recurring = recurring_n_grams)
    
    return final_output

#tokenize all rows in a dataframe on a specified column
def tokenize_df(df,bpemb = bpemb_da, column = "training_phrases", n_grams = 2, recurring_n_grams = True, remove_stopwords = False):
    df[column] = df[column].apply(lambda x: tokenize(x, bpemb = bpemb, n_grams = n_grams, recurring_n_grams = recurring_n_grams, remove_stopwords = remove_stopwords))
    return df

#-----------------------------------------------------------------
### Loading bert and dataframes ###
#st.cache ensures that this function only runs once to improve runtime
@st.cache
def load_data():
	#load danlp bert model
	bert = load_bert_ner_model()

	#load dataframes (excel)
	df_training = pd.read_excel("Agent intent mapping.xlsx",sheet_name = 0)
	df_training = tokenize_df(df_training,column = "training_phrases")

	df_parameters = pd.read_excel("Agent intent mapping.xlsx",sheet_name = 1)
	df_actions = pd.read_excel("Agent intent mapping.xlsx",sheet_name = 2)

	df_prompt = pd.read_excel("prompt-generation.xlsx")

	return bert, df_training, df_parameters, df_actions, df_prompt

bert, df_training, df_parameters, df_actions, df_prompt = load_data()

#-----------------------------------------------------------------
### misc helper functions
def rerun():
    raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))

def isNaN(num):
    return num != num

def string_contains(string,words):
	string = re.sub("[,.!?#]", "", string)
	for token in string.split(" "):
		if token.lower() in words:
			return True
	return False

#return the index of the last (right-most) "needle" (string) in haystack (list)
def find_latest_index(needle,haystack):
    if needle in haystack:
        return - (haystack[::-1].index(needle) +1)
    else:
        return None

#-----------------------------------------------------------------
### NER function using re and bert
def Our_NER(line, bert):

    tokens, labels = bert.predict(line)
    tekst_tokenized = tokens
    predictions = bert.predict(tekst_tokenized, IOBformat=False)

    info_dict = {}

    for i in range(len(predictions['entities'])):
        if predictions['entities'][i]['type'] == 'PER':
            if 'Name' not in info_dict.keys():
                info_dict['Name'] = []
            
            name_list = predictions['entities'][i]['text'].split()
            new_name = ''
            for name in name_list:
                new_name += " " +name.capitalize()
            
            info_dict['Name'].append(new_name.strip())

        if predictions['entities'][i]['type'] == 'LOC':
            if 'Location' not in info_dict.keys():
                info_dict['Location'] = []
            
            loc_list = predictions['entities'][i]['text'].split()
            new_loc = ''
            for loc in loc_list:
                loc = loc.replace('#','')
                if loc.isnumeric():
                    if new_loc[-3:].isnumeric():
                        new_loc += loc.capitalize()
                    else:
                        new_loc += " " +loc.capitalize()
                else:
                    new_loc += " " +loc.capitalize()
            info_dict['Location'].append(new_loc.strip())

            
    loc2 = re.findall(r'[0-9]{4}',line)
    if len(loc2) and 'Location' not in info_dict.keys():
        info_dict['Location'] = []
        for loc in loc2:
            info_dict['Location'].append(loc)
    
    
    email_match = re.findall(r'[\w\.-]+@[\w\.-]+', line)
    if len(email_match) >0:
	    for email in email_match:
	        if 'email' not in info_dict.keys():
	            info_dict['email'] = []
	        info_dict['email'].append(email)

    phone_match = re.findall(r"((\(?\+45\)?)?)(\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{2})$",line)
    if len(phone_match) > 0:
	    for num in phone_match[0]:
	        num = num.replace(' ','')
	        if len(num) == 8:
	            if 'Phone Number' not in info_dict.keys():
	                info_dict['Phone Number'] = []
	            info_dict['Phone Number'].append(num)
	             
    return info_dict
#-----------------------------------------------------------------
### prompt generating function
def return_random_prompt(df): #df_prompt
	er = [x for x in df["er"] if str(x) != 'nan']
	fra = [x for x in df["fra"] if str(x) != 'nan']
	har = [x for x in df["har"] if str(x) != 'nan']
	fordi = [x for x in df["fordi"] if str(x) != 'nan']

	prompt = "Du er {} og kommer fra {}. Du har {}. Du skriver til Fitness Worlds chatbot fordi {}.".format(
	random.choice(er),random.choice(fra),random.choice(har),random.choice(fordi))
	return prompt

#-----------------------------------------------------------------
### Lookup functions to find the appropriate response

#looks at the entire conversation "history" and looks for parameters about the user
def NER_conversation(conversation,bert_model):
    user_dialog = [msg[5:] for msg in conversation if msg[:4] == "You:"]
    return Our_NER(". ".join(user_dialog),bert_model)

# When the intent is specified, this function checks if the intent requires any parameters
def check_parameters(intent,df_parameters):
    #if you need to collect parameters, return then as a list
    #otherwise return False
    parameters = df_parameters[df_parameters["Intent"] == intent]["parameters"]
    parameters = parameters.to_string(index=False)#[1:]
    print(parameters)
    if str(parameters) ==  "NaN":
        return False
    else:
        return parameters[1:].split(",")

# When you are done with collecting all parameters, the "answer" column for a specific intent is returned
def give_answer(intent,df_parameters,column = "Answer"):
    answer = df_parameters[df_parameters["Intent"] == intent][column]
    print(df_parameters[df_parameters["Intent"] == intent])
    #print(intent)
    #print(answer)
    return "Bot:" + answer.values[0]

# return parameters if any, otherwise give the answer straight away
def parameter_collecter(intent,df_parameters):
    parameters = check_parameters(intent,df_parameters)
    print(parameters)
    if parameters:
        return "Par:," + ",".join(parameters)
    else:
        return give_answer(intent,df_parameters)

# lookups to find the appropriate way to ask the user to specify a parameter
# type: "ask","verify" or "options"
def give_action(parameter,action,df_actions):
    #if you need to collect parameters, return then as a list
    #otherwise return False
    action = df_actions[df_actions["parameter"] == parameter][action]
    action = action.to_string(index=False)[1:]
    if action ==  "NaN":
        return False
    else:
        return action
#-----------------------------------------------------------------
### The Naive bayes language model. Training and prediction functions

# probabilities helper function
# The input will more often than not be summed log  values (negative) for each class
# therefore we ue softmax.
# x: one-demensional array of numbers
def probabilities(x):
	x = list(x)
	return np.exp(x)/sum(np.exp(x))


# function that trains the model using naive bayes
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

# predicting the most likely class for the input, X_test
# X_test: list of tokens to be classified. so pre-cleaned
# return_dict: if False, only return the most likely class.
# if True, return a dict with all classes and their respective probabilities
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


# this is the threshold we chose for the minimum probability that and intent or group
# has to meet for it to be presented to the user.
threshold = 0.1

#function that returns a predicted intent based on the prior conversation
#it can return multiple intents based on if they meet the threshold set above
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

#-----------------------------------------------------------------
def bot_response(conversation):

	latest_message = conversation[-1][4:]
	latest_message_type = conversation[-1][:4]

	#E.g  conversation_types= ["Bot:","You:","Bot:",...]
	
	if latest_message_type == "You:" and string_contains(latest_message,["ikke","nej","forkert"]):
		ss.intent = None

	if ss.intent == None:
		if latest_message_type == "You:" or latest_message_type == "Ans:":
			response = intent_classification(df_training,conversation)

			if isinstance(response, str) == False and response != None:
				
				for r in response:
					if r in ["Unknown","small-talk"]:
						return give_answer("Unknown",df_parameters)

				present_choices(response)
				return "Bot: Jeg er desværre ikke helt sikker på jeg forstår. Skrev du om {}?".format(" eller ".join(response))
			else:
				no_varification_intents = ["hilsen","farvel","rephrase question"]
				if response not in no_varification_intents:
					present_choices([response])
					return "Bot: Vil du vide mere om {}? Så klik på knappen nedenfor!".format(response)
				else:
					return parameter_collecter(response,df_parameters)

		elif ss.intent == None and latest_message_type == "Ver:":
			#print(response)
			ss.intent = latest_message
			#if this is true, we have verified the users intent, and can move on to parameters
			required_parameters = parameter_collecter(ss.intent,df_parameters)
			print(required_parameters)
			if required_parameters[:4] == "Bot:":
				ss.intent = None
				ss.info_dict = None
				#if this is true, the default answer to the intent will be returned (no parameters needed)
				return required_parameters

			elif required_parameters[:4] == "Par:":
				#prompt the intent to be != None. This will activate if statement further down
				ss.intent = latest_message
				#log in conversation
				ss.conversation += [required_parameters]
				#search for existing parameters in prior conversation
				ss.potential_parameters = NER_conversation(conversation, bert)

	if ss.intent != None:
		conversation_types = [msg[:4] for msg in ss.conversation]
		required_parameters = ss.conversation[find_latest_index("Par:",conversation_types)].split(",")[1:]
		print(required_parameters)

		if conversation_types[-1] == "You:" or conversation_types[-1] == "Ver:":
			for i in range(len(required_parameters)):
				if required_parameters[i] not in ss.verified_info_dict.keys():
					ss.verified_info_dict[required_parameters[i]] = ss.conversation[-1][4:]
					break

		if ss.potential_parameters != {}:
			pass
		for par in required_parameters:
			if par not in ss.verified_info_dict.keys():
				if par in ss.potential_parameters.keys():
					present_choices([ss.potential_parameters[par][0]])
					return "Bot:"+give_action(par,"verify",df_actions)+" "+ss.potential_parameters[par][0]
				else:
					return "Bot:"+give_action(par,"ask",df_actions)

	intent_copy = str(ss.intent)
	ss.intent = None
	return give_answer(intent_copy,df_parameters, column="completed") #give the final answer to the intent!


def print_messages(conversation):
	#time.sleep(4)

	if len(conversation) == 0:
		return
	for message in conversation:
		if message[:4] == "You:":
			#chat_container.info(message) 
			chat_container.markdown('<p class="chat-right-name">{}</p>'.format("Mig"), unsafe_allow_html=True)
			chat_container.markdown('<p class="chat-right">{}</p>'.format(message[4:]), unsafe_allow_html=True)
		elif message[:4] == "Bot:":
			#time.sleep(4)
			#chat_container.info(message)
			chat_container.markdown('<p class="chat-left-name">{}</p>'.format("Chad Bot"), unsafe_allow_html=True)
			chat_container.markdown('<p class="chat-left">{}</p>'.format(message[4:]), unsafe_allow_html=True)
		elif message[:4] == "Ans:" or message[:4] == "Ver:":
			chat_container.markdown('<p class="chat-notification">Du valgte -{}-</p>'.format(message[4:]), unsafe_allow_html=True)

	#st.sidebar.info("{}".format("---".join(ss.conversation)))

	if len(conversation) > 1:
		if conversation[-2][:4] == "Pol:":
			choice_list = conversation[-2].split(",")[1:]
			present_choices(choice_list)
		elif conversation[-2][:4] == "You:":
			col1.empty()
			col2.empty()
			col3.empty()


#this function create the "selection boxes" that the user can chose to click on
def present_choices(choice_list):
	placeholders = [col1,col2,col3]
	
	choice1 = False
	choice2 = False
	choice3 = False

	if len(choice_list) > 0:
		choice1 = placeholders[0].button(str(choice_list[0]))
	if len(choice_list) > 1:
		choice2 = placeholders[1].button(str(choice_list[1]))
	if len(choice_list) > 2:
		choice3 = placeholders[2].button(str(choice_list[2]))

	if ss.conversation[-2][:4] != "Pol:":
		temp_conversation = ss.conversation
		temp_conversation.append("Pol:"+","+",".join(choice_list))
		ss.conversation = temp_conversation

	if choice1:
		if len(choice_list) < 2:
			ss.conversation += ["Ver:{}".format(str(choice_list[0]))]
		else:
			ss.conversation += ["Ans:{}".format(str(choice_list[0]))]
		prev_conversation = ss.conversation
		prev_conversation.append(bot_response(prev_conversation))
		ss.conversation = prev_conversation
		rerun()
	if choice2:
		ss.conversation += ["Ans:{}".format(str(choice_list[1]))]
		prev_conversation = ss.conversation
		prev_conversation.append(bot_response(prev_conversation))
		ss.conversation = prev_conversation
		rerun()
	if choice3:
		ss.conversation += ["Ans:{}".format(str(choice_list[2]))]
		prev_conversation = ss.conversation
		prev_conversation.append(bot_response(prev_conversation))
		ss.conversation = prev_conversation
		rerun()


def main():
	
	global chat_container, developer_mode, col1, col2, col3
	#chat_empty = st.empty() 

	#markdown test - delete later
	st.markdown('<style>' + open('styles - v2.css').read() + '</style>', unsafe_allow_html=True)
	#st.markdown('<i class="chat-left">How can I help you?</i>', unsafe_allow_html=True)
	#st.markdown('<i class="chat-right">Bbrr Brrr Whats good</i>', unsafe_allow_html=True)
	#st.markdown('<i class="chat-left">Brooo fantastic</i>', unsafe_allow_html=True)

	st.header("Fitness World Chatbot")
	st.markdown("""Hej! Tak fordi du vil hjælpe os (Filip og Simon) med vores bachelor-projekt!""")
	st.markdown("""**Formalia:** Intet af det, vores chatbot skriver den gør, har rent faktisk nogen effekt i den virkelig verden! Det er
		udelukkende en prototype der skal ses som et udkast til hvordan en eventuel chatbot
		kunne se ud/aggere!""")
	st.markdown("""*'Giv os Feedback'*-sektionen lagrer den samtale du har ført med 
		vores chatbot op til det tidspunkt hvor du trykker *'Send Feedback'*. Alt information vil blive lagret 
		annonymt og ikke tilgængeligt for offentligheden. Vi forbeholder os dog retten til at bruge 
		enkeltstående eksempler i eksamensprojektet og den mundtlige præsentation, såfremt det findes relevant.
		For at være helt sikre på at ingen personfølsom data slipper ud, vil vi dog stadig opfordre til at man ikke skriver sit eget fulde navn, email adresse eller lign., men 
		i stedet opfinder en fiktiv karakter som man kan skrive ud fra, eller bruger 'giv scenarie' funktionen.""")
	st.markdown("""Du kan se en komplet liste af chatbottens funktionaliteter i sidepanelet til venstre.
		Du kan trykke på 'giv scenarie' for at få inspiration til hvad du evt ka skrive til vores Chatbot""")
	st.markdown("""Det er ret væsentligt for os at høre jeres feedback om produktet. Derfor håber vi at i
		vil give en thumbs up/down i Feedback sektionen. Hvis i vil prøve flere scenarier af, så efterlad gerne feedback efter hver gang. Tak :)""")

	st.markdown("Hvis du oplever fejl kan du også altid trykke på 'Reset Session State' for at nultille programmet.")

	if st.button("giv scenarie"):
		ss.scenario = return_random_prompt(df_prompt)
		rerun()

	st.markdown("**"+ss.scenario+"**")
	st.markdown("")

	#write conversation to this box
	chat_container = st.beta_container()

	#placeholder for options
	col1_temp, col2_temp, col3_temp = st.beta_columns(3)
	# empty() makes the columns reset right before new content is sent to them
	col1 = col1_temp.empty()
	col2 = col2_temp.empty()
	col3 = col3_temp.empty()

	#Sidebar settings
	st.sidebar.subheader("Indstillinger")

	developer_mode = st.sidebar.checkbox("Developer Mode On/Off")

	print_messages(ss.conversation)
	if developer_mode:
		st.sidebar.success(ss.intent)
		st.sidebar.info("Conversation: "+"\n --".join(ss.conversation))
		st.sidebar.info("Collected user-parameters: "+str(ss.verified_info_dict))
		st.sidebar.info("utterances: "+str(ss.utterances))


	if st.sidebar.button("Reset Session State"):
		ss.conversation = []
		ss.intent=None
		ss.verified_info_dict={}
		ss.utterances=[]
		ss.feedback_given = 0
		ss.scenario = ""
		rerun()

	functionalities = set(list(df_training["intent"]))

	st.sidebar.markdown(""" Understøttede funktioner: 
- {}""".format("\n- ".join(functionalities)))

	#placeholders for layout
	ta_placeholder = st.empty()
	submit_placeholder = st.empty()
	st.subheader("Giv os din feedback")
	feedback_text_placeholder = st.empty()
	feedback_submit_placeholder = st.empty()

	#feedback section
	if ss.feedback_given==0:
		dummy1,col1_submit, col2_submit,dummy2 = feedback_text_placeholder.beta_columns(4)

		if col1_submit.button("👍"):
			ss.feedback_given = 1
			rerun()
		if col2_submit.button("👎"):
			ss.feedback_given = -1
			rerun()

	elif ss.feedback_given == 1 or ss.feedback_given == -1:
		feedback_text_area = feedback_text_placeholder.text_area("Overordnet set, hvor godt håndterede chatbotten så din forespørgsel? Var der særlige ting den håndterede mindre godt?",value="")
		if feedback_submit_placeholder.button("Send Feedback"):
			comment = feedback_text_area
			usefull = ss.feedback_given
			try:
				write_line_to_table(mycursor,mydb,"user_logs",["--".join(ss.conversation),usefull,comment])
			except:
				mycursor,mydb = connect_ITU_database(st.secrets["DB_HOST"],
					st.secrets["DB_NAME"],
					st.secrets["DB_USERNAME"],
					st.secrets["DB_PASSWORD"])
				write_line_to_table(mycursor,mydb,"user_logs",["--".join(ss.conversation),usefull,comment])
			ss.feedback_given = 99
			rerun()

	elif ss.feedback_given == 99:
		feedback_text_placeholder.success("Tak for din feedback!")

	#Welcome message print
	if len(ss.conversation) == 0:
		#time.sleep(1)
		welcome_message = "Bot: Hej, mit navn er Chad. Hvordan kan jeg hjælpe dig?"
		ss.conversation = [welcome_message]
		print_messages(ss.conversation)

	#chat text_area
	text_area = ta_placeholder.text_area("Skriv din besked",value="",key=ss.key)
	if text_area != "":
		ss.utterances += [text_area]

	#chat submit button
	submit_button = submit_placeholder.button("Send")
	if submit_button:
		if text_area == "":
			rerun()
		utterance = ss.utterances[-1]
		ss.key += 1
		text_area = ta_placeholder.text_area("Skriv din besked",value="",key=ss.key)

		prev_conversation = ss.conversation
		prev_conversation.append("You: "+ utterance)
		ss.conversation = prev_conversation

		print_messages([ss.conversation[-1]])

		#time.sleep(1)

		prev_conversation = ss.conversation

		prev_conversation.append(bot_response(prev_conversation))

		ss.conversation = prev_conversation

		print_messages([ss.conversation[-1]])
		#print_messages(ss.conversation)
		rerun()




if __name__ == '__main__':
	main()