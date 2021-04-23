# 1.4 build: added feedback functionality

#How To run
#	1.	Install Streamlit
#	2.	open terminal (Anaconda Prompt), and locate the folder where this script + SessionState.py is
#	3.	run command: streamlit run "StreamLit UI v1.0.py"
#	4.	to close/ restart the script, press Ctrl + C in the terminal

import pandas as pd
import numpy as np
import re

import streamlit as st
import SessionState

import time

#might make sense to change function name to intent_matching
#from review_classification_v3 import chat_response
from intent_classification import intent_classification
from intent_classification import find_latest_index
from parameter_search import parameter_collecter
from parameter_search import NER_conversation
from parameter_search import give_action
from parameter_search import give_answer

from prompt_generation import return_random_prompt

from connect_ITU import connect_ITU_database
from connect_ITU import write_line_to_table

#from Our_NER import Our_NER

from tokenization import tokenize_df
from tokenization import tokenize


#SessionState saves variables that should NOT reset when Streamlit reruns the script
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


#---- Replace this part with a connection to mySQL instead ------
mycursor,mydb = connect_ITU_database(st.secrets["DB_HOST"],
	st.secrets["DB_NAME"],
	st.secrets["DB_USERNAME"],
	st.secrets["DB_PASSWORD"])

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
file = "Agent intent mapping.xlsx"

df_training = read_file_to_df(path,file, sheet_name = 0)
df_parameters = read_file_to_df(path,file, sheet_name = 1)
df_actions = read_file_to_df(path,file, sheet_name = 2)

#tokenize training data - set parameters for tokenization here!
df_training = tokenize_df(df_training,column = "training_phrases")

#-----------------------------------------------------------------

def rerun():
    raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))


def string_contains(string,words):
	string = re.sub("[,.!?#]", "", string)
	for token in string.split(" "):
		if token.lower() in words:
			return True
	return False


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
						return give_answer("Unknown")

				present_choices(response)
				return "Bot: Jeg er desværre ikke helt sikker på jeg forstår. Skrev du om {}?".format(" eller ".join(response))
			else:
				no_varification_intents = ["hilsen","farvel","rephrase question"]
				if response not in no_varification_intents:
					present_choices([response])
					return "Bot: Vil du vide mere om {}? Så klik på knappen nedenfor!".format(response)
				else:
					return parameter_collecter(response)

		elif ss.intent == None and latest_message_type == "Ver:":
			#print(response)
			ss.intent = latest_message
			#if this is true, we have verified the users intent, and can move on to parameters
			required_parameters = parameter_collecter(ss.intent)
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
				ss.potential_parameters = NER_conversation(conversation)

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
					return "Bot:"+give_action(par,"verify")+" "+ss.potential_parameters[par][0]
				else:
					return "Bot:"+give_action(par,"ask")

	intent_copy = str(ss.intent)
	ss.intent = None
	return give_answer(intent_copy, column="completed") #give the final answer to the intent!


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
		ss.scenario = return_random_prompt()
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
		time.sleep(1)
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

		time.sleep(1)

		prev_conversation = ss.conversation

		prev_conversation.append(bot_response(prev_conversation))

		ss.conversation = prev_conversation

		print_messages([ss.conversation[-1]])
		#print_messages(ss.conversation)
		rerun()




if __name__ == '__main__':
	main()