#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from mysql.connector import MySQLConnection, Error
#from python_mysql_dbconfig import read_db_config
import mysql.connector
import datetime


# In[2]:

def connect_ITU_database(host,database,user,password):
	mydb = mysql.connector.connect(host=host,
	                               user=user,
	                               password=password,
	                               charset='utf8',
	                               database=database)
	mycursor = mydb.cursor()
	return mycursor,mydb


# In[3]:
#'id' and 'time' is added automatically: so values should start from conversation and end with comment
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