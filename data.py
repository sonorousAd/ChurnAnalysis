#I will be using Pandas library for data manipulation
# Sklearn for data analysis tools 
from multiprocessing.sharedctypes import Value
from pyexpat import model, native_encoding
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

# PART 1
#Firs we will load the data as dataframes 
#load clients table
clients_table = pd.read_csv("Includes/BASE DE DATOS/clients_table.txt")

# First fillter will find all contracts from 2015 Onwards and store it in clients_table_clean0
clients_table_clean0 = clients_table[clients_table["application_date"]>="2015-01-01"]
print(str(clients_table_clean0.shape[0])+" Number of columns after first filter ")

# Second filter will remove all rows where geography is italy AND application date is 2019 since all
# operations in italy were closed on 2019
clients_table_clean1 = clients_table_clean0.drop(clients_table_clean0.loc[(clients_table_clean0['Geography']=="Italy") & (clients_table_clean0["application_date"].str.startswith("2019"))].index)
print(str(clients_table_clean1.shape[0])+ " Number of columns after second filter ")

#Third Filter:
#clients table has 10 columns,
#so for us to take out clients with more than 75% of their info missing we have to decide between
# taking out clients with more than 8 (80%) NaN values per row or 7(70%) NaN values per row
# I choose to take out more than 7(70%) since this will allow us to have more data 
trs = clients_table.shape[1]-7
clients_table_clean2 = clients_table_clean1.dropna(axis=0, thresh=trs)
print(str(clients_table_clean2.shape[0]) + " Number of columns after third filter")

#Fourth filter will ensure each client only has one contract on the database
#This means removing duplicate CustomerId's
clients_table_clean3 = clients_table_clean2.drop(clients_table_clean2[clients_table_clean2["CustomerId"].duplicated()==True].index)
print(str(clients_table_clean3.shape[0]) + " Number of columns after fourth filter")

#Fifth filter will ensure each client has at least two years of information within the company
# First we change the type of both 'exit_date' and 'application_date' to datetime so we can mamipulate them
clients_table_clean3["exit_date"] = pd.to_datetime(clients_table_clean3["exit_date"])
clients_table_clean3["application_date"] = pd.to_datetime(clients_table_clean3["application_date"])

# Tofilter the information based on the premise that the contracts must be 
# two years or older, would mean to discard all the clients who left the product within the 
# first two years, leaving out an important section of the data. hence this filter will be applied later
# when we divide the data between clients who stay and clients who left

# clients_table_clean4 = clients_table_clean3[(clients_table_clean3["exit_date"].isnull()) | (clients_table_clean3["exit_date"] >= clients_table_clean3["application_date"]+pd.DateOffset(years=2)) ]
# print(str(clients_table_clean4.shape[0])+" Number of columns after fifth filter ")

# PART 2
# load products table
products_table = pd.read_csv("Includes/BASE DE DATOS/products_table.txt")
# Here we join our clean client data with their respective products and their info
# I use a inner join on CustomerId
clients_and_prods = pd.merge(products_table,clients_table_clean3, on="CustomerId", how="inner")
# Here we find the Number of products per client at the moment of application
number_prod_per_client = clients_and_prods.groupby(["CustomerId"])['Products'].count().reset_index(name='number_products')

#load transactions table
transactions_table = pd.read_csv("Includes/BASE DE DATOS/transactions_table.txt")
# Here we join our clean client data with their respective transactions data from transactions_table
# I use a inner join on CustomerId
clients_and_transac = pd.merge(transactions_table,clients_table_clean3,on="CustomerId",how="inner")
balance = clients_and_transac.groupby(["CustomerId"])['Value'].sum().reset_index(name='bank_balance_at_transac')

#load credit score table
credit_score_table = pd.read_csv("Includes/BASE DE DATOS/credit_score_table.txt")
# Here we join our clean client data with their respective credits from credit score table
# I use a inner join on CustomerId
clients_and_credit = pd.merge(credit_score_table,clients_table_clean3,on="CustomerId",how="inner")
credit_score = clients_and_credit.groupby(["CustomerId"])["Score"].mean().reset_index(name="credit_score")

# Now we obtain the age of the client at the moment of application
# Convert birthdate into datetime type 
clients_table_clean3["birth_date"] = pd.to_datetime(clients_table_clean3["birth_date"])
# append this age to the clean clients dataset
clients_table_clean3["application_age"] = (clients_table_clean3["application_date"]).dt.year-(clients_table_clean3["birth_date"]).dt.year

# Now we append all the created variables to the clean clients dataset
# NOTE: age at application has already been appended
# Append the number of products of each client
clients_table_final = pd.merge(number_prod_per_client,clients_table_clean3,on="CustomerId" ,how="inner")
# Append the clients' bank balance at time of transactions 
clients_table_final = pd.merge(balance,clients_table_final,on="CustomerId" ,how="inner")
# Append the clients credit bureau score 
clients_table_final = pd.merge(credit_score,clients_table_final,on="CustomerId" ,how="inner")

# Here we create a new Column that will contain 1 if the client stayed less than two years with the product and 0 otherwise
clients_table_final["left_before_2_years"] = np.where(clients_table_final["exit_date"] < clients_table_final["application_date"]+pd.DateOffset(years=2), 1, 0) 

# Here we divide the clients who left before the two years and the ones who stayed
#clients_left will contain all clients who left before completing the two years with the product
clients_left = clients_table_final[clients_table_final["exit_date"] < clients_table_final["application_date"]+pd.DateOffset(years=2)]
# clients_stay contains all the clients who have not left the product within the first two years
clients_stay = clients_table_final[(clients_table_final["exit_date"].isnull()) | (clients_table_final["exit_date"] >= clients_table_final["application_date"]+pd.DateOffset(years=2)) ]

# Descriptive Statistics of new variables 
print("-------Descriptive Variables-------")
# Mean
print("******Mean")
print(str(clients_table_final["credit_score"].mean()) + " is the mean Credit score")
print(str(clients_table_final["bank_balance_at_transac"].mean())+ " is the mean Bank balance at the moment of application")
print(str(clients_table_final["number_products"].mean()) + " is the mean number of products each client has")
print(str(clients_table_final["application_age"].mean())+ " is the mean client age at time of application")

print("******Stndard deviation")
# Standard deviation
print(str(clients_table_final["credit_score"].std()) + " is the standard deviation of the Credit score")
print(str(clients_table_final["bank_balance_at_transac"].std())+ " is the standard deviation of the Bank balance at the moment of application")
print(str(clients_table_final["number_products"].std()) + " is the standard deviation of the number of products each client has")
print(str(clients_table_final["application_age"].std())+ " is the standard deviation of the clients age at time of application")

print("******Maximum")
# Standard deviation
print(str(clients_table_final["credit_score"].max()) + " is the maximum Credit score")
print(str(clients_table_final["bank_balance_at_transac"].max())+ " is the maximum Bank balance at the moment of application")
print(str(clients_table_final["number_products"].max()) + " is the maximum number of products a client has")
print(str(clients_table_final["application_age"].max())+ " is the maximum age of a client at time of application")

print("******Minimum")
# Standard deviation
print(str(clients_table_final["credit_score"].min()) + " is the minumum Credit score")
print(str(clients_table_final["bank_balance_at_transac"].min())+ " is the minimum Bank balance at the moment of application")
print(str(clients_table_final["number_products"].min()) + " is the minimum number of products a client has")
print(str(clients_table_final["application_age"].min())+ " is the minimum age of a client at time of application")

#PART 3

# feature veriables 

# feature_cols = ["Geography","Gender","HasCrCard","IsActiveMember","EstimatedSalary","number_products","bank_balance_at_transac","credit_score","application_age"]
# Make Gender Variable 1 if female and 0 if male so we can use it in model
clients_table_final["Gender"] = np.where(clients_table_final["Gender"] == "Female", 1, 0) 
feature_cols = ["Gender","HasCrCard","IsActiveMember","EstimatedSalary","number_products","bank_balance_at_transac","credit_score","application_age","left_before_2_years"]
data_for_model = clients_table_final[feature_cols].dropna()
x_cols=["Gender","HasCrCard","IsActiveMember","EstimatedSalary","number_products","bank_balance_at_transac","credit_score","application_age"]

# Train model
X_train,X_test,y_train,y_test=train_test_split(data_for_model[x_cols],data_for_model["left_before_2_years"],test_size=0.3,random_state=0)

#since Logistic Regression doesnt accept missing values we have some options to deal with these ocurrances:
# 1.Replace the missing values with column averages
# 2.remove records that have missing values 
# I chooose to remove missing values 

model =  LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, model.predict(X_test)))
# print(model.coef_)
# print(model.intercept_)



