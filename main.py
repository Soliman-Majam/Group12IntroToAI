import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# =============================================================================
# # reading gym members dataset
# df = pd.read_csv('gym_members_exercise_tracking.csv', na_values=['NA', '?'])
# 
# # print first 5 rows to check it works
# print(df[:5])
# print(" ")
# 
# # checking to see if there is any missing data
# print(df.isnull().any())
# print(" ")
# =============================================================================

#------------------------------------------------------------------------------
#Treisis Dataset:
#df = pd.read_csv('housing_price_dataset.csv', na_values=['NA', '?'])

#print 5 rows

#print(df[:5])
#print(" ")

#checking for missing data

#print(df.isnull().any())
#------------------------------------------------------------------------------
#Fatima Dataset:
import pandas as pd

# Load the dataset and handle NA values
#df = pd.read_csv('heart_attack_analysis.csv', na_values=['NA', '?'])

# Print the entire DataFrame
#pd.set_option('display.max_rows', None)  # all rows
#print(df)
#print(" ")

# Checking for missing data in all columns
#print("Missing data in each column:")
#print(df.isnull().any())



# Load the dataset and handle NA values
df = pd.read_csv('mushroom_dataset.csv', na_values=['NA', '?'])

# Print the entire DataFrame
pd.set_option('display.max_rows', None)  # all rows
print(df)
print(" ")

# Checking for missing data in all columns
print("Missing data in each column:")
print(df.isnull().any())