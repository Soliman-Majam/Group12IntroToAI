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
