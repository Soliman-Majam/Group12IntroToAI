import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report 
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
df1 = pd.read_csv('waterpotability.csv', na_values=['NA', '?'])

# Print the entire DataFrame
pd.set_option('display.max_rows', None)  # all rows
print(df1)
print(" ")

# Checking for missing data in all columns
print("Missing data in each column:")
print(df1.isnull().any())



# Load the dataset and handle NA values
df = pd.read_csv('mushroom_cleaned.csv', na_values=['NA', '?'])

# Print the entire DataFrame
pd.set_option('display.max_rows', None)  # all rows
print(df)
print(" ")

# Checking for missing data in all columns
print("Missing data in each column:")
print(df.isnull().any())
print(" ")

# collecting  the collumn names for non-target features
result = []
for x in df.columns:
    if x !='class':
        result.append(x)

print("\nFeature names: ")        
print(result)       
# defining feature and target data
X = df[result].values
y= df['class'].values

# Print the shape of the original data
print("\nOriginal dataset shape:")
print("Features (X):", X.shape)
print("Target (y):", y.shape)

# Splitting data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
# accuracy score is 0.48

#test with a higher percentage 0.25
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
# accuracy is 0.58

#test with a higher percentage 0.3
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
# accuracy is 0.58

#test with a higher percentage 0.35
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=7)
# accuracy is 0.62


# Print the shape of the split data
print("\nAfter splitting:")
print("Training set - Features (X_train):", X_train.shape, "Target (y_train):", y_train.shape)
print("Testing set - Features (X_test):", X_test.shape, "Target (y_test):", y_test.shape)
print(" ")

# preparing the standard scaler
sc = StandardScaler()

sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print("printing training set")
print(X_train[0])
print(X_train_std[0])
print(" ")

# not using scaler yield higher accuracy score?
# =============================================================================
# per = Perceptron(max_iter=35,tol=0.001,eta0=1)
# per.fit(X_train, y_train)
# pred0 = per.predict(X_test)
# print('Accuracy: %.2f' % accuracy_score(y_test, pred0))
# print(" ")
# =============================================================================

# creating perceptron model
per1 = Perceptron(max_iter=35,tol=0.001,eta0=1)

# training the model with training data (scaled by standard scaler) and training target
per1.fit(X_train_std,y_train)

# making prediction for the test data
pred = per1.predict(X_test_std)

# check accuracy with accuracy score
print('Accuracy: %.2f' % accuracy_score(y_test, pred))
print(" ")



# Calculate confusion matrix
cm = confusion_matrix(y_test, pred)
print("Confusion Matrix:")
print(cm)

# Display confusion matrix as a plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=per1.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

report = classification_report(y_test, pred) 

print(report) 




