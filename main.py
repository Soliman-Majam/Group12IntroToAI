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

# Load the dataset and handle NA values
df = pd.read_csv('water_potability.csv', na_values=['NA', '?'])

# Print the entire DataFrame
pd.set_option('display.max_rows', None)  # all rows
print(df)
print(" ")
# 3276 rows

# Checking for missing data in all columns
print("Missing data in each column:")
print(df.isnull().any())
print(" ")

# ph, Sulfate, and Trihalomethanes have missing data

# =============================================================================
# dfcut = df.dropna()
# print(dfcut)
# print(" ")
# # if we drop all rows with null values, only 2011 rows remain
# # 1,265 rows would be dropped
# =============================================================================


# first replace null ph data with median value
phMed = df['ph'].median()
df['ph'] = df['ph'].fillna(phMed)
 
# then Sulfate
sulfMed = df['Sulfate'].median()
df['Sulfate'] = df['Sulfate'].fillna(sulfMed)
 
# lastly Trihalomethanes
triMed = df['Trihalomethanes'].median()
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(triMed)

# Now checking again for missing data in all columns
print("Missing data in each column:")
print(df.isnull().any())

# collecting  the collumn names for non-target features
result = []
for x in df.columns:
    if x !='Potability':
        result.append(x)
 
print("\nFeature names: ")        
print(result)   
    
# defining feature and target data
X = df[result].values
y= df['Potability'].values

#Print the shape of the original data
print("\nOriginal dataset shape:")
print("Features (X):", X.shape)
print("Target (y):", y.shape)
 
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
# accuracy score is 0.51
# 0 - precision=0.58 recall=0.60 f1score=0.59 support=388
# 1 - precision 0.39 recall=0.37 f1score=0.38 support=268
 
#test with a higher percentage 0.25
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
# accuracy is 0.5
# 0 - precision=0.59 recall=0.54 f1score=0.56 support=484
# 1 - precision 0.40 recall=0.45 f1score=0.43 support=335
 
#test with a higher percentage 0.3
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
# accuracy is 0.56
# 0 - precision=0.60 recall=0.79 f1score=0.68 support=582
# 1 - precision 0.42 recall=0.22 f1score=0.29 support=401
 
#test with a higher percentage 0.35
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=7)
# accuracy is 0.56
# 0 - precision=0.59 recall=0.80 f1score=0.68 support=680
# 1 - precision 0.42 recall=0.21 f1score=0.28 support=467
 
 
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
 
# not using scaler yield no longer yields higher accuracy
# it drops from 0.51 to 0.41
# so we should keep standard scaler applied
# =============================================================================
# per2 = Perceptron(max_iter=35,tol=0.001,eta0=1)
# per2.fit(X_train, y_train)
# pred2 = per2.predict(X_test)
# print('Accuracy: %.2f' % accuracy_score(y_test, pred2))
# print(" ")
# =============================================================================
 
# creating perceptron model
per = Perceptron(max_iter=35,tol=0.001,eta0=1)
 
# training the model with training data (scaled by standard scaler) and training target
per.fit(X_train_std,y_train)
 
# making prediction for the test data
pred = per.predict(X_test_std)
 
# check accuracy with accuracy score
print('Accuracy: %.2f' % accuracy_score(y_test, pred))
print(" ")
   
# Calculate confusion matrix
cm = confusion_matrix(y_test, pred)
print("Confusion Matrix:")
print(cm)
 
# Display confusion matrix as a plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=per.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

report = classification_report(y_test, pred) 

print(report) 





