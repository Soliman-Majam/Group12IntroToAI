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
from sklearn.svm import SVC
import seaborn as sns
from sklearn.utils import shuffle
from scipy.stats import skew
from imblearn.over_sampling import SMOTE

# Load the dataset and handle NA values
df = pd.read_csv('water_potability.csv', na_values=['NA', '?'])

# Print the entire DataFrame
pd.set_option('display.max_rows', None)  # all rows
print(df.head())
print(" ")
# 3276 rows overall

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

# Plot histogram for the 'ph' column
df['ph'].hist(bins=30, figsize=(8, 6))
plt.title('Histogram of pH')
plt.xlabel('pH')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for the 'Sulfate' column
df['Sulfate'].hist(bins=30, figsize=(8, 6))
plt.title('Histogram of Sulfate')
plt.xlabel('Sulfate')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for the 'Trihalomethanes' column
df['Trihalomethanes'].hist(bins=30, figsize=(8, 6))
plt.title('Histogram of Trihalomethanes')
plt.xlabel('Trihalomethanes')
plt.ylabel('Frequency')
plt.show()

# Calculate skewness for the target variables
for column in ['ph', 'Sulfate', 'Trihalomethanes']:
    skewness = skew(df[column].dropna())  # Drop missing values
    print(f"Skewness for {column}: {skewness:.2f}")








# first replace null ph data with median value
phMed = df['ph'].median()
#phMed = df['ph'].mean()

df['ph'] = df['ph'].fillna(phMed)

 
# then Sulfate
sulfMed = df['Sulfate'].median()
#sulfMed = df['Sulfate'].mean()
df['Sulfate'] = df['Sulfate'].fillna(sulfMed)
 
# lastly Trihalomethanes
triMed = df['Trihalomethanes'].median()
#triMed = df['Trihalomethanes'].mean()
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(triMed)

# calculating with the average will give the accurac 50% whereas with median it gave 51%


# Now checking again for missing data in all columns
print("Missing data in each column:")
print(df.isnull().any())

# Plot histogram for the 'ph' column
df['ph'].hist(bins=30, figsize=(8, 6))
plt.title('Histogram of pH')
plt.xlabel('pH')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for the 'Sulfate' column
df['Sulfate'].hist(bins=30, figsize=(8, 6))
plt.title('Histogram of Sulfate')
plt.xlabel('Sulfate')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for the 'Trihalomethanes' column
df['Trihalomethanes'].hist(bins=30, figsize=(8, 6))
plt.title('Histogram of Trihalomethanes')
plt.xlabel('Trihalomethanes')
plt.ylabel('Frequency')
plt.show()

# Calculate skewness for the target variables
for column in ['ph', 'Sulfate', 'Trihalomethanes']:
    skewness = skew(df[column].dropna())  # Drop missing values
    print(f"Skewness for {column}: {skewness:.2f}")

# Creating pairplots comparing each of the features against each other,
# to check whether solution can be linear
sns.pairplot(df, hue="Potability", palette="Set2")
plt.title("Pair Plot of Features with Potability as Hue")
plt.show()

# Correlation heatmap to measure if any of the features correlate with each other at all
print("\nCorrelation Matrix:")
corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
print(corr_matrix)

print("\nHeatmap:")
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()
print("\nplot shown")

#randomizing the order of rows to avoid bias in the dataframe
print("\nRandomizing order of rows")
df = shuffle(df,random_state=42)
#gives 0.59 accuracy
#df = shuffle(df,random_state=15)
#0.58 accuracy
#df = shuffle(df,random_state=35)
#0.55 accuracy
#df = shuffle(df,random_state=7)
#0.51 accuracy
#df = shuffle(df,random_state=62)
#0.53 accuracy

print("\nRandomized DataFrame:")
print(df.head())

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
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
# accuracy score is 0.59
# 0 - precision=0.62 recall=0.87 f1score=0.72 support=405
# 1 - precision 0.37 recall=0.12 f1score=0.19 support=251
 
#test with a higher percentage 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
# accuracy is 0.6
# 0 - precision=0.65 recall=0.78 f1score=0.71 support=511
# 1 - precision 0.46 recall=0.32 f1score=0.38 support=308
 
#test with a higher percentage 0.3
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
# accuracy is 0.52
# 0 - precision=0.63 recall=0.57 f1score=0.60 support=616
# 1 - precision 0.37 recall=0.43 f1score=0.40 support=367
 
#test with a higher percentage 0.35
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=7)
# accuracy is 0.50
# 0 - precision=0.63 recall=0.47 f1score=0.54 support=710
# 1 - precision 0.39 recall=0.56 f1score=0.46 support=437
 
 
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
print("Confusion Matrix (Perceptron):")
print(cm)
 
# Display confusion matrix as a plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=per.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Perceptron)")
plt.show()

report = classification_report(y_test, pred) 

print("\nClassification Report (Perceptron):")
print(report) 

# =============================================================================

# Now working with support vector machine
svm_model = SVC(kernel='rbf', C=10, class_weight='balanced', random_state=42)
svm_model.fit(X_train_std, y_train)

# prediction on the test data
y_pred = svm_model.predict(X_test_std)

# print y_test and predicted data for comparison
print("True Labels (y_test):")
print(y_test)
print("\nPredicted Labels (y_pred):")
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (SVM):")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=per.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (SVM)")
plt.show()

# print classification report
print("\nClassification Report (SVM):")
print(classification_report(y_test, y_pred))


# Before applying SMOTE
print("Original class distribution:")
print(pd.Series(y_train).value_counts())

# Apply SMOTE to the training data (fix imbalance)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_std, y_train)

#after SMOTE
print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Split the SMOTE-resampled data into new training and testing sets
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(
    X_train_resampled, y_train_resampled, test_size=0.2, random_state=42
)

# Train SVM model after SMOTE
svm_smote = SVC(kernel='rbf', C=10, class_weight='balanced', random_state=42)
svm_smote.fit(X_train_smote, y_train_smote)

# Prediction on test data from the SMOTE split
y_pred_smote = svm_smote.predict(X_test_smote)

cm = confusion_matrix(y_test_smote, y_pred_smote)
print("Confusion Matrix (SVM with SMOTE):")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=per.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (SVM with SMOTE)")
plt.show()

# Classification report for SMOTE data
print("\nClassification Report (SVM After SMOTE):")
print(classification_report(y_test_smote, y_pred_smote))



