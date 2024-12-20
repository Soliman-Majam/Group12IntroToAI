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
from sklearn.model_selection import ParameterGrid
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc


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

# First replace null ph data with median value
phMed = df['ph'].median()
#phMed = df['ph'].mean()
df['ph'] = df['ph'].fillna(phMed)

# Then Sulfate
sulfMed = df['Sulfate'].median()
#sulfMed = df['Sulfate'].mean()
df['Sulfate'] = df['Sulfate'].fillna(sulfMed)
 
# Lastly Trihalomethanes
triMed = df['Trihalomethanes'].median()
#triMed = df['Trihalomethanes'].mean()
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(triMed)

# Calculating with the average will give the accuracy 50% whereas with median it gave 51%

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

# Create pairplots comparing each of the features against each other,
# to check whether solution can be linear
sns.pairplot(df, hue="Potability", palette="Set2")
plt.title("Pair Plots of Features coloured by Potability")
plt.show()
# It is evident from output that solution is not linearly seperable

# Correlation heatmap to measure if any of the features correlate with each other at all
# to check whether it would be worth it to use PCA for dimensionality reduction
print("\nCorrelation Matrix:")
corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
print(corr_matrix)

print("\nHeatmap:")
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of the Features")
plt.show()
print("\nplot shown")
# From the output it is evident that no two features have high correlation with each other
# Therefore using PCA would be unwise

# Randomize the order of rows to avoid bias in the dataframe
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

# Collect the column names for non-target features
result = []
for x in df.columns:
    if x !='Potability':
        result.append(x)
 
print("\nFeature names: ")        
print(result)   
    
# Define feature and target data
X = df[result].values
y= df['Potability'].values

# Print the shape of the original data for reference
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
 
# Prepare the standard scaler
sc = StandardScaler()
 
# Fit the scaler to train and test data
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
 
print("printing training set")
print(X_train[0])
print(X_train_std[0])
print(" ")
 
# Not using scaler yield no longer yields higher accuracy
# it drops from 0.51 to 0.41
# so we should keep standard scaler applied
# =============================================================================
# per2 = Perceptron(max_iter=35,tol=0.001,eta0=1)
# per2.fit(X_train, y_train)
# pred2 = per2.predict(X_test)
# print('Accuracy: %.2f' % accuracy_score(y_test, pred2))
# print(" ")
# =============================================================================
 
# Preparation for applying SMOTE to data to fix the class imbalance (less 1 values than 0 values)

# Class distribution before applying SMOTE
print("Original class distribution:")
print(pd.Series(y_train).value_counts())

# Apply SMOTE to the training data (fix imbalance)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_std, y_train)

# Class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())
# Now number of rows per class is the same for both 0 and 1

# Split the SMOTE-resampled data into new training and testing sets
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(
    X_train_resampled, y_train_resampled, test_size=0.25, random_state=42
)

# =============================================================================

# Create the Perceptron model
per = Perceptron(max_iter=100,shuffle=True,eta0=0.1) 
# Less parameters gives better result for us but might be a worse model overall?

#per = Perceptron(max_iter=100,shuffle=True,tol=1e-05,eta0=0.001, early_stopping=True)
# More parameter make the results more reliable but those results are worse overall

# Train the model with training data (scaled by standard scaler) and training target
per.fit(X_train_std,y_train)
 
# Make prediction for the test data
pred = per.predict(X_test_std)
 
# Check accuracy with accuracy score
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

# Classification report to show important values (precision, recall, f1-score, accuracy etc.)
per_report = classification_report(y_test, pred) 

print("\nClassification Report (Perceptron):")
print(per_report) 

# Highest we can get smote dataset perceptron is 0.49 accuracy

per_smote = Perceptron(max_iter=100,shuffle=True,tol=0.01,eta0=0.1, early_stopping=True)
per_smote.fit(X_train_smote, y_train_smote)
smote_pred = per_smote.predict(X_test_smote)

# Check accuracy with accuracy score
print('Accuracy: %.2f' % accuracy_score(y_test_smote, smote_pred))
print(" ")
   
# Calculate confusion matrix
cm = confusion_matrix(y_test_smote, smote_pred)
print("Confusion Matrix (Perceptron with SMOTE):")
print(cm)
 
# Display confusion matrix as a plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=per.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Perceptron with SMOTE)")
plt.show()

# Classification report to show important values (precision, recall, f1-score, accuracy etc.)
per_smote_report = classification_report(y_test_smote, smote_pred) 

print("\nClassification Report (Perceptron with SMOTE):")
print(per_smote_report) 

# =============================================================================

# Now working with support vector machine
svm_model = SVC(kernel='rbf', C=30, gamma='scale', shrinking=True, class_weight='balanced', random_state=42)
svm_model.fit(X_train_std, y_train)

# Prediction on the test data
y_pred = svm_model.predict(X_test_std)

# Print y_test and predicted data for comparison
print("True Labels (y_test):")
print(y_test)
print("\nPredicted Labels (y_pred):")
print(y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (SVM):")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=per.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (SVM)")
plt.show()

# Classification report
print("\nClassification Report (SVM):")
print(classification_report(y_test, y_pred))

# Train SVM model on SMOTE data (outdated model)
#svm_smote = SVC(kernel='rbf', C=30, gamma='scale', shrinking=True, random_state=42)
#svm_smote.fit(X_train_smote, y_train_smote)

#c50 = 0.71
#c40=0.71
#c30=0.72
#c20=0.71
#c10=0.69
#c1=0.65
#c100=0.69
#c90=0.70
#c80=0.70
#c70=0.70
#c60=0.70
#Prediction on test data from the SMOTE split
#y_pred_smote = svm_smote.predict(X_test_smote)

#cm = confusion_matrix(y_test_smote, y_pred_smote)
#print("Confusion Matrix (SVM with SMOTE):")
#print(cm)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=per.classes_)
#disp.plot(cmap=plt.cm.Blues)
#plt.title("Confusion Matrix (SVM with SMOTE)")
#plt.show()

# Classification report for SMOTE data
#print("\nClassification Report (SVM with SMOTE):")
#print(classification_report(y_test_smote, y_pred_smote))

# Train SVM model on SMOTE data with more tuning (outdated)
#svm_smote = SVC(kernel='rbf', C=1, gamma=1, shrinking=True, random_state=42)
#svm_smote.fit(X_train_smote, y_train_smote)

#y_pred_smote = svm_smote.predict(X_test_smote)

#cm = confusion_matrix(y_test_smote, y_pred_smote)
#print("Confusion Matrix (SVM with SMOTE tuned):")
#print(cm)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=per.classes_)
#disp.plot(cmap=plt.cm.Blues)
#plt.title("Confusion Matrix (SVM with SMOTE tuned)")
#plt.show()
# Classification report for SMOTE data
#print("\nClassification Report (SVM with SMOTE tuned):")
#print(classification_report(y_test_smote, y_pred_smote))


# Train SVM model after SMOTE (up to date version)
svm_smote = SVC(kernel='rbf', C=3.062735057040824, gamma=1.662669390630025, shrinking=True, random_state=42, max_iter=1000, class_weight='balanced')
svm_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = svm_smote.predict(X_test_smote)

# Confusion matrix
cm = confusion_matrix(y_test_smote, y_pred_smote)
print("Confusion Matrix (SVM with SMOTE more tuned):")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=per.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (SVM with SMOTE more tuned)")
plt.show()

# Classification report for SMOTE data
print("\nClassification Report (SVM with SMOTE  more tuned):")
print(classification_report(y_test_smote, y_pred_smote))

# Decision function scores for the test set
y_scores = svm_smote.decision_function(X_test_smote)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test_smote, y_scores)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (SVM)")
plt.legend()
plt.grid()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test_smote, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid()
plt.show()

# =============================================================================

# Define the Random Forest model
rf_model = RandomForestClassifier(n_estimators=700, max_features='sqrt', max_samples=0.8, min_samples_split=6, min_samples_leaf=2, max_depth=25, random_state=42, class_weight='balanced')

# Train the model on the original (non-SMOTE) training data
rf_model.fit(X_train_std, y_train)

# Predict on the test data
y_pred_rf = rf_model.predict(X_test_std)

# Evaluate the model
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix (Random Forest):")
print(cm_rf)

disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=np.unique(y_test_smote))
disp_rf.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Random Forest)")
plt.show()

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred_rf))

# Define the Random Forest model with smote data
rf_model_smote = RandomForestClassifier(n_estimators=700, max_features='sqrt', max_samples=0.8, min_samples_split=6, min_samples_leaf=2, max_depth=25, random_state=42, class_weight='balanced')

# Train the model on the SMOTE-resampled training data
rf_model_smote.fit(X_train_smote, y_train_smote)

# Predict on the SMOTE test data
y_pred_rf_smote = rf_model_smote.predict(X_test_smote)

# Evaluate the model
cm_rf_smote = confusion_matrix(y_test_smote, y_pred_rf_smote)
print("Confusion Matrix (Random Forest with SMOTE):")
print(cm_rf_smote)

disp_rf_smote = ConfusionMatrixDisplay(confusion_matrix=cm_rf_smote, display_labels=np.unique(y_test_smote))
disp_rf_smote.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Random Forest with SMOTE)")
plt.show()

print("\nClassification Report (Random Forest with SMOTE):")
print(classification_report(y_test_smote, y_pred_rf_smote))

# Accuracy
print("\nAccuracy:", accuracy_score(y_test_smote, y_pred_rf_smote))

features = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 
            'Organic_carbon', 'Trihalomethanes', 'Turbidity']

feature_importances = rf_model_smote.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance when it comes to random forest decision-making
plt.figure(figsize=(8, 4))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Random Forest with SMOTE')
plt.gca().invert_yaxis()
plt.show()

# =============================================================================
# Now working with Neural Network

# One-hot encode the SMOTE-resampled target variable for the neural network
ohe = OneHotEncoder(sparse_output=False)
y_train_resampled_encoded = ohe.fit_transform(y_train_resampled.reshape(-1, 1))
y_test_smote_encoded = ohe.transform(y_test_smote.reshape(-1, 1))

# Define the neural network model
nn_model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001), input_dim=X_train_resampled.shape[1]),
    Dropout(0.1),  # Reduced dropout
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

nn_model.compile(optimizer=Adam(learning_rate=0.005),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])


es = EarlyStopping(monitor='val_loss', patience=14, restore_best_weights=True)

# Train the neural network on the SMOTE-resampled data (increased epochs to 100, added early stopping)
history = nn_model.fit(X_train_resampled, y_train_resampled_encoded,
                       validation_split=0.2,
                       epochs=100,
                       batch_size=32,
                       verbose=1,
                       callbacks=[es])

# Evaluate the neural network on the SMOTE test data
nn_loss, nn_accuracy = nn_model.evaluate(X_test_smote, y_test_smote_encoded)
print(f"\nNeural Network Accuracy (SMOTE Data): {nn_accuracy:.2f}")

# Predictions with the neural network (SMOTE data)
nn_predictions = nn_model.predict(X_test_smote)
y_pred_nn = np.argmax(nn_predictions, axis=1)

# Classification report for the neural network (SMOTE data)
print("\nClassification Report (Neural Network with SMOTE Data):")
print(classification_report(y_test_smote, y_pred_nn))

# Confusion matrix for the neural network (SMOTE data)
cm_nn = confusion_matrix(y_test_smote, y_pred_nn)
print("\nConfusion Matrix (Neural Network with SMOTE Data):")
print(cm_nn)
disp_nn = ConfusionMatrixDisplay(confusion_matrix=cm_nn, display_labels=np.unique(y_test_smote))
disp_nn.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Neural Network with SMOTE Data)")
plt.show()

# Line chart showing the Loss rates (validation and training) over the course of the epochs it runs for
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss rates over epochs (Neural Network)")
plt.legend()
plt.show()


# Parameter grid used for tuning the SVM, we started off with small values, and tweaked them to be
# higher over time
# =============================================================================
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform
# 
# # Define the parameter grid
# param_dist = {
#     'C': uniform(0.1, 10),  # Continuous values between 0.1 and 10
#     'gamma': uniform(0.01, 10),  # Continuous values between 0.01 and 10
#     'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
#     'shrinking': [True, False],
#     'max_iter': [100, 500, 1000]
# }
# 
# # Randomized search
# random_search = RandomizedSearchCV(SVC(), param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', random_state=42, verbose=2)
# 
# # Perform random search
# random_search.fit(X_train_smote, y_train_smote)
# 
# # Best parameters and best score
# print("Best parameters found: ", random_search.best_params_)
# print("Best score: ", random_search.best_score_)
# 
# # Make predictions with the best model
# best_svm = random_search.best_estimator_
# y_pred_smote = best_svm.predict(X_test_smote)
# 
# # Confusion matrix
# cm = confusion_matrix(y_test_smote, y_pred_smote)
# print("Confusion Matrix (SVM with SMOTE, best tuned):")
# print(cm)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test_smote))
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix (Best tuned SVM with SMOTE)")
# plt.show()
# 
# # Classification Report
# print("\nClassification Report (Best tuned SVM with SMOTE):")
# print(classification_report(y_test_smote, y_pred_smote))
# =============================================================================



# Initial parameter grid for random forest, we did manual tuning afterwards
# =============================================================================
# param_grid = {
#     'n_estimators': [100, 200, 300, 400, 500],
#     'max_depth': [5, 10, 20, None],  # Try varying depth limits
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2', None],
#     'class_weight': [None, 'balanced'],
#     'criterion': ['gini', 'entropy']
# }
# 
# # Set up GridSearchCV
# grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, 
#                            scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
# 
# # Fit GridSearchCV on the training set
# grid_search.fit(X_train_smote, y_train_smote)
# 
# # Best parameters and best score
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)
# 
# # Best model from grid search
# best_rf_model = grid_search.best_estimator_
# 
# # Make predictions with the best model
# y_pred_best = best_rf_model.predict(X_test_smote)
# 
# # Confusion matrix and classification report
# cm_best = confusion_matrix(y_test_smote, y_pred_best)
# print("Confusion Matrix (Best Random Forest with SMOTE):")
# print(cm_best)
# 
# print("\nClassification Report (Best Random Forest with SMOTE):")
# print(classification_report(y_test_smote, y_pred_best))
# =============================================================================


# =============================================================================
# History of tuning for Neural Network:

#l2 relu 0.1, 0.1 dropout 0.90, 0.92, 0.86,0.89,0.94
#0.2, 0.1 dropout, 0.88,0.90,0.89,0.84,0.89
#0.2,0.2, dropout, 0.7,0.87,0.72,0.66,0.86
#0.2,0.15 dropout, 0.84,0.63,0.67,0.64,0.68
# all relu 0.87,0.86
# relu, tanh, leaky_relu , 0.82, 0.87, 0.82,0.84,0.68
#all tanh 0.78, 0.66,0.71,0.72,0.73
#relu,tanh,tanh 0.78,0.78,0.85,0.73,0.81
#relu,leaky_relu,leaky_relu 0.68,0.79,0.82,0.76

# learning rate testing: 0.01 = 0.72

#with l2 0.90,

# Compile the model (made the learning rate slightly higher)
#nn_model.compile(optimizer=Adam(learning_rate=0.01),
#                loss='categorical_crossentropy',
#               metrics=['accuracy'])

#0.005 =0.74, 0.71, 0.73,0.79,0.79
#nn_model.compile(optimizer=Adam(learning_rate=0.005), Adjusting learning rate here
#                 loss='categorical_crossentropy',
 #                metrics=['accuracy'])

#learning rate 0.0045 = 0.71,0.68,0.72,0.74,0.76
#nn_model.compile(optimizer=Adam(learning_rate=0.0045),
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])


#learning rate 0.0055 =0.68,0.77,0.80,0.68,0.75
#nn_model.compile(optimizer=Adam(learning_rate=0.0055),
#                loss='categorical_crossentropy',
#               metrics=['accuracy'])

#learning rate 0.006 = 0.75,0.75,0.76,0.74,0.73
# =============================================================================
