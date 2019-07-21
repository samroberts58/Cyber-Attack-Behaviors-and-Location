# -*- coding: utf-8 -*-
"""
Title: Cyber Attack Behaviors and a Malicious User’s Country of Origin
Author: Samantha Roberts
"""

#######################################
#Data Pre-processing
#######################################

# Import required packages.
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Read the data into a data frame and drop unnecessary columns.
honeypot = pd.read_csv('C:/Users/samro/Desktop/GW/Mach Learn/Final Project/Data/Amazon Honeypot/AWS_Honeypot_marx-geo.csv')
honeypot.drop(['type','spt','src','srcstr','cc', 'latitude', 'longitude','localeabbr','postalcode','Unnamed: 15'], axis=1, inplace=True)


# Find how many rows have NA's
print(honeypot.isna().sum())

# Remove all NA's.  Total samples goes from 451,581 to 312,761.
honeypot.dropna(how='any',inplace=True)    

# Remove group of countries with < 10 occurences because StratifiedKFold will require 10 in a class.
# DF shape goes down to 312,591.
honeypot_group = honeypot.groupby(['country'], as_index=True).count()
honeypot_group_list = list(honeypot_group[honeypot_group['host'] >= 10].reset_index().iloc[:,0])
honeypot = honeypot[honeypot['country'].isin(honeypot_group_list)]


# Verify no strange values.
print(honeypot['host'].unique())
print(honeypot['proto'].unique())
print(honeypot['dpt'].unique())
print(honeypot['country'].unique())
print(honeypot['locale'].unique())


# Check datatypes
for i in honeypot:
    print(i, honeypot[i].dtype)
    
# Update dpt (destination port) to object so it can be used as a categorical class
honeypot['dpt']=honeypot['dpt'].astype(int).astype(str)

# Break out datetime column for easier visualization
ymsplit = honeypot['datetime'].str.split("/", n=3, expand=True)
honeypot['Month'] = ymsplit[0]


# Group by and count for countries
country_count = honeypot.groupby(['country'], as_index=True).count()
country_count = country_count.reset_index()
# print(country_count)
# print(honeypot.shape)

# US and China have vast majority of entries.  Drop them so all the other countries can be used.
country_list = list(country_count[country_count['datetime'] > 20000].reset_index().iloc[:,1])

for i in country_list:
    honeypot = honeypot[~honeypot['country'].isin([i])]

# Create dict to update all honeypot rows with count data
count_dict = pd.Series(country_count['host'].values, index=country_count['country']).to_dict()

# Use the dictionary to update all entries of count.
for i in honeypot['country']:
    if i in count_dict:
        honeypot['count'] = count_dict[i]

#######################################
# Exploratory Data Analysis
#######################################

# Import all required packages
import matplotlib.pyplot as plt
import seaborn as sns

# Basic Descriptives

# Number of distinct years (all 2013), months (7 month range: March - September)
print(honeypot['Month'].unique().shape)

# Number of host options - 9
print(honeypot['host'].unique().shape)

# Number of prototypes involved - 2
print(honeypot['proto'].unique().shape)

# Number of countires involved - 95
print(honeypot['country'].unique().shape)

# Number of destination ports involved - 466
print(honeypot['dpt'].unique().shape)

# Number of locale involved - 1028
print(honeypot['locale'].unique().shape)


# Count of entries per Feature
for i in honeypot:
    sns.set(style="darkgrid")
    ax = sns.countplot(x=honeypot[i], data=honeypot)
    plt.xticks(rotation=45)
    plt.savefig(i+' count.png')
    plt.tight_layout()
    plt.show()
   
# Swarmplot of countries by the other columns
country_count_all = honeypot.groupby('country').nunique()
country_count_all.drop(['datetime','count'], axis=1, inplace=True)
# print(country_count_all['dpt'].loc[country_count_all['dpt'] > 70])
# print(country_count_all['locale'].loc[country_count_all['locale'] > 40])

for i in country_count_all:
    ax = sns.catplot(x='country',y=i, kind='swarm',hue='Month', data=country_count_all)
    plt.xticks(rotation=45)
    plt.savefig(i+' swarmplot.png')
    plt.tight_layout()
    plt.show()


#######################################
# Hyperparameter Tuning and Model Selection - Code from Exercise 12 used!    
#######################################

# Build a pipeline to implement the following models: Logistic Regression, RandomForestClassifier, SVC, and PCA

# Import all required packages
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import time

# Split a small sample for parameter selection 
honeypot_sample=honeypot.sample(n=100,random_state=0)

# Get the target vector
y = honeypot_sample['country'].values

# Get the feature vector
X = honeypot_sample[(list(honeypot_sample.drop(['country', 'locale', 'datetime', 'count'], axis=1).columns))]

# Encode the features using one-hot-encoding
X = pd.get_dummies(X)

# Declare the LabelEncoder
le = LabelEncoder()

# Enclode the target
y = le.fit_transform(y)


# Create a dictionary of the chosen models   
clfs = {'lr': LogisticRegression(random_state=0),
        'rf': RandomForestClassifier(random_state=0),
        'svc': SVC(random_state=0)}


n_components = [X.shape[1] // 4, X.shape[1] // 2, X.shape[1]]

# Create a dictionary for whether PCA is used or not.
pipe_clfs = {}

for name, clf in clfs.items():
    pipe_clfs[name] = {}
    for n_component in n_components:
        if n_component < X.shape[1]:
            pipe_clfs[name][n_component] = Pipeline([('StandardScaler', StandardScaler()), 
                                                     ('PCA', PCA(n_components=n_component, random_state=0)), 
                                                     ('clf', clf)])
        else:
            pipe_clfs[name][n_component] = Pipeline([('StandardScaler', StandardScaler()), 
                                                     ('clf', clf)])

# Create a dictionary for parameter specifications based on each individual model.
param_grids = {}


# Logistic Regression parameters
C_range = [10 ** i for i in range(-4, 5)]

param_grid = [{'clf__multi_class': ['ovr'], 
               'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
               'clf__C': C_range},
              {'clf__multi_class': ['multinomial'],
               'clf__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
               'clf__C': C_range}]

param_grids['lr'] = param_grid


# Random Forest Classifier parameters
param_grid = [{'clf__n_estimators': [2, 10, 30],
               'clf__min_samples_split': [2, 10, 30],
               'clf__min_samples_leaf': [1, 10, 30]}]

param_grids['rf'] = param_grid


# SVM parameters
param_grid = [{'clf__C': [0.01, 0.1, 1, 10, 100],
               'clf__gamma': [0.01, 0.1, 1, 10, 100],
               'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]

param_grids['svc'] = param_grid

# The list of [best_score_, best_params_, best_estimator_]
best_score_param_estimators = []

# Set the start time.
start = time.time()

# Run for each classifier with specified variety of parameters to determine the best combination.
for name in pipe_clfs.keys():
    for n_component in n_components:   
        gs = GridSearchCV(estimator=pipe_clfs[name][n_component],
                          param_grid=param_grids[name],
                          scoring='accuracy',
                          n_jobs=-1,
                          cv=StratifiedKFold(n_splits=10,
                                             shuffle=True,
                                             random_state=0))
        # Fit the pipeline
        gs = gs.fit(X, y)

        # Update best_score_param_estimators
        best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

# Sort best_score_param_estimators in descending order of the best_score_
best_score_param_estimators = sorted(best_score_param_estimators, key=lambda x : x[0], reverse=True)

# Capture the end time. 
end = time.time()

# Calculate how long it took to run the segment.
print(end - start) #122 seconds

# Print out best_estimator
print(best_score_param_estimators[0][2])
print("Accuracy: ", best_score_param_estimators[0]) 



#######################################
# Full Model Run:
#######################################

# Import required packages.
from sklearn.model_selection import train_test_split


# 1. Predict Country!


# Ensure each country has at least 10 samples included again.
honeypot_fullsample = honeypot
honeypot_fullsample_count = honeypot.groupby(['country'], as_index=True).count()
honeypot_fs_list = list(honeypot_fullsample_count.loc[honeypot_fullsample_count['host'] > 10].reset_index().iloc[:,0])
honeypot_fullsample = honeypot_fullsample.loc[honeypot_fullsample['country'].isin(honeypot_fs_list)]
# print(honeypot_fullsample.columns)

#Use the larger sample dataset to run the model.
# Get the target vector
y = honeypot_fullsample['country'].values

# Get the feature vector
X = honeypot_fullsample[(list(honeypot_fullsample.drop(['datetime','country', 'locale', 'count'], axis=1).columns))]
# print(X.columns)
# Encode the features using one-hot-encoding
X = pd.get_dummies(X)

# Declare the LabelEncoder
le = LabelEncoder()

# Enclode the target
y = le.fit_transform(y)

# Randomly choose 30% of the data for testing (set randome_state as 0 and stratify as y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


# Declare the classifier dynamically with hyperparameter tuning results
pipe_clf = best_score_param_estimators[0][2]

#Set the start time
start = time.time()


# Fit the pipeline
pipe_clf.fit(X_train, y_train)

#Track the end time and compare.
end = time.time()
print(end - start)

#14 seconds

# Get the score (rounding to two decimal places)
score = round(pipe_clf.score(X_test, y_test), 2)
print(score)
#0.4



# 2. Predict Locale (without knowing the country)


# Ensure each country has at least 10 samples included.
honeypot_fullsample_count = honeypot.groupby(['locale'], as_index=True).count()
honeypot_fs_list = list(honeypot_fullsample_count.loc[honeypot_fullsample_count['host'] > 10].reset_index().iloc[:,0])
honeypot_fullsample = honeypot_fullsample.loc[honeypot_fullsample['locale'].isin(honeypot_fs_list)]
# print(honeypot_fullsample.columns)

# Use the larger sample dataset to run the model.
# Get the target vector
y = honeypot_fullsample['locale'].values

# Get the feature vector
X = honeypot_fullsample[(list(honeypot_fullsample.drop(['datetime','country', 'locale', 'count'], axis=1).columns))]
# print(X.columns)
# Encode the features using one-hot-encoding
X = pd.get_dummies(X)

# Declare the LabelEncoder
le = LabelEncoder()

# Enclode the target
y = le.fit_transform(y)

# Randomly choose 30% of the data for testing (set randome_state as 0 and stratify as y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

start = time.time()
# Fit the pipeline

pipe_clf.fit(X_train, y_train)

end = time.time()
print(end - start)

#64 seconds

# Get the score (rounding to two decimal places)
score = round(pipe_clf.score(X_test, y_test), 2)
print(score)
#0.42


# 3. Predict Locale (with Country known)

# Ensure each country has at least 10 samples included.
honeypot_fullsample_count = honeypot.groupby(['locale'], as_index=True).count()
honeypot_fs_list = list(honeypot_fullsample_count.loc[honeypot_fullsample_count['host'] > 10].reset_index().iloc[:,0])
honeypot_fullsample = honeypot_fullsample.loc[honeypot_fullsample['locale'].isin(honeypot_fs_list)]
# print(honeypot_fullsample.columns)

# Use the larger sample dataset to run the model.
# Get the target vector
y = honeypot_fullsample['locale'].values

# Get the feature vector
X = honeypot_fullsample[(list(honeypot_fullsample.drop(['datetime', 'locale', 'count'], axis=1).columns))]
# print(X.columns)
# Encode the features using one-hot-encoding
X = pd.get_dummies(X)

# Declare the LabelEncoder
le = LabelEncoder()

# Enclode the target
y = le.fit_transform(y)

# Randomly choose 30% of the data for testing (set randome_state as 0 and stratify as y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

start = time.time()
# Fit the pipeline

pipe_clf.fit(X_train, y_train)

end = time.time()
print(end - start)

#74 seconds

# Get the score (rounding to two decimal places)
score = round(pipe_clf.score(X_test, y_test), 2)
print(score)
#0.55
