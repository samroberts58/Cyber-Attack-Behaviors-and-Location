# Cyber Attack Behaviors and a Malicious User's Country of Origin
Samantha Roberts, M.S. Data Science Student
George Washington University

## Introduction

### Brief Summary:
```
This project will explore the data patterns and variable relationships of a recorded dataset from an AWS Honeypot from March 3, 2013 
through September 8, 2013. The goal is to develop a model that will accurately classify which country an attacker is originating from 
based on multiple input features, including destination ports, protocol category, and host origin information. 
```

### Dataset Variables Used:
```
Host: Host region of attack's origin
Proto: Protocol; TCP or UDP
DPT: Destination Port
Country: Attack's country of origin
Month: The month the attack occurred
```

## Project Methodology

### How to Run: 
```
Intel Distribution for Python (IDP) environment was downloaded for Python 3.6.
Import the following packages:
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
```
 
### Files included:
```
README.rmd
RobertsSamantha_FinalProj_Code.py
Roberts, Samantha Final Report.doc
Amazon_Honeypot_marx-geo.csv
Roberts, Samantha Cyber Attack Behaviors.ppt
```
### Model Selection:
```
Hyperparameter Tuning was used to find the strongest classification model with parameters for the data selected.  
Initial considerations included Logistic Regression, Random Forest Classifier, SVC, and PCA.  Resulting model selected was PCA with Logistic Regression and parameters exemplified below.
```
```
Pipeline(memory=None,
     steps=[('StandardScaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('PCA', PCA(copy=True, iterated_power='auto', n_components=11, random_state=0,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr',
          n_jobs=None, penalty='l2', random_state=0, solver='newton-cg',
          tol=0.0001, verbose=0, warm_start=False))])
```

## Project Design and Issues
```
The script is organized into chunks named "Data Pre-Processing", "Exploratory Data Analysis", "Hyperparameter Tuning and Model Selection",
 and "Full Model Run: ".  Several steps in the script may be resource-intensive.  Data Pre-processing can be run in a single batch, however,
the other sections may require much more time, CPU, and memory.  In order to improve the timeliness of the script, a sample size of 100
was selected from the full dataset for hyperparameter tuning.  The full honeypot dataframe was ran in the last section ("Full Model Run").  Run times are exemplified below.
```
```
Hyperparameter Tuning Run time: 122 seconds

Full Model Run Times:
Country - 14 seconds
Locale (without country) - 64 seconds
Locale (with country) - 74 seconds
```
## Results
```
A pipeline with standard scaler and the selected model were initialized.  The hyperparameter tuning results return specific parameters
for the most accurate model given a variety of inputs.  The final run dynamically used the best parameter scored model result as the definition for the full model.  It was determined that country could be predicted using the PCA and Logistic Regression model (with specified parameters) with 40% accuracy.  Additionally, the model was ran for Locale (without country) and Locale (with country).  Even without including country as a known variable, the algorithm was able to maintain a 42% accuracy score.  When country was known, the model improved to a 55% accuracy score.
```
### Model Review

```
The hyperparameter tuning included several classification models that are popular, with extensive packages and support within the Python
software.  The models selected were Logistic Regression, Random Forest Classifier, Support Vector Classifier (SVC), and Principal 
Component Analysis (PCA).  Each of these were preloaded with lists of specific parameters included for each model.  The overarching model runs 
each model with all possible combinations of parameters to find the optimal model according to the scoring emphasis indicated.  This project focused on accuracy, however, precision could also be a focal point for the GridSearchCV model in Python.  Once the highest scoring model with the ideal parameters have been selected, it also fits the model with the X and y variables identified so an initial scoring is available.  In this case, combining PCA and Logistic Regression was the ideal model with an initial accuracy score of 0.35.
```

```
The final model selected was based on the highest scoring model from the hyperparameter tuning.  PCA and Logistic Regression together are capable of predicting the country class 40% of the time based on the greater region data (host), destination port attacked, month, count of samples per country, and protocol type.  The hyperparameter tuning model used only .1% of the available samples to select the model, and the accuracy was overestimated by only 5%.  Predictive accuracy improved drastically when attempting to find the locale with country data included.  Region data is important ('host' variable), and country data is somewhat common to access so it is possible that future studies could look more closely at a more granular location with reasonable success.

```

## Conclusion
```
A successful model was found in this project that could with some accuracy predict the geographical location of a malicious attacker based on other metadata information routinely collected, particularly by intrusion detection systems used in conjunction with honeypots.  This information could be very useful on a large scale when attempting to rebuff specific attack styles and protect sensitive information.  One example that was found in this study is the potential increase in attacks due to the AWS 101 release in July 2013.  A company interested in identifying it's weaknesses may want to identify company events, releases, or socioeconomic variables among others that could catch the intrigue of attackers.  Future research recommendations include focus on socioeconomic and political events that may cause motivated attackers and groups to surge attacks against a target.  It would also be of interest to determine if there are regional cultural differences among attackers, such as styles that may differ between individualistic and collectivist cultures.


```

## Authors

Samantha Roberts 

## Acknowledgements

Code from Exercise 12 was used extensively in assistance of the hyperparameter tuning chunk of script.
Other general layout was taken from the outline of other classroom exercises as well.
