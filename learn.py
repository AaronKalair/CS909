# Code for carrying out machine learning experiments
# Needed for floating point division
from __future__ import division
# For converting our data into a BOW
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
import csv

# Load in our training data from the CSV file
array = np.array(list( csv.reader( open(  'train.csv', 'rU' ) ) ))

# Set up our vectoriser that will transform the data into a BOW representation
vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), stop_words="english")
# Select the releveant parameters for your test
# vectorizer = TfidfTransformer()
# binary=true
# ngram_range=(2, 2)

body =  array[:,3]
label = array[:,0]
topic1 = array[:,4]
title = array[:,2]
dateline =  array[:,1]

x = vectorizer.fit_transform(body)
# temp = vectorizer.build_analyzer()
new_array = x.toarray()
# Select the classifier to work with

# clf = svm.SVC()
#  C=1.0, kernel='linear'
# clf = GaussianNB()
clf = RandomForestClassifier(n_estimators=10)

# Return the average of an array of values
def getAverage(values):
    return sum(values) / len(values)


accuracies = []
micro_precisions = []
micro_recalls = []
macro_precisions = []
macro_recalls = []

kf = KFold(len(label), n_folds=10, indices=True)
for train, test in kf:
    x_train, x_test, y_train, y_test = new_array[train], new_array[test], label[train], label[test]
    clf.fit(x_train, y_train)
    res = clf.predict(x_test)
    # Accuracy
    accuracy = metrics.accuracy_score(y_test, res, normalize=True)
    accuracies.append(accuracy)
    # Micro Precision
    micro_prec  = metrics.precision_score(y_test, res, average='micro')
    micro_precisions.append(micro_prec)
    # Micro Recall
    micro_rec = metrics.recall_score(y_test, res, average='micro')
    micro_recalls.append(micro_rec)
    # Macro Precision
    macro_prec = metrics.precision_score(y_test, res, average='macro')
    macro_precisions.append(macro_prec)
    # Macro Recall
    macro_rec = metrics.recall_score(y_test, res, average='macro')
    macro_recalls.append(macro_rec)

print getAverage(accuracies)
print getAverage(micro_precisions)
print getAverage(micro_recalls)
print getAverage(macro_precisions)
print getAverage(macro_recalls)
