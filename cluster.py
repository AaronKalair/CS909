# Code for carrying out clustering experiments
# Needed for floating point division
from __future__ import division
# For converting our data into a BOW
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn import cross_validation
from sklearn import metrics
from sklearn.cross_validation import KFold
# Our Three Clustering Algorithms
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GMM
import csv

# Load in our training data from the CSV file
array = list( csv.reader( open(  'train_cluster.csv', 'rU' ) ) )
array = np.array(array)

# Set up our vectoriser that will transform the data into a BOW representation
vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 1), stop_words="english")


body =  array[:,3]
label = array[:,0]
topic1 = array[:,4]
title = array[:,2]
dateline =  array[:,1]

x = vectorizer.fit_transform(body)
new_array = x.toarray()
# Select the clustering algorithm to work with

# clf = KMeans(n_clusters=10)
# DBScan doesnt support labelling new instances so we can only report the sillouete coefficient for it
# clf = DBSCAN(eps=0.3, min_samples=1)
clf = GMM(n_components=10, covariance_type='full')

# Return the average of an array of values
def getAverage(values):
    return sum(values) / len(values)


accuracies = []
micro_precisions = []
micro_recalls = []
macro_precisions = []
macro_recalls = []

# For getting the silhouette score for the clusters
clf.fit(new_array)
labels = clf.labels_
print metrics.silhouette_score(new_array, labels)


# For extracting accuracy / precision etc from the clusters, wont work with DBScan
# kf = KFold(len(label), n_folds=10, indices=True)
# for train, test in kf:
#     x_train, x_test, y_train, y_test = new_array[train], new_array[test], label[train], label[test]
#     clf.fit(x_train)
#     res = clf.predict(x_test)
#     # Need to convert the strings to ints so we can compare the labels from the clusters to our known labels
#     y_test = [int(numeric_string) for numeric_string in y_test]
#     # Accuracy
#     accuracy = metrics.accuracy_score(y_test, res, normalize=True)
#     accuracies.append(accuracy)
#     # Micro Precision
#     micro_prec  = metrics.precision_score(y_test, res, average='micro')
#     micro_precisions.append(micro_prec)
#     # Micro Recall
#     micro_rec = metrics.recall_score(y_test, res, average='micro')
#     micro_recalls.append(micro_rec)
#     # Macro Precision
#     macro_prec = metrics.precision_score(y_test, res, average='macro')
#     macro_precisions.append(macro_prec)
#     # Macro Recall
#     macro_rec = metrics.recall_score(y_test, res, average='macro')
#     macro_recalls.append(macro_rec)

# print getAverage(accuracies)
# print getAverage(micro_precisions)
# print getAverage(micro_recalls)
# print getAverage(macro_precisions)
# print getAverage(macro_recalls)
