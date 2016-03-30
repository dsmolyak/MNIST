__author__ = 'smolydb1'

import time
import csv
import argparse
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics
from sklearn.datasets import fetch_mldata
from sklearn.svm import LinearSVC
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

# let the user select a scale for the image size, default is 2
ap.add_argument("-s", "--start", required=False, default=100, help="number of images to start training")

ap.add_argument('-f', "--factor", required=False, default=2, help="geometric series")

ap.add_argument('-d', "--difference", required=False, default=100, help="arithmetic series")

# parse the arguments
args = vars(ap.parse_args())
start = int(args['start'])
factor = int(args['factor'])
difference = int(args['difference'])
# out_img_dir = args['out_dir']

geometric = True
if factor == 0:
    geometric = False

ofile  = open('/home/smolydb1/Projects/MNIST/accuracy_data.csv', "wb")
writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
row = ["Training","Precision","Recall","F1-score"]
writer.writerow(row)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names)
    plt.yticks(tick_marks, label_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

dataset = fetch_mldata('MNIST Original', data_home='/home/smolydb1/Documents/Datasets/')
features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')

# shuffle the data
rand = np.random.RandomState(0)
shuffle = rand.permutation(len(features))
data, labels = features[shuffle], labels[shuffle]

data_train, labels_train = data[0:60000], labels[0:60000]
data_test, labels_test = data[60000:], labels[60000:]


def test_classifier(i, classifier):
    expected = labels_test[0:len(labels_test)]
    predicted = classifier.predict(data_test[0:len(data_test)])
    cm = metrics.confusion_matrix(expected, predicted)
    cr = metrics.classification_report(expected, predicted)
    clf_scores = [str(i),cr[603:607],cr[613:617],cr[623:627]]
    # print 'Precision: ' + cr[603:607]
    # print 'Recall: ' + cr[613:617]
    # print 'f1-score: ' + cr[623:627]
    writer.writerow(clf_scores)
    print("Classification report:\n%s\n" % (cr))
    print("Confusion matrix:\n%s\n\n" % (cm))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix: ' + str(i))

i = 100
classifiers = []
while i < 60000:
    print i
    clf = LinearSVC()
    clf.fit(data_train[0:i], labels_train[0:i])
    test_classifier(i,clf)
    if geometric:
        i *= factor
    else:
        i += difference

plt.show()

# t1 = time.clock()
# classifier.fit(data_train[0:n_samples], labels_train[0:n_samples])
# t2 = time.clock()
# total = t2-t1
# print 'Total Time to Train: ' + str(total) + ' seconds\n'

# Now predict the value of the digit on 0 to n_tests
# expected = labels_test[0:n_tests]
# predicted = classifier.predict(data_test[0:n_tests])

# cm = metrics.confusion_matrix(expected, predicted)
# print("Classification report:\n%s\n"
#       % (metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s\n\n" % (cm))

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print('Normalized confusion matrix:')
# print(cm_normalized)
# plt.figure()
# plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

# plt.figure()
# plot_confusion_matrix(cm)
# plt.show()