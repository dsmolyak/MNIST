__author__ = 'smolydb1'

import time
import argparse
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_mldata
from sklearn.svm import LinearSVC
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

# let the user select a scale for the image size, default is 2
ap.add_argument("-n", "--train", required=False, default=1000, help="number of images for training")

ap.add_argument('-t', "--test", required=False, default=10000, help="number of images for testing")

# parse the arguments
args = vars(ap.parse_args())
n_samples = int(args['train'])
n_tests = int(args['test'])
# out_img_dir = args['out_dir']


dataset = fetch_mldata('MNIST Original', data_home='/home/smolydb1/Documents/Datasets/')
features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')

# shuffle the data
rand = np.random.RandomState(0)
shuffle = rand.permutation(len(features))
data, labels = features[shuffle], labels[shuffle]

data_train, labels_train = data[0:60000], labels[0:60000]
data_test, labels_test = data[60000:], labels[60000:]

# Create a classifier: a support vector classifier
classifier = LinearSVC()

#Learn digits on 0 to n_samples
t1 = time.clock()

classifier.fit(data_train[0:n_samples], labels_train[0:n_samples])
t2 = time.clock()
total = t2-t1

# print 'Total Time to Train: ' + str(total) + ' seconds\n'


# Now predict the value of the digit on 0 to n_tests

expected = labels_test[0:n_tests]
predicted = classifier.predict(data_test[0:n_tests])

cm = metrics.confusion_matrix(expected, predicted)

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

print("Classification report:\n%s\n"
      % (metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s\n\n" % (cm))

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print('Normalized confusion matrix:')
# print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

# plt.figure()
# plot_confusion_matrix(cm)
plt.show()