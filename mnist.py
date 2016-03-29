__author__ = 'smolydb1'

import time
# Standard scientific Python imports
import matplotlib.pyplot as plt
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np
from sklearn.svm import LinearSVC

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics


def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 'test10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'test10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


# The digits datasets: Each image is 28x28 pixels
images_train, labels_train = load_mnist(dataset="training", path="/home/smolydb1/Documents/Datasets/MNIST")
images_test, labels_test = load_mnist(dataset="testing", path="/home/smolydb1/Documents/Datasets/MNIST")

# Reformat labels_train
labels_formatted_train = np.zeros(len(labels_train))
for i in range(0, len(labels_train)):
    labels_formatted_train[i] = labels_train[i][0]
labels_train = labels_formatted_train

# Reformat labels_test
labels_formatted_test = np.zeros(len(labels_test))
for i in range(0, len(labels_test)):
    labels_formatted_test[i] = labels_test[i][0]
labels_test = labels_formatted_test

# images_and_labels_train = list(zip(images_train, labels_train))

# for index, (image, label) in enumerate(images_and_labels_train[:4]):
#     plt.subplot(2, 4, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
# n_samples_train = 200 #len(images_train)/100
data_train = []

for img in images_train:
    data_train.append(img.flatten())

# data_train = np.asarray(data_train)


data_test = []

for img in images_test:
    data_test.append(img.flatten())

# data_test = np.asarray(data_test)

print data_train[0]

# Create a classifier: a support vector classifier
classifier = LinearSVC()

t1 = time.clock()
# We learn the digits on the first half of the digits
classifier.fit(data_train[0:1000], labels_train[0:1000])
t2 = time.clock()
total = t2 - t1

print 'Total Time to Train: %s seconds' % (str(total))

# Now predict the value of the digit on the second half:
t1 = time.clock()
expected = labels_test[0:200]
predicted = classifier.predict(data_test[0:200])
t2 = time.clock()
total = t2 - t1

# expected_and_predicted = list(zip(expected,predicted))
# for i in range (len(expected_and_predicted)):
#     print expected_and_predicted[i]

print 'Total Time to Predict: %s seconds' % (str(total))

# print 'Predicted Size and Expected Size: (%d, %d)' % (len(predicted),len(expected))

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# images_and_predictions = list(zip(images_test[:n_samples_test], predicted))

# print(images_and_predictions[0])

# for index, (image, prediction) in enumerate(images_and_predictions[:4]):
#     plt.subplot(2, 4, index + 5)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Prediction: %i' % prediction)
#
# plt.show()
