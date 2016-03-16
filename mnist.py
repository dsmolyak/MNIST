__author__ = 'smolydb1'

import time
# Standard scientific Python imports
import matplotlib.pyplot as plt
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import numpy as np

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

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


# The digits dataset: Each image is 28x28 pixels
images, labels = load_mnist(dataset="training", path="/home/smolydb1/Documents/Datasets/MNIST")

images_and_labels = list(zip(images, labels))
# print(images_and_labels[0])

for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(images)/1000
data = images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

t1 = time.clock()
# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples / 2], labels[:n_samples / 2])
t2 = time.clock()
total = t2-t1

print 'Total Time to Train: ' + str(total) + ' seconds'

# Now predict the value of the digit on the second half:
t1 = time.clock()
expected = labels[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])
t2 = time.clock()
total = t2-t1

print 'Total Time to Predict: ' + str(total) + ' seconds'

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(images[n_samples / 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()