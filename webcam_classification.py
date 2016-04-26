__author__ = 'smolydb1'

import cv2
import time
from sklearn import svm, metrics
from sklearn.datasets import fetch_mldata
from sklearn.svm import LinearSVC
import numpy as np
import cPickle


fid = open('my_dumped_classifier.pkl', 'rb')
clf2 = cPickle.load(fid)

def convert_to_data(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    im_th = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)
    ret, im_th = cv2.threshold(im_th, 100, 255, cv2.THRESH_BINARY_INV)

    # im_th = binary_invert(im_th)
    img_flt = im_th.flatten()
    cv2.imshow('cool', im_th)
    # print unflatten(img_flt)

    # with open('my_dumped_classifier.pkl', 'rb') as fid:

    my_image = [img_flt]
    # cv2.imshow('wow', unflatten(data[9]))
    predicted = clf2.predict(my_image)
    im = cv2.imread("/home/smolydb1/Projects/MNIST/white.png")
    cv2.putText(im, str(predicted[0]), (100,100), cv2.FONT_HERSHEY_TRIPLEX, 10, (255,255,255), 5)
    cv2.imshow("Predicted", im)
    print predicted
        # print unflatten(data[5])

def binary_invert(img):
    for i in range(0,27):
        for j in range(0,27):
            if img[i][j] > 100:
                img[i][j] = 255
    return img

def unflatten(img):
    matrix = np.zeros((28,28))
    for i in range(0,27):
        for j in range(0,27):
            num = i * 28 + j
            matrix[i][j] = img[num]
    return matrix


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)

        convert_to_data(img)
        cv2.imshow('my webcam', img)
        # if cv2.waitKey(1) == 13:

            # while True:
        if cv2.waitKey(5) == 27:
            break  # esc to quit
        # if cv2.waitKey(1) == 27:
        #     break  # esc to quit

    cv2.destroyAllWindows()

def main():
    show_webcam(mirror=False)

if __name__ == '__main__':
    main()