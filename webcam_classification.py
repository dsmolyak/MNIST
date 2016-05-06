__author__ = 'smolydb1'

import cv2
import time
import numpy as np
import cPickle


fid = open('my_poly_classifier.pkl', 'rb')
clf2 = cPickle.load(fid)

def show_contours(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_th = cv2.GaussianBlur(im_gray, (5, 5), 0)
    ret, im_th = cv2.threshold(im_th, 90, 255, cv2.THRESH_BINARY_INV)
    roi, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    for rect in rects:
        cv2.rectangle(im_th, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 255), 3)
    cv2.imshow('Contours', im_th)

def convert_to_data(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_th = cv2.GaussianBlur(im_gray, (5, 5), 0)
    ret, im_th = cv2.threshold(im_th, 90, 255, cv2.THRESH_BINARY_INV)
    roi, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print len(ctrs)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    for rect in rects:
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        leng = int(rect[3] * 1.5)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
        if (pt1 < 0):
            break
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        img_flt = roi.flatten()
        my_image = [img_flt]
        predicted = clf2.predict(my_image)
        cv2.putText(img, str(int(predicted[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    cv2.imshow('Classifications and Rectangles', img)

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
    t1 = time.clock()
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        show_contours(img)
        if cv2.waitKey(5) == 13:
            convert_to_data(img)
        if cv2.waitKey(5) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()

def main():
    show_webcam(mirror=False)

if __name__ == '__main__':
    main()