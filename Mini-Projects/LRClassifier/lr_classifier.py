import sys, os
import numpy as np
import pandas as pd
import h5py, pickle
import math

def load_dataset(data_path = '.'):
    train_dataset = h5py.File(data_path + '/train_catvnoncat.h5') #'./Data/train_catvnoncat.h5')
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File(data_path + '/test_catvnoncat.h5')
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, -1))
    test_set_y_orig = test_set_y_orig.reshape((1, -1))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class LRClassifier:
    def __init__(self, n, c):
        self.w = np.random.rand(n, c) / math.sqrt(n)
        self.n = n
        self.c = c

    def forward_prop(self, x):
        m ,n = x.shape
        if n != self.n:
            x = x.T
        m ,n = x.shape
        if n != self.n:
            print('shape mismatch!! m : {m} and n is {n} after transpose')
            return None
        z = np.matmul(x, self.w)
        #print(z)
        a = sigmoid(z)
        a[a == 1.0] = 0.9999
        return a

    def backprop(self, x, y, a, lr=1e-2):
        dw = (a - y).T
        #print(dw.shape, x.shape)
        dw1 = dw@x
        dw1 = dw1.T / x.shape[0]
        #print((dw1*lr))
        self.w = self.w - (dw1*lr)

def accuracy(y ,a):
    thr_a = a.copy()
    thr_a[thr_a >= 0.5] = 1.0
    thr_a[thr_a < 0.5] = 0.0
    
    crr = sum(y == thr_a)
    #print(sum(thr_a))
    return crr / y.shape[0]

def lr_loss_fn(y, a):
    t1 = np.log(a)
    t2 = np.log(1 - a)
    t = y.T@t1
    t = t + ((1 - y).T@t2)
    return -t / y.shape[0]

def eval_model(y, a):
    print('accuracy ', accuracy(y, a))
    print('loss value ', lr_loss_fn(y, a)[0])

def train(lr_model, train_set_x, train_set_y_orig, test_set_x, test_set_y_orig):
    for i in range(1000):
        a = lr_model.forward_prop(train_set_x)
        lr_model.backprop(train_set_x, train_set_y_orig.T, a, lr=1e-3)
        if i%100 == 0:
            a = lr_model.forward_prop(test_set_x)
            eval_model(test_set_y_orig.T, a)    
def main():
    data_path = './Data'
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset(data_path)
    train_set_x = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1))
    test_set_x = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1))
    train_set_x = train_set_x / 255.0
    test_set_x = test_set_x / 255.0
    m = train_set_x.shape[0]
    n = train_set_x.shape[1]
    lr_classifier = LRClassifier(n , 1)
    '''
    for i in range(1000):
        a = lr_classifier.forward_prop(train_set_x)
        lr_classifier.backprop(train_set_x, train_set_y_orig.T, a, lr=1e-1)
        if i%100 == 0:
            eval_model(train_set_y_orig.T, a)
    '''
    train(lr_classifier, train_set_x, train_set_y_orig, test_set_x, test_set_y_orig)
if __name__ == '__main__':
    main()
