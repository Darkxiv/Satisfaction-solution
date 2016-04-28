import csv
import numpy as np
import math
import random
import re
from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

with open('train.csv', 'rb') as f:
	train = list(csv.reader(f))

train.remove(train[0])

target = [int(d[-1]) for d in train]
features = [[float(v) for v in d[1:-1]] for d in train]

minv = features[0][:]
maxv = features[0][:]
for fl in features:
    length = len(fl)
    for i in range(0, length):
        if fl[i] < minv[i]:
            minv[i] = fl[i]
        elif fl[i] > maxv[i]:
            maxv[i] = fl[i]

for fl in features:
    length = len(fl)
    for i in range(0, length):
        if (maxv[i] - minv[i]):
            fl[i] = (fl[i] - minv[i]) / (maxv[i] - minv[i])

dfrom = 0
dto = len(features)

X, y = features[dfrom:dto], target[dfrom:dto]
model = OneVsRestClassifier(linear_model.LinearRegression()).fit(X, y)

result = model.predict(X)

true_positive = 0
for i in range(0, len(y)):
    if result[i] == y[i]:
        true_positive += 1
precision = float(true_positive) / len(y)
print precision

with open('test.csv', 'rb') as f:
	test = list(csv.reader(f))
test.remove(test[0])

features = [[float(v) for v in d[1:]] for d in test]

for fl in features:
    length = len(fl)
    for i in range(0, length):
        if fl[i] < minv[i]:
            minv[i] = fl[i]
        elif fl[i] > maxv[i]:
            maxv[i] = fl[i]

for fl in features:
    length = len(fl)
    for i in range(0, length):
        if (maxv[i] - minv[i]):
            fl[i] = (fl[i] - minv[i]) / (maxv[i] - minv[i])

predict = model.predict(features)

with open('result.csv', 'w') as csvfile:
    fieldnames = ['ID', 'TARGET']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(0, len(features)):
        writer.writerow({'ID': int(test[i][0]), 'TARGET': int(predict[i])})
