#Classification

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv("C:/Users/Hi/Documents/Prep/Datasets/breast-cancer-wisconsin.data.txt")
df.replace("?", -99999, inplace = True)
df.drop(['Id'], 1, inplace = True)
df.drop(['Unnamed: 11'], 1, inplace = True)

X = np.array(df.drop['class'], 1)
y = np.array(df['class'])

X_train, y_train, X_test, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measure = np.array([4,2,4,1,1,,2,3,4,1])
example_measure = example-measure.reshape(len(example_measure),-1)
prediction = clf.predict(example_measure)
print(prediction)

#Euclidean Distance
from math import sqrt
plot1 = [1,3]
plot2 = [2, 5]

euclidean_dist = sqrt((plot1[0] - plot2[0])**2  + (plot1[1] - plot2[1])**2)
print(euclidean_dist)
