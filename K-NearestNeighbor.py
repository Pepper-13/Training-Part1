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
