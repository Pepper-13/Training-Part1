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


#Forming K-Nearest Neighbor algo 
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use("fivethirtyeight")

dataset= {'k':[[1,2],[2,3],[3,4]], 'r':[[5,6],[7,8],[8,8]]}
new_feature = [5,7]

#Visulaise Plot 
[[plt.scatter(ii[0],ii[1], s= 100, color =i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1])
plt.show()

#Building the KNearest algo
def k_nearest_neighbors(data, predict, k =3):
    if len(data) >= k:
        warnings.warn('k is set to a lower value!')
        distances = []
        for group in data:
            for features in data[group]:
                #euc_dist = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
                #more simpliar version from numpy
                euc_dist = np.linalg.norm(np.array(features) - np.array(predict))
                distances.append([euc_dist, group])
                
        votes = [i[1] for i in sorted(distances)[:k]]
        print(Counter(votes).most_common(1))
        vote_result = Counter(votes).most_common(1)[0][0]
        
        return vote_result
    
result = k_nearest_neighbors(dataset, new_feature, k =2)
print(result)


