from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd   
import numpy as np

data = pd.read_excel('../original-data/Training-Data.xlsx')
index = data.index
columns = data.columns

data = data.replace({"XC": {'A': 1,'B': 2,'C': 3,'D': 4,'E': 5}})

data = np.array(data)
# data = np.delete(data, -2, 1)
# columns = np.delete(columns, -2, 0)
features = data[::,:-1:].astype('float32')
target = data[::,-1].astype('int64')

estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=16, step=1)
selector = selector.fit(features, target)
# print(selector.support_)
# print(selector.ranking_)

unimportant_features = []

for i, importance in enumerate(selector.support_,0):
    if not importance:
        unimportant_features.append(i)


np.savetxt("../temp-data/unimportant_features.csv", unimportant_features, delimiter=",")

features = np.delete(features, unimportant_features, 1)
columns = np.delete(columns, unimportant_features, 0)

target = np.reshape(target, (-1, 1))
modified_data = np.concatenate((features, target), axis=1)

modified_data = pd.DataFrame(data=modified_data, columns=columns)
modified_data.to_excel("../modified-data/featureselected.xlsx", startcol=-1)
