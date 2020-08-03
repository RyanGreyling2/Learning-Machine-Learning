import pandas as pd
import numpy as np

training_data = pd.read_excel('original-data/Training-Data.xlsx')
training_data = np.array(training_data)
training_data = training_data[::, 0:-2].astype('float32')

cov = np.cov(training_data)
print(cov.shape)

# diag = np.diag(cov)

# for x in diag:
#     print(x)

# np.savetxt("modified-data/covariance.csv", cov, delimiter=" ")

# example = [[0,1], [2,3], [4,5]]

# example = np.array(example)

# print(example[::, 0:-1])