import pandas as pd
import numpy as np

training_data = pd.read_excel('../original-data/Training-Data.xlsx')
training_data = training_data.replace({"XC": {'A': 1,'B': 2,'C': 3,'D': 4,'E': 5}})

training_data = np.array(training_data)
training_data = training_data[::, 0:-1].astype('float32')

cov = np.corrcoef(training_data, rowvar=False)
# print(cov)

# diag = np.diag(cov)

# for x in diag:
#     print(x)

np.savetxt("../modified-data/correlation.csv", cov, delimiter=" ")

# example = [[0,1], [2,3], [4,5]]

# example = np.array(example)

# print(example[::, 0:-1])