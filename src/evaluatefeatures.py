import pandas as pd
import numpy as np

unimportant_features = np.genfromtxt("../temp-data/unimportant_features.csv", delimiter=",")
unimportant_features = np.array(unimportant_features)
unimportant_features = unimportant_features.astype('int64')

eval_data = pd.read_excel('../original-data/Evaluation-Data.xlsx')
columns = eval_data.columns

eval_data = eval_data.replace({"XC": {'A': 1,'B': 2,'C': 3,'D': 4,'E': 5}})

eval_data = np.array(eval_data)

eval_data = np.delete(eval_data, unimportant_features, 1)
columns = np.delete(columns, unimportant_features, 0)

eval_data = pd.DataFrame(data=eval_data, columns=columns)
eval_data.to_excel("../modified-data/evaluatefeatures.xlsx", startcol=-1)
