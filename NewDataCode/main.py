from functions import get_data,build_model
import pandas as pd
import numpy as np
from HyperParameterTune import tune_model#,GuidedTuneModel

#First column

ColName = 'action_taken'
data = get_data(ColName,nr = 100000)
s1 = data[data[ColName] == 1].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 3].shape[0]))
s2 = data[data[ColName] == 3].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 3].shape[0]))
d = pd.concat([s1,s2])
y = d[ColName]
X = d.drop(columns = [ColName])
# build_model(X,y,cross = 10,models = ['nvb','RandomForest','xgb','Logistic'])
# GuidedTuneModel(X,y)
tune_model(X,y,n_it = 50,models = ['RandomForest','xgb','Logistic'])

ColName = 'denial_reason_1'
data = get_data(ColName,nr = 100000)
s1 = data[data[ColName] == 1].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 3].shape[0]))
s2 = data[data[ColName] == 3].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 3].shape[0]))
d = pd.concat([s1,s2])
y = d[ColName]  
X = d.drop(columns = [ColName])
# build_model(X,y,cross = 10,models = ['nvb','RandomForest','xgb','Logistic'])
tune_model(X,y,n_it = 50,models = ['RandomForest','xgb','Logistic'])

ColName = 'denial_reason_2'
data = get_data(ColName,nr = 100000)
s1 = data[data[ColName] == 1].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 3].shape[0]))
s2 = data[data[ColName] == 3].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 3].shape[0]))
d = pd.concat([s1,s2])
y = d[ColName]
X = d.drop(columns = [ColName])
# build_model(X,y,cross = 10,models = ['nvb','RandomForest','xgb','Logistic'])
tune_model(X,y,n_it = 50,models = ['RandomForest','xgb','Logistic'])

ColName = 'denial_reason_3'
data = get_data(ColName,nr = 100000)
s1 = data[data[ColName] == 1].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 9].shape[0]))
s2 = data[data[ColName] == 9].sample(n = np.minimum(data[data[ColName] == 1].shape[0],data[data[ColName] == 9].shape[0]))
d = pd.concat([s1,s2])
y = d[ColName]
X = d.drop(columns = [ColName])
# build_model(X,y,cross = 10,models = ['nvb','RandomForest','xgb','Logistic'])
tune_model(X,y,n_it = 50,models = ['RandomForest','xgb','Logistic'])

#Pipeline to predict loan/no-loan and then predict reason

