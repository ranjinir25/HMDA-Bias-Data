"""
Necessary imports

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as skm
import pickle
from sklearn import svm
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB


def get_data(drop1,nr = 10000,nona = 1):
    data = pd.read_csv(r"C:\Users\tsmur\Desktop\Spring2020\StatDataMining\Project\hmda_2017_nationwide_all-records_codes.csv", nrows = nr)
    data.replace(' ', None)
    data.replace('', None)
    c = ['rate_spread','sequence_number','co_applicant_race_5','agency_code',
       'co_applicant_race_4', 'co_applicant_race_3','applicant_race_4','census_tract_number',
       'applicant_race_3', 'applicant_race_2','denial_reason_3', 'denial_reason_2',
         'denial_reason_1','action_taken','county_code', 'purchaser_type',
       'applicant_race_1','applicant_race_5','as_of_year',
       'co_applicant_race_2', 'co_applicant_race_1','application_date_indicator',
         'respondent_id','state_code','edit_status','msamd']
    c.remove(drop1)
    data = data.drop(columns = c)
    print(np.shape(data))
    if nona == 1:
        data = data.dropna()
#     names = ['agency_abbr','agency_name','applicant_ethnicity_name','co_applicant_sex_name',
#              'co_applicant_ethnicity_name','preapproval_name','applicant_sex','county_name',
#              'hoepa_status_name','lien_status_name','loan_purpose_name','loan_type_name','owner_occupancy_name',
#              'property_type_name','purchaser_type_name','state_abbr']
#     for i in names:
#         data = pd.concat([data,pd.get_dummies(data[i], prefix=i)],axis=1)
#         data.drop([i],axis=1, inplace=True)
#     print(data.shape)
#     mapping = dict(zip(data[drop1].unique(),[i for i in range(len(data[drop1].unique()))]))
#     print(mapping)
#     data = data.replace({ColName: mapping})
#     for i in data.columns:
#         data[i].astype(int)
    data[drop1].hist()
    return data

def post_proc(X,model):
    """
    Need Post processing support like
    1. What to do with the output of feature importance
    """
    for ind,val in zip(X.columns,model.feature_importances_*100):
        if val>1:
            print(ind,val)

def build_model(X,y,cross = 5,models = ['xgb']):
    """
    Need support for more models, along with cross validation and feature importances which can be easily taken out
    something like
    build_model(X,y,cross = 5,model)
        if model == 'xgb':
            ...
        if model == 'logistic'
            ...
    """
    seed = 7
    test_size = 0.30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    for model1 in models:
        if model1 == 'xgb':
            print("XGBoost Classifier: \n")
            model = XGBClassifier()
            model.fit(X_train,y_train)
            joblib.dump(model, 'xgb.pkl')
            pred = model.predict(X_test)
            print("Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
            results = cross_val_score(model, X_train, y_train, cv=cross,scoring = 'balanced_accuracy')
            print("Cross Validation Balanced Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
            post_proc(X,model)
        if model1 == 'Logistic':
            print("\n Logistic Classifier: \n")
            model = LogisticRegression(solver = 'liblinear')
            model.fit(X_train, y_train)
            joblib.dump(model, 'logi.pkl')
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)
            print("Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
            results = cross_val_score(model, X_train, y_train, cv=cross,scoring = 'balanced_accuracy')
            print("Cross Validation Balanced Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
            cm = confusion_matrix(y_test, pred)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(cm)
            ax.grid(False)
            ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
            ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
            ax.set_ylim(1.5, -0.5)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
            plt.show()
            Logi = pickle.dumps(model) 
        if model1 == 'auto':
            print("\n Auto: \n")
            tpot = TPOTClassifier(verbosity=2, scoring = 'balanced_accuracy')
            tpot.fit(X_train, y_train)
            print(tpot.score(X_test, y_test))
        if model1 == 'SVM':
            print("\n SVM: \n")
            model = svm.NuSVC(gamma='auto')
            model.fit(X_train,y_train)
            joblib.dump(model, 'svm.pkl')
            pred = model.predict(X_test)
            print("Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
            results = cross_val_score(model, X_train, y_train, cv=cross,scoring = 'balanced_accuracy')
            print("Cross Validation Balanced Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
        if model1 == 'RandomForest':
            print("\n Random Forest: \n")
            model = RandomForestClassifier()
            model.fit(X_train,y_train)
            joblib.dump(model, 'rf.pkl')
            pred = model.predict(X_test)
            print("Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
            results = cross_val_score(model, X_train, y_train, cv=cross,scoring = 'balanced_accuracy')
            print("Cross Validation Balanced Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
        if model1 == 'nvb':
            print("\n Naive Bayes Classifier: \n")
            model = GaussianNB()
            model.fit(X_train,y_train)
            joblib.dump(model, 'nvb.pkl')
            pred = model.predict(X_test)
            print("Balanced Accuracy is ",balanced_accuracy_score(y_test,pred)*100)
            results = cross_val_score(model, X_train, y_train, cv=cross,scoring = 'balanced_accuracy')
            print("Cross Validation Balanced Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
        else:
            print(model1, "- Name not detected. Try using one of the models that are defined")
if __name__ == "__main__":
    ColName = 'action_taken_name'
    data,mapping = get_data(ColName,nr = 1000)
    y = data[ColName]
    X = data.drop(columns = [ColName])
    build_model(X,y,cross = 10,models = ['xgb','Logistic'])
    