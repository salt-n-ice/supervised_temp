from math import gamma
from numpy.core.numeric import cross
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
import collections
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
                                        # Transformers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
                                        # Modeling Evaluation
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, classification_report
                                        # Pipelines
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

                                        # ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
from sklearn.model_selection import cross_val_score

def make_hash():
    return collections.defaultdict(dict)



                                        # input data

train_df = pd.read_csv(os.getcwd()+"/train.csv")
test_df = pd.read_csv(os.getcwd()+"/test.csv")

                                        # preprocessing data

r_common = ["PassengerId", "Name", "Ticket", "Cabin"]
r_test = r_common.copy()
r_common.append("Survived")
r_train = r_common

y=train_df["Survived"]

id = test_df["PassengerId"]

train_df = train_df.drop(r_train, axis = 1)
test_df = test_df.drop(r_test, axis = 1)
X = train_df

nume_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
alpha_nume_cols = ["Sex", "Embarked"]

def complete_the_data(X):
    thresh_hold = 5
    for col in X.columns:
        tot_null = X.loc[:, col].isnull().sum()
        tot = len(X.loc[:, col])
        if((tot_null*100)/tot>thresh_hold):
            X[col+"_was_missing"] = X[col].isnull()
                                # flexible imputing
    def complete_nume(df, cols):
        for key, col in df.loc[:, cols].items():
            key_mean = df[key].mean()
            key_std = df[key].std()
            for i in col.index:
                if pd.isnull(df.loc[i, key]):
                    df.loc[i, key] = random.uniform(key_mean-key_std, key_mean+key_std)
        return df

    def complete_alpha_nume(df, cols):
        for key, col in df.loc[:, cols].items():
            key_mode = df[key].mode()[0]
            for i in col.index:
                if pd.isnull(df.loc[i, key]):
                    df.loc[i, key] = key_mode
        return df

    nume_imputer = SimpleImputer(strategy = "mean")
    alpha_nume_imputer = SimpleImputer(strategy = "most_frequent")
    X[nume_cols] = nume_imputer.fit_transform(X[nume_cols])
    X[alpha_nume_cols] = alpha_nume_imputer.fit_transform(X[alpha_nume_cols])
    return X

X = complete_the_data(X)
test_df = complete_the_data(test_df)


def convert_to_numbers(df):
    features_map = make_hash()
    for key in df.keys():
        x = 0
        for j in df.loc[:, key].unique():
            features_map[key][j] = x
            x+=1 
    for key, col in df.items():
        for i in col.index:
            df.loc[i, key] = features_map[key][df.loc[i, key]]
    return df

X[alpha_nume_cols] = convert_to_numbers(X[alpha_nume_cols].copy())
test_df[alpha_nume_cols] = convert_to_numbers(test_df[alpha_nume_cols].copy())

                                                # scale data
def scale_it_up(df, type):
    if(type=="standard"):
        scale = StandardScaler()
    elif(type=="minmax"): 
        scale = MinMaxScaler()
    df= scale.fit_transform(df)
    return df
X[nume_cols+alpha_nume_cols] = scale_it_up(X[nume_cols+alpha_nume_cols].copy(), "standard")
test_df[nume_cols+alpha_nume_cols] = scale_it_up(test_df[nume_cols+alpha_nume_cols].copy(), "standard")
                                                # now apply ML models


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
def best_classifier():
    models = [RandomForestClassifier()]
    names = ["rforest"]
    best_clf = np.NaN
    best_clf_name = ""
    best_score = 0
    for clf, name in zip(models, names):
        cv = cross_val_score(clf, X_train, y_train, cv = KFold())
        if(cv.mean()>best_score):
            best_score = cv.mean()
            best_clf = clf
            best_clf_name = name
    return {"name": best_clf_name, "clf": best_clf, "score": best_score}


def best_classifier_params():
    models = { LogisticRegression() : {'max_iter' : [2000],'penalty' : ['l1', 'l2'],'C' : np.logspace(-4, 4, 20),'solver' : ['liblinear']},
               KNeighborsClassifier() : {'n_neighbors' : [3,5,7,9],'weights' : ['uniform', 'distance'],'algorithm' : ['auto', 'ball_tree','kd_tree'],'p' : [1,2]},
               RandomForestClassifier(random_state=3): {'n_estimators': [400,450,500,550],'criterion':['gini','entropy'],'bootstrap': [True],'max_depth': [15, 20, 25],'max_features': ['auto','sqrt', 2],'min_samples_leaf': [2,3],'min_samples_split': [2,3]},
               SVC(probability=True) : [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],'C': [.1, 1, 10, 100]},{'kernel': ['linear'], 'C': [.1, 1, 10]},{'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100]}],
               
                
            }
    names = ["logreg", "knn","rforest", "svc"]
    ret = []
    best = {}
    best_score = 0
    for clf,param_grid, name in zip(models.keys(), models.values(), names):
        if(["rforest", "svc"].count(name)==0):
            searched_clf = GridSearchCV(clf, param_grid=param_grid, cv = 5, verbose=True, n_jobs=-1)
        else:
            searched_clf = RandomizedSearchCV(clf, param_grid, n_iter=10, cv = 5, verbose=True, n_jobs =-1)
        fit_clf = searched_clf.fit(X_train, y_train)
        curr = {"name": name, "clf" : clf, "score": fit_clf.best_score_, "best_params": fit_clf.best_params_}
        print(curr)
        ret.append(curr)
        if(curr["score"]>best_score):
            best = curr
            best_score = curr["score"]
    return best
    

# param_tuning = best_classifier_params()
# print("Best of the best(probably): ")
# print(param_tuning)

# clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, min_samples_split=2, max_features=2, max_depth=25, criterion="entropy", bootstrap=True)
# clf.fit(X, y)
# print(f"score on test data : {clf.score(X_test, y_test)}")
# print(f"score on train data : {clf.score(X_train, y_train)}")

# prediction = clf.predict(test_df)
# print(prediction)
# output = pd.DataFrame({"PassengerId" : id, "Survived" : prediction})
# output.to_csv("output.csv", index = False)