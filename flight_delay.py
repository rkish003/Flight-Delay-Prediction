import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import tabula as tb
from sklearn.preprocessing import LabelEncoder
import re
import os
import random
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB,CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from pandas.io.pytables import DataFrame
import warnings
import pickle
import json
warnings.filterwarnings("ignore")



df=pd.read_csv('./DSA dataset/data_final.csv')

print(df.shape)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes('object').columns.tolist()


req_cols=['DEP_AIRPORT','ARR_AIRPORT','AIRLINE','ARR_STATUS']
df['ARR_STATUS']= pd.cut(df['ARR_DELAY'], bins=[-999,0,999], include_lowest=True, labels=['Early','Delayed'])
le = LabelEncoder()
mapping_dict = {}
for col in req_cols:
    df[col] = le.fit_transform(df[col])
 
    le_name_mapping = dict(zip(le.classes_,
                               le.transform(le.classes_)))
 
    mapping_dict[col] = le_name_mapping

dep_air_list=list(mapping_dict['DEP_AIRPORT'].keys())
arr_air_list=list(mapping_dict['DEP_AIRPORT'].keys())
airline_list=list(mapping_dict['AIRLINE'].keys())


fture_clmns = ['DEP_AIRPORT','ARR_AIRPORT','AIRLINE']
X = df[fture_clmns]
y = df["ARR_STATUS"]
df = df.dropna(subset=['ARR_STATUS'])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=124)


GNB = RandomForestClassifier(n_estimators=100,random_state=0)
GNB.fit(X_train,y_train)
y_pred=GNB.predict(X_test)
print("Accuracy is :",accuracy_score(y_test,y_pred))


pickle.dump(GNB,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))







