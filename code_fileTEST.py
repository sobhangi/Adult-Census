import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#---IMPORTING DATA SET---
dataset = pd.read_csv('adult_data.csv')
X= dataset.iloc[:,:-1].values
Y=dataset.iloc[:,14].values

#---IMPORTING TEST DATA SET---
dataset2 = pd.read_csv('BOOK.csv')
X_new= dataset.iloc[:,:-1].values
Y_new=dataset.iloc[:,14].values

#---LABEL ENCODING FOR STRING VALUES---
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labenc_X = LabelEncoder()
X[:,1]=labenc_X.fit_transform(X[:,1])
X[:,3]=labenc_X.fit_transform(X[:,3])
X[:,5]=labenc_X.fit_transform(X[:,5])
X[:,6]=labenc_X.fit_transform(X[:,6])
X[:,7]=labenc_X.fit_transform(X[:,7])
X[:,8]=labenc_X.fit_transform(X[:,8])
X[:,9]=labenc_X.fit_transform(X[:,9])
X[:,13]=labenc_X.fit_transform(X[:,13])
labenc_Y = LabelEncoder()
Y=labenc_Y.fit_transform(Y)

#---LABEL ENCODING FOR STRING VALUES OF TEST CASE---
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labenc_X1 = LabelEncoder()
X_new[:,1]=labenc_X1.fit_transform(X_new[:,1])
X_new[:,3]=labenc_X1.fit_transform(X_new[:,3])
X_new[:,5]=labenc_X1.fit_transform(X_new[:,5])
X_new[:,6]=labenc_X1.fit_transform(X_new[:,6])
X_new[:,7]=labenc_X1.fit_transform(X_new[:,7])
X_new[:,8]=labenc_X1.fit_transform(X_new[:,8])
X_new[:,9]=labenc_X1.fit_transform(X_new[:,9])
X_new[:,13]=labenc_X1.fit_transform(X_new[:,13])
labenc_Y1 = LabelEncoder()
Y_new=labenc_Y1.fit_transform(Y_new)


#---HANDLING MISSING TEST DATA---
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values="NaN",strategy="mean",axis=0)
imp.fit(X)
X=imp.transform(X)

#---HANDLING MISSING TEST DATA---
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values="NaN",strategy="mean",axis=0)
imp.fit(X_new)
X_new=imp.transform(X_new)

#---FEATURE SCALING---
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)

#---FEATURE SCALING TEST CASE---
from sklearn.preprocessing import StandardScaler
sc_X1=StandardScaler()
X_new=sc_X1.fit_transform(X_new)

#---APPLYING THE GAUSSIAN BAYES---
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() 
classifier.fit(X,Y)

#---PREDICT RESULT---
y_pred = classifier.predict(X_new)

#creating confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_new,y_pred)

from sklearn.metrics import accuracy_score
percentage_accuracy=accuracy_score(Y_new,y_pred)*100