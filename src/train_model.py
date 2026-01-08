
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

df = pd.read_csv('E:\iris_classifier\data\iris.csv')    
df.head()

df = df.drop('Id', axis=1)

X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
print(X_test.shape)
print(X_test)

from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
y=Le.fit_transform(y)
print(y)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(acc)

cm = confusion_matrix(y_test, y_pred)
print(cm)

import joblib
joblib.dump(model, 'E:\iris_classifier\models\model.pkl')