import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv("/content/Dataset_watering.csv")
col=["Air temperature (C)","Wind speed (Km/h)","Air humidity (%)","Wind gust (Km/h)","Pressure (KPa)"]
for i in col:
  df[i].fillna(df[i].median(),inplace=True)

df["Status"] = df["Status"].str.strip().str.lower()
df["Status"] = df["Status"].map({"on": 1, "off": 0})

col=["Soil Moisture","Temperature"," Soil Humidity","Time","Air temperature (C)","Wind speed (Km/h)","Air humidity (%)","Wind gust (Km/h)","Pressure (KPa)"]
y=df["Status"]
x=df[col]
x_train=x
y_train=y

model1=LogisticRegression(max_iter=3000)
model1.fit(x_train,y_train)

model2=RandomForestClassifier(n_estimators=450,max_depth=13,random_state=30,class_weight="balanced")
model2.fit(x_train,y_train)

import joblib
joblib.dump(model1,"logistic_plantWater.pkl")
joblib.dump(model2,"randomForest_plantWater.pkl")
