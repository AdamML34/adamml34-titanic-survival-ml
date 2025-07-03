import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
data=pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
names=data["Name"]
data=data.drop("Name",axis=1)
data=data.drop("Cabin",axis=1)
data=data.drop("Ticket",axis=1)
data=data.drop("PassengerId",axis=1)
m=data["Age"].median()
data["Age"]=data["Age"].fillna(m)
l=data["Embarked"].mode()[0]
data["Embarked"]=data["Embarked"].fillna(l)
print(data.groupby("Pclass")["Age"].mean())
print(data.groupby("Sex")["Survived"].mean())
print(data.groupby("Survived")["Fare"].mean())
print(pd.crosstab(index=data["Survived"],columns=data["Pclass"]))
def risk(row):
    if row["Age"]<18:
        return "young"
    elif row["Fare"]>100 and row["Pclass"]==1:
        return "rich"
    else:
        return "standard"
data["risk_category"]=data.apply(risk,axis=1)
data=pd.get_dummies(data,columns=["Sex"])
data=pd.get_dummies(data,columns=["Embarked"])
data=pd.get_dummies(data,columns=["risk_category"])
x=data.drop("Survived",axis=1)
y=data["Survived"]
pipe=Pipeline([("scaler",StandardScaler()),("model",RandomForestClassifier(n_estimators=100,n_jobs=-1,verbose=1,random_state=42,max_depth=4,min_samples_leaf=6,ccp_alpha=0.001))])
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.15,random_state=42)
pipe.fit(x_train,y_train)
y_est_val=pipe.predict(x_val)
y_est_test=pipe.predict_proba(x_val)[:,1]
print(confusion_matrix(y_val,y_est_val))
print(classification_report(y_val,y_est_val))
print(roc_auc_score(y_val,y_est_test))
joblib.dump(pipe,"model")