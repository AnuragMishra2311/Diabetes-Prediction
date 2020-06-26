#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle

#Reading File
df=pd.read_csv('Dia.csv')

#Replacing 0 with nan
def replacing_0_with_NAN():
    for i in range(1,8):
        df.iloc[:,i]=np.where(df.iloc[:,i]==0,np.nan,df.iloc[:,i])        

replacing_0_with_NAN()

#Handling Null Values
df.Glucose.fillna(df.Glucose.mean(),inplace=True)
df.BloodPressure.fillna(df.BloodPressure.mean(),inplace=True)
df.SkinThickness.fillna(df.SkinThickness.median(),inplace=True)
df.Insulin.fillna(df.Insulin.mean(),inplace=True)
df.BMI.fillna(df.BMI.median(),inplace=True)


x=df.iloc[:,:-1].values
y=df.iloc[:,8].values

# Splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


classifier=XGBClassifier(min_child_weight=1,max_depth=12,learning_rate=0.25,gamma= 0.4,colsample_bytree=0.5)

#Fitting model
classifier.fit(x_train,y_train)

#saving model to disk
pickle.dump(classifier,open('model.pkl','wb'))

#loading model to compare results
model=pickle.load(open('model.pkl','rb'))
print(model.predict(np.array([6,166.0,74.000000,29.0,153.743295,26.6,0.304,66.0]).reshape(1,-1)))
