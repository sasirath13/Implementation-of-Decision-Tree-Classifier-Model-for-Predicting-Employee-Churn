# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SASIDHARAN P
RegisterNumber:  212223080051
*/
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
X=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
X.head()
Y=data["left"]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train,Y_train)
Y_pred=dt.predict(X_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
data.head()

![image](https://github.com/sasirath13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568449/522af085-b32b-4659-a8c8-44b456a0a839)

data.info()

![image](https://github.com/sasirath13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568449/b3ea549d-2f5a-41ad-b581-f3b1e727ec71)

isnull() and sum()

![image](https://github.com/sasirath13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568449/dbbc1472-ac3f-458a-9395-6a8687a7bcc9)

data value counts()

![image](https://github.com/sasirath13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568449/b4691f6e-5ce2-4b98-b95a-6755c64d63ab)

data.head() for salary


![image](https://github.com/sasirath13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568449/42ac836c-097f-4ae8-9cc4-f6a2d66cb021)


x.head()


![image](https://github.com/sasirath13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568449/9150e96d-5614-4456-8b5a-005293aadd81)


accuracy value()


![image](https://github.com/sasirath13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568449/1689aff5-43f9-4896-9553-c935d8631adc)

data prediction


![image](https://github.com/sasirath13/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160568449/1fe8d4e0-5c1f-4cf1-b0d5-85c435fc9fcc)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
