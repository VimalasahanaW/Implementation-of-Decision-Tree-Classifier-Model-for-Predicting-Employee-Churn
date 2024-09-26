# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset. 

3.Check for any null values using the isnull() function. 

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VIMALA SAHANA W
RegisterNumber:212223040241
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:
## HEAD() AND INFO():
![Screenshot 2024-09-26 142642](https://github.com/user-attachments/assets/09c16fce-8341-4bcb-9752-5ef2353a599d)

![Screenshot 2024-09-26 142707](https://github.com/user-attachments/assets/86da9c79-7a5b-4a17-a389-e5c66e67e7dc)

## NULL & COUNT:
![Screenshot 2024-09-26 142720](https://github.com/user-attachments/assets/860c0aa1-e0f2-4b91-ac29-ae33b5137765)

![Screenshot 2024-09-26 142737](https://github.com/user-attachments/assets/7af19156-dc95-441b-8c4b-0c72ca7dfa84)

![Screenshot 2024-09-26 142756](https://github.com/user-attachments/assets/dc93d913-d203-4043-b9d5-d6e36313f54b)

![Screenshot 2024-09-26 142815](https://github.com/user-attachments/assets/80bce9ff-5694-491e-8c4e-6f7a31a784dc)
## ACCURACY SCORE:
![Screenshot 2024-09-26 142825](https://github.com/user-attachments/assets/c4bbbddb-cacd-45ef-a89c-bdade1c0da33)
## DECISION TREE CLASSIFIER MODEL:
![Screenshot 2024-09-26 142836](https://github.com/user-attachments/assets/03c293d2-bb7f-4dc4-b125-45f376789113)

![Screenshot 2024-09-26 142854](https://github.com/user-attachments/assets/17da1e67-50c7-4e24-958d-3d34eeb3a0a0)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
