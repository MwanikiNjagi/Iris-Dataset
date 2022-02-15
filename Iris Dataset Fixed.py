from tkinter import Y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

Data = pd.read_csv('/home/mwanikii/Desktop/Iris1.csv')
df = Data.drop(Data.loc[:, ['Id']], axis=1)


#Labelencoder
labelencoder  = LabelEncoder()
df['Species_Encoded'] = labelencoder.fit_transform(df['Species'])
target = df['Species_Encoded']
y = target

#features
features = df[['Sums']]
X = features

print(X.shape)
print(y.shape)
#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=None)#Any int will lead to the same train test split
 
#Scatter plots
plt.scatter(X,y)
#plt.show()

#Pairplot
#sns.pairplot(data=df, hue='Species', palette='Set2')
#plt.show()
#Testing multiple algorithms
LR = LogisticRegression()
LR.fit(X_train, y_train)
predictor  = LR.predict(X_test)
print(predictor)

#Support Vector Machine
model = SVC()
model.fit(X_train, y_train)
predict = model.predict(X_test)
print(predict)

print (confusion_matrix(y_test, predict))
print (classification_report(y_test, predict))

