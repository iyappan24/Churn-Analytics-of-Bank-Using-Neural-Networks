
#import the necessary libraries for examining the dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#reading the dataset

df = pd.read_csv('Churn_Modelling.csv')



x = df.iloc[:,3:13].values
#loading the independant variables as an numpy array into X 

y = df.iloc[:,13].values
#loading the final classfier column( feature ) which is to predicted as numpy array



#importing the labelencoder for processing te 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder




lx_1 = LabelEncoder()
lx_2 = LabelEncoder()
#intializing the labelencoder for enocoding the categorical variables into numericals

x[:,1]= lx_1.fit_transform(x[:,1])
x[:,2]= lx_1.fit_transform(x[:,2])
#fitting the column 1 (geography) and column 2 (Gender) for encoding



#onehotenocder to remove the dummy variables  as column 1 has 3 categorical variables
ohe = OneHotEncoder(categorical_features=[1])
x = ohe.fit_transform(x).toarray()
x = x[:,1:]


#importing the train test split module for splitting the train and test values

from sklearn.model_selection import train_test_split
x_t ,x_test,y_t,y_test = train_test_split(x,y,test_size = 0.3 , random_state = 0)
#splitting the dataset

#we have to transform all the values within a standard scale for avoiding large dimensional differences in the values 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_t = sc.fit_transform(x_t)
x_test = sc.transform(x_test)



import keras
from keras.models import Sequential
from keras.layers import Dense


#creating the neuralNet
#ourfinal problem is to get an output of 0 or 1 hence we are creating a neural classifier


NeuralClf = Sequential()


#creating the neural layers

NeuralClf.add(Dense(6,kernel_initializer='uniform',activation='relu',input_shape=(11,)))

NeuralClf.add(Dense(6,kernel_initializer='uniform',activation='relu'))

NeuralClf.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

#compiling the layer with stochastic gradient and accuracy metrics

NeuralClf.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[16]:


NeuralClf.fit(x_t,y_t,batch_size=10,epochs=30)


#running a batch size of 10 

y_pred = NeuralClf.predict(x_test)


# predicting the with Test set
y_pred = (y_pred>0.5)

#the prediction returns the probability, we are taking the probability above 0.5 to see wether a customer will stay in the bank or not
#generating the confusion matrix

from sklearn.metrics import confusion_matrix
c = confusion_matrix(y_test,y_pred)
tp = int(c[0][0])
tn = int(c[1][1])



#printing the accuracy 

print("Accuracy : ",((tp+tn)/3000))






