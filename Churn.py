# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:45:41 2019

@author: DELL
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import History,EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import keras as k
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,RobustScaler 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
from keras import losses

from IPython.display import Image
from keras.layers.core import Activation
from keras.layers.core import Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

warnings.filterwarnings('ignore')



#dataset is imported 
dataset = pd.read_csv('Churn_Modelling.csv')


for column in dataset.columns:
        if dataset[column].dtype == np.number:
            continue
        dataset[column] = LabelEncoder().fit_transform(dataset[column])

#X  that independent variable is taken from the 3rd column until the end 
#Y variable(prediction) is taken last column
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values



#Data set is divided into testing and training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

#Callback is defined.
mycallbacks = EarlyStopping(monitor='loss', patience=2,verbose=1,restore_best_weights=True)

#Scaling of features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#3 artificial neural networks:input(1),hide(1),exit(1)
model = Sequential()

#input layer
model.add(Dense(256,activation="relu", input_dim=10, kernel_initializer="uniform"))

#hide layer
model.add(Dense(128,activation="relu", kernel_initializer="uniform"))

#exit layer
model.add(Dense(1,activation="sigmoid",  kernel_initializer="uniform"))


#Neural network model is run
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

#Neural network applies training dataset
history=model.fit(X_train, y_train,validation_split=0.2, batch_size=X_train.shape[0], epochs=200, callbacks=[mycallbacks], verbose=1)

#Result are predicted 
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Confusion Matrix is created
cm = confusion_matrix(y_test, y_pred)
print(cm)


#Accuracy Function of Model is done visualization
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig("acc.png")

#Lost Function of Model is done visualization
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig("loss.png")
