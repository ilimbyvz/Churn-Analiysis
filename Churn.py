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



#Veriseti import edilir
dataset = pd.read_csv('Churn_Modelling.csv')


for column in dataset.columns:
        if dataset[column].dtype == np.number:
            continue
        dataset[column] = LabelEncoder().fit_transform(dataset[column])

#Bağımsız değişken olan X:3.kolondan itibaren sona kadar alınır
#Y değişkeni(tahmin) son kolon alınır.
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values



#Veriseti eğitim ve test olarak  bölünür
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

#callback tanımlanır.
mycallbacks = EarlyStopping(monitor='loss', patience=2,verbose=1,restore_best_weights=True)

#Özelliklerin Ölçeklenmesi(Scaling) yapılır
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#3 tane yapay sinir ağı oluşturulur:giriş(1),gizli(1),çıkış(1)
model = Sequential()

# giriş katmanı
model.add(Dense(256,activation="relu", input_dim=10, kernel_initializer="uniform"))

#gizli katman
model.add(Dense(128,activation="relu", kernel_initializer="uniform"))

#çkış katmanı
model.add(Dense(1,activation="sigmoid",  kernel_initializer="uniform"))


#sinir ağı modeli çalıştırılır
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

#Ağ modeli eğitim setine uygulanır. 
history=model.fit(X_train, y_train,validation_split=0.2, batch_size=X_train.shape[0], epochs=200, callbacks=[mycallbacks], verbose=1)

#sonuçlar tahmin edilir.
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Confusion Matrix oluşturulur
cm = confusion_matrix(y_test, y_pred)
print(cm)


#Modelin doğruluk fonksiyonu görselleştirmesi yapılır
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig("acc.png")

#Modelin kayıp fonksiyonu göselleştirmesi yapılır
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show(),
plt.savefig("loss.png")
