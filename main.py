import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 1
EPOCHS = 50
BATCH_SIZE = 32

# Load data
df = pd.read_csv('heart.csv')
X = df.drop('target',axis=1)
y = df.iloc[:,-1]

# Preprocess data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=SEED)
sd = StandardScaler()
sd.fit(X_train)
X_train = sd.transform(X_train)
X_test = sd.transform(X_test)

#Create model
nIn = X_test.shape[1]
nClass = len(np.unique(y_test))
l1 = keras.regularizers.L1(l1=0.001)

model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(nIn,)))
model.add(layers.Dense(256,activation='relu',kernel_regularizer=l1))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(128,activation='relu',kernel_regularizer=l1))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64,activation='relu',kernel_regularizer=l1))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(32,activation='relu',kernel_regularizer=l1))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(nClass,activation='softmax')) # Don't apply regularizer to output layer

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test,y_test),batch_size=BATCH_SIZE,epochs=EPOCHS)

# With 50 epochs and a batch size of 32, we reach a training accuracy of 95.12% and validation accuracy of 96.59%