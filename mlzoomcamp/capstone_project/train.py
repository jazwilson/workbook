#!/usr/bin/env python
# coding: utf-8

# In[265]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pickle
 

df = pd.read_csv('data.csv')
df = df.drop(columns=['index', 'district'])
df = df.fillna(0)
df = pd.get_dummies(df, columns=["state"])
df = pd.get_dummies(df, columns=["type"])
df_factors = list(df.dtypes[df.dtypes == 'uint8'].index)


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val       = train_test_split(df_full_train, test_size=0.25, random_state=42)

len(df_train), len(df_val), len(df_test)

df_test  = df_test.reset_index(drop=True)

y_full_train = df_full_train.mmr.values
y_test       = df_test.mmr.values

del df_full_train['mmr']
del df_train['mmr']
del df_val['mmr']
del df_test['mmr']

dv = DictVectorizer(sparse=False)

train_dict = df_full_train[df_factors].to_dict(orient='records')
X_full_train    = dv.fit_transform(train_dict)

test_dict = df_test[df_factors].to_dict(orient='records')
X_test    = dv.fit_transform(test_dict)


output_file='model.dv'
with open(output_file, 'wb') as f_out:
    pickle.dump(dv, f_out)

def train_c_NN(X_train,y_train,X_val,y_val,layer_size=16,learning_rate=0.1, droprate=0.5):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.asarray(X_train))
    
    c_model = tf.keras.Sequential([normalizer,layers.Dense(layer_size,activation='relu'),layers.Dense(units=1), layers.Dropout(droprate)])
    c_model.summary()
    
    c_model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate)
                    ,loss='mean_squared_error')
    c_history = c_model.fit(X_train,y_train,epochs=100,validation_data=(np.asarray(X_val),y_val))
    return c_history


lr   = 0.1
size = 16
dr   = 0

best = train_c_NN(X_full_train,y_full_train,X_test,y_test, layer_size=size,learning_rate=lr, droprate = dr)

#best.model.save('keras_model')
tf.saved_model.save(best.model, 'view-model')

