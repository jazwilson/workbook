#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

print('Reading Data')
df_original = pd.read_csv('data.csv')

df = df_original[[
           'max_players',
           'max_playtime',
           'min_age',
           'min_players',
           'min_playtime',
           'playing_time',
           'year_published',
           'category',
           'mechanic',
           'average_rating',
           'users_rated']]

categorical_columns   = [
           'name',           
           'category',
           'mechanic']


df = df.fillna(method="ffill")

# Get a list of the categories
category   = sorted(set(",".join(df.category).split(",")))

## Convert to binary readout: 0 = not category, 1 = category 
for c in category:
    df[c] = df.category.str.contains(c).astype(int)


## Mechanic variable eg. 'Campaign', 'Player Elimination', 'Commodity Speculation'

mechanic   = sorted(set(",".join(df.mechanic).split(",")))
for m in mechanic:
    df[m]  = df.mechanic.str.contains(m).astype(int)

## Define variable columns 
game_col   = list(df.dtypes[df.dtypes == 'int64'].index)

# Define full_train and test dataset

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_test                = df_test.reset_index(drop=True)

y_full_train = df_full_train.average_rating.values
y_test       = df_test.average_rating.values

del df_full_train['average_rating']
del df_test['average_rating']

dv = DictVectorizer(sparse=False)

full_train_dict = df_full_train[game_col].to_dict(orient='records')
X_full_train    = dv.fit_transform(full_train_dict)

test_dict       = df_test[game_col].to_dict(orient='records')
X_test          = dv.transform(test_dict)

# Random Forest Regressor model 

rf = RandomForestRegressor(n_estimators=125,
                            max_depth=15,
                            min_samples_leaf=1,
                            random_state=1)
rf.fit(X_full_train, y_full_train)

y_pred = rf.predict(X_test)

# Save the model

output_file='model.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)

print(f'the model is saved to {output_file}')