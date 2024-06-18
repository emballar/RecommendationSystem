#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 23:09:36 2024

@author: erinballar
"""

import pandas as pd

#Load Data
business_ratings = pd.read_pickle("biz_reviews.pkl")
users = pd.read_pickle("users.pkl")

#### DATA PRE-PROCESSING ####

#drop unecessary columns from users df
users = users.drop(columns = ['compliment_hot',
'compliment_more', 'compliment_profile',
'compliment_cute', 'compliment_list',
'compliment_note', 'compliment_plain',
'compliment_cool', 'compliment_funny',
'compliment_writer', 'compliment_photos','useful','funny','cool','elite','fans'])

#only restaurants open/in business
business_ratings = business_ratings[business_ratings['is_open'] == 1]

#restaurants only in CA
business_ratings = business_ratings[business_ratings['state'] =='CA']

#drop unecessary columns from business ratings df
business_ratings = business_ratings.drop(columns= ['latitude','longitude','address','postal_code','is_open'])

#rename ratings cols
business_ratings.rename(columns={'stars_x': 'business_avg_rating','stars_y': 'user_rating'}, inplace=True)

#merge business ratings and users df
df = pd.merge(business_ratings, users, how = 'inner', on= 'user_id')

#drop na
df.dropna()

#rename merged cols
df.rename(columns = {'name_x':'business_name', 'review_count_x':'business_review_count', 'name_y':'user_name',
                     'review_count_y':'user_review_count','average_stars':'average_user_rating'},
          inplace = True)


#drop business with no reviews
df = df[df['business_review_count'] != 0]

#only business greater than or equal 4 stars
df = df[df['business_avg_rating'] >= 4]

#drop users that have not made any reviews
df = df[df['user_review_count'] != 0]


#### MACHINE LEARNING MODEL TESTING ####

from surprise import Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import Dataset, accuracy
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import CoClustering
import random
from statistics import mode

reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[['user_id', 'business_id', 'user_rating']], reader)

#set similarity calculation
sim_options = {'name': 'cosine', 'user_based': True}

## HEADS UP: Running crossvalidation for KNN takes a very long time.
# It is recommended to run one cross validation at a time.
"""
#KNN Basic
algo = KNNBasic(sim_options = sim_options)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

#KNN WithMeans
algo = KNNWithMeans(sim_options = sim_options)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

#KNN With ZScore
algo = KNNWithZScore(sim_options = sim_options)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

#KNN Baseline
algo = KNNBaseline(sim_options = sim_options)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
"""

#CoClustering
algo = CoClustering()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)



#### FINAL MODEL ####

train, test = train_test_split(data, test_size=0.25)

algo = KNNBaseline(sim_options = sim_options)

algo.fit(train)
predictions = algo.test(test)

print(accuracy.rmse(predictions))
print(accuracy.mae(predictions))

impossible = sum(1 for prediction in predictions if prediction.details['was_impossible'] == True)
possible = [prediction for prediction in predictions if prediction.details['was_impossible'] == False]

print(f'Impossible to predict: {impossible} out of {len(predictions)}')

lastdf = pd.DataFrame(possible)
lastdf = lastdf.rename(columns= {'uid':'user_id', 'iid':'business_id', 'r_ui': 'actual_rating', 'est':'model_est'})
lastdf = lastdf.drop(columns = 'details')


#make new df with unique businesses & their names
biz_names = business_ratings[['business_id', 'name']]
biz_names = biz_names.drop_duplicates()

#merge with lastdf on business_id
recs = pd.merge(lastdf, biz_names, how = 'inner', on = 'business_id')

#drop business id
recs = recs.drop(columns = 'business_id')
#rename
recs.rename(columns = {'name':'business_name'}, inplace = True)

#rounded rating estimate
recs['rounded_model_est'] = round(recs['model_est'])

#rearrange cols for an order that makes sense
rec = recs.loc[:,['user_id','business_name','model_est','rounded_model_est','actual_rating']]

#show example of predictions
print('Sample of Model Ratings vs Actual')
print(recs.head())

#### SAMPLE OF HOW RECOMMENDATION SYSTEM WOULD SHOW ON WEBSITE ####

#find user with most predicitons
user = mode(rec['user_id'])

sample = rec[rec['user_id'] == user]

#only recommend restaurants with model estimate of 4 or higher
sample = sample[sample['rounded_model_est'] >= 4]

print('Sample of Recommendation System on Website')
print(f'Hi {user} ! Here are some food spots we think you would like:')
print(sample['business_name'])

