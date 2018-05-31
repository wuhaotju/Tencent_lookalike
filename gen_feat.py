# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
'''联合数据集
print('-------------------- read data  -------------------------------------')
# merge之后的data
ad_feature = pd.read_csv('./data/oria/adFeature.csv')
user_feature = pd.read_csv('./data/oria/userFeature_once.csv')
train = pd.read_csv('./data/oria/train.csv')    #45539700
print("train length", len(train))
test1 = pd.read_csv('./data/oria/test1.csv')    #11729073
print("test1 length", len(test1))
test2 = pd.read_csv('./data/oria/test2.csv')    #11727304
print("test2 length", len(test2))

train.loc[train['label'] == -1, 'label'] = 0
test1['label'] = -1
test2['label'] = -1
data = pd.concat([train, test1])
data = pd.concat([data, test2])
data = pd.merge(data, ad_feature, on='aid', how='left')
data = pd.merge(data, user_feature, on='uid', how='left')
data = data.fillna('-1')
data.to_csv('./data/all_data.csv', index=False)
print('-------------------- train and test data ----------------------------')
train = data[data.label != -1]
print('train length', len(train))
test = data[data.label == -1]
print('test length', len(test))
print('end......')
'''



'''分割数据集，线上线下
data = pd.read_csv('./data/all_data.csv')
train = data[data.label != -1]
print('train length', len(train))
test = data[data.label == -1]
print('test length', len(test))

on_train = train
on_test = test
on_train.to_csv('./data/on_train.csv', index=False) #45539700
on_test.to_csv('./data/on_test.csv', index=False)   #23456377
print('on is over!')
train_y = train.pop('label')
train_x, test_x, train_y, test_y = train_test_split(train, train_y, test_size=0.2, random_state=2018)
off_train = pd.concat([train_x, train_y], axis=1)
off_test = pd.concat([test_x, test_y], axis=1)
off_train.to_csv('./data/off_train.csv', index=False)
print("off_train length", len(off_train))
off_test.to_csv('./data/off_test.csv', index=False)
print("off_test length", len(off_test))
print('off is over!')
'''

'''线下编码
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse

train = pd.read_csv('./data/off_train.csv') #36431760
len_train = len(train)
print("off_train length", len(train))
test = pd.read_csv('./data/off_test.csv')   #9107940
print("off_test length", len(test))
len_test = len(test)
data = pd.concat([train, test])
train = data.iloc[: len_train, :]
print("len train", len(train))
one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct', 'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                   'adCategoryId', 'productId', 'productType']
vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2',
                  'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

train = data.iloc[: len_train, :]
train_y = train.pop('label')
# train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
test = data.iloc[len_train: , :]
test = test.drop('label', axis=1)
enc = OneHotEncoder()
train_x = train[['creativeSize']]
test_x = test[['creativeSize']]

for feature in one_hot_feature:
    print(feature, 'one_hot')
    enc.fit(data[feature].values.reshape(-1, 1))
    train_a = enc.transform(train[feature].values.reshape(-1, 1))
    test_a = enc.transform(test[feature].values.reshape(-1, 1))
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')

cv = CountVectorizer()
for feature in vector_feature:
    print(feature, 'count_ve')
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    test_a = cv.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')

sparse.save_npz("./data/off_train_x.npz", train_x)
sparse.save_npz("./data/off_test_x.npz", test_x)
'''

'''线上编码'''
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse

train = pd.read_csv('./data/on_train.csv')  #45539700
print("train length", len(train)) 
test = pd.read_csv('./data/on_test.csv')    #23456377
print("test length", len(test)) 
data = pd.concat([train, test])

one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct', 'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                   'adCategoryId', 'productId', 'productType']
vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2',
                  'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

train = data[data.label != -1]
print("train length", len(train))
train_y = train.pop('label')
# train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
test = data[data.label == -1]
print("test length", len(test))
test = test.drop('label', axis=1)
enc = OneHotEncoder()
train_x = train[['creativeSize']]
test_x = test[['creativeSize']]

for feature in one_hot_feature:
    print(feature, 'one_hot')
    enc.fit(data[feature].values.reshape(-1, 1))
    train_a = enc.transform(train[feature].values.reshape(-1, 1))
    test_a = enc.transform(test[feature].values.reshape(-1, 1))
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')

cv = CountVectorizer()
for feature in vector_feature:
    print(feature, 'count_ve') 
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    test_a = cv.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')

sparse.save_npz("./data/on_train_x.npz", train_x)
sparse.save_npz("./data/on_test_x.npz", test_x)



