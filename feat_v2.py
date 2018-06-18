# -*- coding: utf-8 -*-
import pandas as pd
import lightgbm as lgb
from scipy import sparse
import gc
import warnings
warnings.filterwarnings("ignore")

submit = True
if submit == True:
    train = pd.read_csv('./data/on_train.csv')  # 45539700
    print('train length', len(train))
    test = pd.read_csv('./data/on_test.csv')    # 23456377
    print('test length', len(test)) 
    train_x = sparse.load_npz('./data/on_train_x.npz')
    test_x = sparse.load_npz('./data/on_test_x.npz')
    print("data over!")
else:
    train = pd.read_csv('./data/off_train.csv')  #
    print('train length', len(train))
    test = pd.read_csv('./data/off_test.csv')
    print('test length', len(test)) 
    train_x = sparse.load_npz('./data/off_train_x.npz')
    test_x = sparse.load_npz('./data/off_test_x.npz')
    print("data over!")
data = pd.concat([train, test])


def train_set(train, train_x):
    drop_feat = ['creativeSize', 'LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                 'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                 'adCategoryId', 'productId', 'productType', 'appIdAction', 'appIdInstall', 'interest1', 'interest2',
                 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    train_w = train.drop(drop_feat, axis=1)
    train_x = sparse.hstack((train_w, train_x))
    return train_x


def test_set(test, test_x):
    drop_feat = ['creativeSize', 'LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                 'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                 'adCategoryId', 'productId', 'productType', 'appIdAction', 'appIdInstall', 'interest1', 'interest2',
                 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    test_w = test.drop(drop_feat, axis=1)
    test_x = sparse.hstack((test_w, test_x))
    return test_x


def LGB_test(train_x, train_y, test_x, test_y):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=64, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], eval_metric='auc', early_stopping_rounds=100)
    # 查看属性重要性
    df = pd.DataFrame(columns=['feature', 'important'], index=None)
    df['feature'] = train_x.columns
    df['important'] = clf.feature_importances_
    df = df.sort_values(axis=0, ascending=True, by='important').reset_index()
    print(df)
    print("best score", clf.best_score_['valid_1']['auc']) 
    return clf


def LGB_predict(train_x, train_y, test_x,res):
    print("LGB submit")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=64, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc', early_stopping_rounds=100)
    res['score'] = clf.predict_proba(test_x)[:, 1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('./data/submission.csv', index=False)
    return clf
sp_len = len(train)
train = data.iloc[: sp_len, :]
train_y = train.pop('label')
test = data.iloc[sp_len:, ]
test = test.drop(['label'], axis=1)

train_x = train_set(train, train_x)
test_x = test_set(test, test_x)
res = test[['aid', 'uid']]
# LGB_test(train_x, train_y, test_x, test_y)
model = LGB_predict(train_x, train_y, test_x, res)

# 0.734032
