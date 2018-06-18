# coding=utf-8
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy import sparse
import gc
import warnings
warnings.filterwarnings("ignore")

submit = True
if submit == True:
    train = pd.read_csv('./data/on_train.csv')  # 全部数据
    print('train length', len(train))
    test = pd.read_csv('./data/on_test.csv')
    print('test length', len(test))
    train_x = sparse.load_npz('./data/on_train_x.npz')
    test_x = sparse.load_npz('./data/on_test_x.npz')
else:
    train = pd.read_csv('./data/off_train.csv')  # 分割的数据
    print('train length', len(train))
    test = pd.read_csv('./data/off_test.csv')
    print('test length', len(test))
    train_x = sparse.load_npz('./data/off_train_x.npz')
    test_x = sparse.load_npz('./data/off_test_x.npz')
data = pd.concat([train, test])


# dFeat构造 userFeat的均值、比率特征
def userFeat(data, column):
    for col in column:
        print('构造 '+ col +' 均值、比率特征')
        # 广告的平均、比率年龄特征
        grouped = data.groupby([col])['age'].mean().reset_index()
        grouped.columns = [col, col+'_mean_age']
        data = data.merge(grouped, how='left', on=col)
        data[col+'_age_ratio'] = data[col+'_mean_age']/data[col]
        # 广告的平均、比率学历特征
        grouped = data.groupby([col])['education'].mean().reset_index()
        grouped.columns = [col, col+'_mean_education']
        data = data.merge(grouped, how='left', on=col)
        data[col+'_education_ratio'] = data[col+'_mean_education']/data[col]
        # 广告的平均、比率地理特征
        grouped = data.groupby([col])['LBS'].mean().reset_index()
        grouped.columns = [col, col+'_mean_LBS']
        data = data.merge(grouped, how='left', on=col)
        data[col+'_LBS_ratio'] = data[col+'_mean_LBS']/data[col]
        del grouped
    gc.collect()
    return data


# userFeat构造 用户的总数、比率特征 相同水平的特征
def adFeat(data, column):
    for col in column:
        print('构造 '+ col +' 总数、比率特征')
        # 用户点击相同的广告数量和比率
        grouped = data.groupby([col])['uid'].size().reset_index()
        grouped.columns = [col, col+'_size_uid']
        data = data.merge(grouped, how='left', on=col)
        data[col+'_uid_ratio'] = data[col+'_size_uid']/data[col]
        # 具有相同教育水平的人
        grouped = data.groupby([col, 'education']).size().reset_index()
        grouped.columns = [col, 'education', col+'_education']
        data = data.merge(grouped, how='left', on=[col, 'education'])
        data[col+'_education_ratio'] = data[col+'_education']/data[col]
        # 具有相同地理位置的人
        grouped = data.groupby([col, 'LBS']).size().reset_index()
        grouped.columns = [col, 'LBS', col+'_LBS']
        data = data.merge(grouped, how='left', on=[col, 'LBS'])
        data[col+'_LBS_ratio'] = data[col+'_LBS']/data[col]
        # 具有相同兴趣1,2,5的人
        grouped = data.groupby([col, 'interest1']).size().reset_index()
        grouped.columns = [col, 'interest1', col+'_interest1']
        data = data.merge(grouped, how='left', on=[col, 'interest1'])
        data[col+'_interest1_ratio'] = data[col+'_interest1']/data[col]

        grouped = data.groupby([col, 'interest2']).size().reset_index()
        grouped.columns = [col, 'interest2', col+'_interest2']
        data = data.merge(grouped, how='left', on=[col, 'interest2'])
        data[col+'_interest2_ratio'] = data[col+'_interest2']/data[col]

        grouped = data.groupby([col, 'interest5']).size().reset_index()
        grouped.columns = [col, 'interest5', col+'_interest5']
        data = data.merge(grouped, how='left', on=[col, 'interest5'])
        data[col+'_interest5_ratio'] = data[col+'_interest5']/data[col]

        # 具有相同kw1的人
        grouped = data.groupby([col, 'kw1']).size().reset_index()
        grouped.columns = [col, 'kw1', col+'kw1']
        data = data.merge(grouped, how='left', on=[col, 'kw1'])
        data[col+'_'+'kw1'+'_ratio'] = data[col+'kw1']/data[col]
        # 具有相同kw2的人
        grouped = data.groupby([col, 'kw2']).size().reset_index()
        grouped.columns = [col, 'kw2', col+'kw2']
        data = data.merge(grouped, how='left', on=[col, 'kw2'])
        data[col+'_'+'kw2'+'_ratio'] = data[col+'kw2']/data[col]
        # 具有相同topic1的人
        grouped = data.groupby([col, 'topic1']).size().reset_index()
        grouped.columns = [col, 'topic1', col+'topic1']
        data = data.merge(grouped, how='left', on=[col, 'topic1'])
        data[col+'_'+'topic1'+'_ratio'] = data[col+'topic1']/data[col]
        # 具有相同topic2的人
        grouped = data.groupby([col, 'topic2']).size().reset_index()
        grouped.columns = [col, 'topic2', col+'topic2']
        data = data.merge(grouped, how='left', on=[col, 'topic2'])
        data[col+'_'+'topic2'+'_ratio'] = data[col+'topic2']/data[col]
        # 具有相同ct的人
        grouped = data.groupby([col, 'ct']).size().reset_index()
        grouped.columns = [col, 'ct', col+'_ct']
        data = data.merge(grouped, how='left', on=[col, 'ct'])
        data[col+'_ct_ratio'] = data[col+'_ct']/data[col]
        # 具有相同运营商的人
        grouped = data.groupby([col, 'carrier']).size().reset_index()
        grouped.columns = [col, 'carrier', col+'_carrier']
        data = data.merge(grouped, how='left', on=[col, 'carrier'])
        data[col+'_carrier_ratio'] = data[col+'_carrier']/data[col]
        del grouped
    gc.collect()
    return data


# 用户喜好构造特征
def user_like(data, columns):
    print('user like feat...')
    for col in columns:
        print('构造 '+ col +' 用户喜好特征')
        grouped = data.groupby(['uid', col]).size().reset_index()
        grouped.columns = ['uid', col, 'uid_'+col]
        data = data.merge(grouped, how='left', on=['uid', col])
        data['uid_'+col+'_radio'] = data['uid_'+col]/data[col]
        del grouped
    gc.collect()
    return data


# 活跃的特征
def hot_feat(data):
    add = pd.DataFrame(data.groupby(['aid']).uid.nunique()).reset_index()
    add.columns = ['aid', 'aid_uid_nun']
    data = pd.merge(data, add, 'left', on=['aid'])
    del add
    gc.collect()
    return data


# 预处理
def pre_feat(data):
    print('process...')
    for feat in data.columns:
        if data[feat].dtype == object:
            data.loc[data[feat] == '-1', feat] = np.nan
        else:
            data.loc[data[feat] == -1, feat] = np.nan
    # 缺失处理
    data['app_nan'] = pd.isnull(data[['appIdAction', 'appIdInstall']]).sum(axis=1)
    data['int_nan'] = pd.isnull(data[['interest1', 'interest2', 'interest3', 'interest4', 'interest5']]).sum(axis=1)
    data['kw_nan'] = pd.isnull(data[['kw1', 'kw2', 'kw3']]).sum(axis=1)
    data['top_nan'] = pd.isnull(data[['topic1', 'topic2', 'topic3']]).sum(axis=1)

    # 兴趣长度和安装长度
    def feat_len(x):
        if pd.isnull(x):
            return 0
        else:
            return len(x.split(' '))
    data['appIdA_len'] = data['appIdAction'].apply(feat_len)
    data['appIdI_len'] = data['appIdInstall'].apply(feat_len)
    data['interest1_len'] = data['interest1'].map(feat_len)
    data['interest2_len'] = data['interest2'].map(feat_len)
    data['interest3_len'] = data['interest3'].map(feat_len)
    data['interest4_len'] = data['interest4'].map(feat_len)
    data['kw1_len'] = data['kw1'].map(feat_len)
    data['kw2_len'] = data['kw2'].map(feat_len)
    data['kw3_len'] = data['kw3'].map(feat_len)
    data['topic1_len'] = data['topic1'].map(feat_len)
    data['topic2_len'] = data['topic2'].map(feat_len)
    data['topic3_len'] = data['topic3'].map(feat_len)
    data['ct_len'] = data['ct'].map(feat_len)
    return data


def radio_feat(data):
    data['age_4_edu5,7or_age_5_edu_4,6,7'] = 0
    list_ = list(
        data[((data['age'] == 4) & (data['education'] == 5)) | ((data['age'] == 4) & (data['education'] == 7)) |
             ((data['age'] == 5) & (data['education'] == 4)) | ((data['age'] == 5) & (data['education'] == 5)) |
             ((data['age'] == 5) & (data['education'] == 7))
             ].index)
    data.loc[list_, 'age_4_edu5,7or_age_5_edu_4,6,7'] = 1

    # low ratio
    data['edu_gender_52_72_42_51'] = 0
    list_ = list(
        data[((data['education'] == 5) & (data['gender'] == 2)) | ((data['education'] == 7) & (data['gender'] == 2)) |
             ((data['education'] == 4) & (data['gender'] == 2)) | ((data['education'] == 5) & (data['gender'] == 1))
             ].index)
    data.loc[list_, 'edu_gender_52_72_42_51'] = 1

    # high ratio
    data['age_gender_20_21_22'] = 0
    list_ = list(data[((data['age'] == 2) & (data['gender'] == 0)) | ((data['age'] == 2) & (data['gender'] == 1)) |
                      ((data['age'] == 2) & (data['gender'] == 2))
                      ].index)
    data.loc[list_, 'age_gender_20_21_22'] = 1
    return data


def train_set_encode(train, train_x):
    drop_feat = ['creativeSize', 'LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                 'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                 'adCategoryId', 'productId', 'productType', 'appIdAction', 'appIdInstall', 'interest1', 'interest2',
                 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    train_w = train.drop(drop_feat, axis=1)
    train_x = sparse.hstack((train_w, train_x))
    return train_x


def train_set(train):
    drop_feat = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1',
                 'kw2', 'kw3', 'topic1', 'topic2', 'topic3', 'ct', 'marriageStatus', 'os']
    train_w = train.drop(drop_feat, axis=1)
    return train_w


def test_set_encode(test, test_x):
    drop_feat = ['creativeSize', 'LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                 'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                 'adCategoryId', 'productId', 'productType', 'appIdAction', 'appIdInstall', 'interest1', 'interest2',
                 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    test_w = test.drop(drop_feat, axis=1)
    test_x = sparse.hstack((test_w, test_x))
    return test_x


def test_set(test):
    drop_feat = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1',
                 'kw2', 'kw3', 'topic1', 'topic2', 'topic3', 'ct', 'marriageStatus', 'os']
    test_w = test.drop(drop_feat, axis=1)
    return test_w


def LGB_test(train_x, train_y, test_x, test_y):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], eval_metric='auc', early_stopping_rounds=100)
    # # 查看属性重要性
    # df = pd.DataFrame(columns=['feature', 'important'], index=None)
    # df['feature'] = train_x.columns
    # df['important'] = clf.feature_importances_
    # df = df.sort_values(axis=0, ascending=True, by='important').reset_index()
    # print df
    print(clf.best_score_['valid_1']['auc'])
    return clf


def LGB_predict(train_x, train_y, test_x, res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=64, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=10000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.04, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc', early_stopping_rounds=100)
    res['score'] = clf.predict_proba(test_x)[:, 1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('./data/submission.csv', index=False)
    return clf

data = pre_feat(data)
col = ['aid', 'advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId']
data = userFeat(data, col)
data = adFeat(data, col)
# 用户喜好特征
col = ['productType', 'adCategoryId', 'creativeId']
data = user_like(data, col)
# 活跃的特征
data = hot_feat(data)
# 添加radio特征
data = radio_feat(data)

sp_len = len(train)
train = data.iloc[: sp_len, :]
train_y = train.pop('label')
test = data.iloc[sp_len:, ]
test_y = test['label']
test = test.drop(['label'], axis=1)

# 使用encode编码
train_x = train_set_encode(train, train_x)
test_x = test_set_encode(test, test_x)
# 不适用encode编码
# train_x = train_set(train)
# test_x = test_set(test)
res = test[['aid', 'uid']]
# LGB_test(train_x, train_y, test_x, test_y)
model = LGB_predict(train_x, train_y, test_x, res)

# 0.734032
