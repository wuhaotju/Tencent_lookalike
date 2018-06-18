import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import numpy as np
import time
import random
import gc
import warnings
warnings.filterwarnings("ignore")

if os.path.exists('./data/train_test_merge.csv'):
    print('Reading...')
    data = pd.read_csv('./data/train_test_merge.csv')
else:
    ##读取数据
    # 注意修改路径
    print("Read & Merge")
    ad_feature=pd.read_csv('../datasets/adFeature.csv')
    if os.path.exists('../datasets/userFeature.csv'):
        user_feature=pd.read_csv('../datasets/userFeature.csv')
        print('User feature prepared')
    else:
        userFeature_data = []
        with open('../datasets/userFeature.data', 'r') as f:
            for i, line in enumerate(f):
                line = line.strip().split('|')
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userFeature_data.append(userFeature_dict)
                if i % 1000000 == 0:
                    print(i)
            user_feature = pd.DataFrame(userFeature_data)
            print('User feature...')
            user_feature.to_csv('../datasets/userFeature.csv', index=False)
            print('User feature prepared')
            
    ##插入test字段，0代表训练集，1代表测试集1，2代表测试集2
    train_pre=pd.read_csv('../datasets/train.csv')
    
    predict1=pd.read_csv('../datasets/test1.csv')
    predict1['test'] = 1
    predict2=pd.read_csv('../datasets/test2.csv')
    predict2['test'] = 2
    predict = pd.concat([predict1,predict2],axis=0,ignore_index=True)
    
    train_pre.loc[train_pre['label']==-1,'label']=0
    predict['label']=-1
    data=pd.concat([train_pre,predict])
    
    data['test'].fillna(value=0,inplace=True)
    
    ##关联数据
    print("Merge...")
    data=pd.merge(data,ad_feature,on='aid',how='left')
    data=pd.merge(data,user_feature,on='uid',how='left')
    user_feature = []
    ad_feature = []
    train_pre = []
    predict = []
    data=data.fillna('-1')
    data = pd.DataFrame(data.values,columns=data.columns)
    data['label'] = data['label'].astype(float)

    #保存文件
    print('Saving merge file...')
    data.to_csv('./data/train_test_merge.csv',index=False)
    print('Over')

##插入字段n_parts数据集进行分块，训练集分成五块1、2、3、4、5，测试集1为6、测试集2为7
##也就是test字段与n_parts字段都是为了区分数据块，n_parts对训练集进行了分块
print('N parts...')
train = data[data['test']==0][['aid','label']]
test1_index  = data[data['test']==1].index
test2_index  = data[data['test']==2].index
n_parts = 5
index = []
for i in range(n_parts):
    index.append([])
aid = list(train['aid'].drop_duplicates().values)
for adid in aid:
    dt = train[train['aid']==adid]
    for k in range(2):
        lis = list(dt[dt['label']==k].sample(frac=1,random_state=2018).index)
        cut = [0]
        for i in range(n_parts):
            cut.append(int((i+1)*len(lis)/n_parts)+1)
        for j in range(n_parts):
            index[j].extend(lis[cut[j]:cut[j+1]])
se = pd.Series()
for r in range(n_parts):
    se = se.append(pd.Series(r+1,index=index[r]))
se = se.append(pd.Series(6,index=test1_index)) 
se = se.append(pd.Series(7,index=test2_index)) 
data.insert(0,'n_parts',list(pd.Series(data.index).map(se).values))
del train
del data['test']
gc.collect()

##根据生成的分块字段n_parts划分训练与验证集(n_parts=1)
print('Index...')
train_part_index = list(data[(data['label']!=-1)&(data['n_parts']!=1)].index)
evals_index = list(data[(data['label']!=-1)&(data['n_parts']==1)].index)
#test1_index  = list(data[data['n_parts']==6].index)
test2_index  = list(data[data['n_parts']==7].index)

del data['n_parts']

print('添加新的统计特征')
# 频数很少的种类，划为其他
def del_little_feature(data,feature):
    data1 = data[feature].value_counts().reset_index().rename(columns = {'index':feature,feature:'count'})
    data2 = data1[data1['count']<5]
    del_kind = data2[feature].values.tolist()
    for i in range(len(del_kind)):
        data.loc[data[feature]==del_kind[i],feature]=-2
    return data
data = del_little_feature(data, 'LBS')
print('LBS is prepared!')

print('添加新的交叉特征')
data['aid_age']=((data['aid']*100)+(data['age']))
data['aid_gender']=((data['aid']*100)+(data['gender']))
data['aid_LBS']=((data['aid']*100)+(data['LBS'])) 

train = data.loc[train_part_index]
evals = data.loc[evals_index]
#test1 = data.loc[test1_index]
test2 = data.loc[test2_index]

train.to_csv('./data/train.csv', index=False)
evals.to_csv('./data/evals.csv', index=False)
#test1.to_csv('./data/test1.csv', index=False)
test2.to_csv('./data/test2.csv', index=False)
