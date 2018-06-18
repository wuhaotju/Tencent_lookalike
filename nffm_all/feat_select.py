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

path = './data/'

ev_cv_cvr = pd.read_csv(path+'evals_x_CV_cvr_select.csv')
li = ev_cv_cvr.columns.tolist()
evcc = li[0:10]
ev_cv_cvr = ev_cv_cvr[evcc]

ev_cvr = pd.read_csv(path+'evals_x_cvr_select.csv')
li = ev_cvr.columns.tolist()
evc = li[0:10]
ev_cvr = ev_cvr[evc]

ev_ratio = pd.read_csv(path+'evals_x_ratio_select.csv')
li = ev_ratio.columns.tolist()
evr = li[0:10]
ev_ratio = ev_ratio[evr]

print(ev_cv_cvr.columns)
print(ev_cvr.columns)
print(ev_ratio.columns)

ev_cv_cvr.to_csv(path+'evals_x_CV_cvr_select.csv', index=False)
ev_cvr.to_csv(path+'evals_x_cvr_select.csv', index=False)
ev_ratio.to_csv(path+'evals_x_ratio_select.csv', index=False)
