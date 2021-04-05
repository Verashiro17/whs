# set the matplotlib backend so figures can be saved in the background

import matplotlib
import os

from sklearn.linear_model import (LinearRegression, Ridge,
								  Lasso)

from sklearn.feature_selection import RFE, f_regression

from sklearn.ensemble import RandomForestRegressor
#import readXLS
#from pri_s import readXLS

from minepy import MINE

matplotlib.use("Agg")
# import the necessary packages

import sys

sys.path.append('..')
matplotlib.use("Agg")
from keras import backend
#import the necessary packages

from keras.optimizers import Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import *
import sys
from collections import Counter
from keras.models import load_model
sys.path.append('..')
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.callbacks import EarlyStopping
import re
import xlrd
from lenet import FulCon
EPOCHS = 50
INIT_LR = 1e-3
BS = 10
# RANDOM_SEED=8
feature_num=10
chengfa=0.5
WARNINGYUZHI=0.99

def primaryList(txt_file):
    file=open(txt_file)
    flag_row=0
    pri_th_dict={}
    for line in file.readlines():
        line = line.strip('\n')
        flag_row=flag_row+1
        if line=='':
            flag_row=0
        if flag_row==1:
            now_14=line
            pri_th_dict[now_14]={'total_num':0,
                                 'details':{}}
        elif flag_row>1:
            pri_th_dict[now_14]['details'][re.sub(' */ *',' / ',line)]=0
    file.close()
    # print(pri_th_dict)
    return pri_th_dict

def regularit(df,col_name,MAX,MIN):
    if col_name=='latitude':
        d = abs(df[col_name])
    else:
        d=df[col_name]
    if MAX == '':
        MAX = d.max()
    if MIN=='':
        MIN = d.min()
    df[col_name] = ((d - MIN) / (MAX - MIN)).tolist()
    return df
def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()

    ranks = minmax.fit_transform(order * np.array([list(ranks)]).T).T[0]
    # print(ranks)
    ranks[np.isnan(ranks)] = 0
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))
def regularit_pd(df):
    df=regularit(df,'buffer','','')
    df = regularit(df, 'arima_p', '', '')
    df = regularit(df, 'arima_d', '', '')
    df = regularit(df, 'arima_q', '', '')
    df = regularit(df, 'arima_P', '', '')
    df = regularit(df, 'arima_D', '', '')
    df = regularit(df, 'arima_Q', '', '')
    df = regularit(df, 'SES', '', '')
    df = regularit(df, 'MSS', '', '')
    df = regularit(df, 'YSS', '', '')
    df = regularit(df, 'SRA', '', '')
    df = regularit(df, 'SRGA', '', '')
    df = regularit(df, 'MRX', '', '')
    df = regularit(df, 'MTX', '', '')
    df = regularit(df, 'latitude', 90, 0)
    df = regularit(df, 'ave_ntl_a', '', '')
    df = regularit(df, 'ave_ntl_p', '', '')
    df = regularit(df, 'ntl_stdev', '', '')
    df = regularit(df, 'ntl_slope', '', '')
    df = regularit(df, 'warning_soc', '', '')
    df =regularit(df,'month_trend_xielv','','')
    df = regularit(df, 'year_trend_xielv', '', '')
    df = regularit(df, 'year_trend_xielv_dx', '', '')
    df = regularit(df, 'month_raw_xielv', '', '')
    df = regularit(df, 'year_raw_xielv', '', '')
    df = regularit(df, 'month_seasonal_std', '', '')
    df = regularit(df, 'year_seasonal_std', '', '')
    df = regularit(df, 'seasonal_std', '', '')
    df = regularit(df, 'seasonal_ratio_ave', '', '')
    df = regularit(df, 'seasonal_ratio_geoave', '', '')
    df = regularit(df, 'p', '', '')
    df = regularit(df, 'd', '', '')
    df = regularit(df, 'q', '', '')
    df = regularit(df, 'P', '', '')
    df = regularit(df, 'D', '', '')
    df = regularit(df, 'Q', '', '')
    df = regularit(df, 's', '', '')

    df = regularit(df, 'Sviirs_sum_whs', '', '')
    df = regularit(df, 'IR_sum_whs_1', '', '')
    df = regularit(df, 'IR_sum_whs_2', '', '')
    df = regularit(df, 'IR_sum_whs_3', '', '')
    df = regularit(df, 'IR_sum_whs_4', '', '')
    df = regularit(df, 'IR_sum_whs_5', '', '')
    df = regularit(df, 'IR_sum_whs_6', '', '')
    df = regularit(df, 'IR_sum_whs_7', '', '')
    df = regularit(df, 'IR_sum_whs_8', '', '')
    df = regularit(df, 'IR_sum_whs_9', '', '')
    df = regularit(df, 'IR_sum_whs_10', '', '')
    df = regularit(df, 'IR_sum_whs_11', '', '')
    df = regularit(df, 'IR_sum_whs_12', '', '')
    df = regularit(df, 'google_country_var', '', '')
    df=regularit(df,'per_GDP_LRR_max','','')
    df = regularit(df, 'per_GDP_LRR_min', '', '')
    df = regularit(df, 'per_GDP_LRR_ave', '', '')
    # df = regularit(df, 'war_LRR', '', '')
    df = regularit(df, 'per_area_conflict', '', '') #单位面积国家冲突
    # df = regularit(df, 'conflict', '', '')
    # df =regularit(df,'google_country_ave','','')
    # df = regularit(df, 'google_country_min', '', '')


    return df

def danger_year(df):
    label_list=[]
    train_predict_list=[]
    for i in range(len(df)):
        if df.loc[i, 'date_inscribed']<2012:
            if type(df.loc[i, 'danger_list']) != type(np.nan):
                danger_year_split=df.loc[i, 'danger_list'].split(' ')
                danger_year_flag_list=[]
                for danger_year_split_num in range(len(danger_year_split)):
                    if danger_year_split[danger_year_split_num]=='Y':
                        danger_year_min=(int(danger_year_split[danger_year_split_num+1]))
                        danger_year_max=(2019)
                        if danger_year_min<2012:
                            danger_year_flag_list.append([0,1])
                        else:
                            danger_year_flag_list.append([1,1])
                    elif danger_year_split[danger_year_split_num]=='P':
                        danger_year_split0=danger_year_split[danger_year_split_num + 1].split('-')
                        danger_year_min=(int(danger_year_split0[0]))
                        danger_year_max=(int(danger_year_split0[1]))
                        if danger_year_min<2012 and danger_year_max<2012:
                            danger_year_flag_list.append([0,0])
                        elif danger_year_min<2012 and danger_year_max>=2012:
                            danger_year_flag_list.append([0, 1])
                        else:
                            danger_year_flag_list.append([1, 1])
                if [0,1] in danger_year_flag_list:
                    label_list.append(1)
                    train_predict_list.append(0)
                else:
                    if [1,1] in danger_year_flag_list:
                        label_list.append(1)
                        train_predict_list.append(1)
                    else:
                        label_list.append(0)
                        train_predict_list.append(0)
            else:
                label_list.append(0)
                train_predict_list.append(0)
        else:
            train_predict_list.append(1)
            if type(df.loc[i, 'danger_list']) != type(np.nan):
                label_list.append(1)
            else:
                label_list.append(0)

    df.insert(2, 'label', label_list)
    df.insert(2, 'train_predict', train_predict_list)
    return df

def df2array(df):
    data = []
    labels = []
    data_pred = []
    labels_pred = []
    num_pred=[]
    arr=np.arange(647)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(arr)
    print(arr)
    # print(df)
    feature_list = getFeatureName(df, 0, feature_num)
    # feature_list.append('country_pra')
    for i in arr:
        data0=[]
        for feature0 in feature_list:
            data0.append(df.loc[i, feature0])

        if df.loc[i, 'train_predict']==0:
            data.append(data0)
            labels.append(df.loc[i, 'label'])
        else:
            data_pred.append(data0)
            labels_pred.append(df.loc[i, 'label'])
            num_pred.append(df.loc[i, 'id_no'])

    data = np.array(data, dtype="float")
    labels = np.array(labels)
    data_pred = np.array(data_pred, dtype="float")
    labels_pred = np.array(labels_pred)
    global FEATURES
    FEATURES = len(data0)
    return data, labels, data_pred, labels_pred, num_pred
    pass

def getFeatureName(df,threshold,feature_num):
    pd_col_names = list(df)[:len(list(df))]
    insert_num=2
    names = []
    X = []

    df=df[df['train_predict'].isin([0])]

    X.append(df['buffer'].values)
    names.append('buffer')
    X.append(df['latitude'].values)
    names.append('latitude')
    X.append(df['ave_ntl_a'].values)
    names.append('ave_ntl_a')
    X.append(df['ave_ntl_p'].values)
    names.append('ave_ntl_p')
    X.append(df['ntl_stdev'].values)
    names.append('ntl_stdev')
    X.append(df['ntl_slope'].values)
    names.append('ntl_slope')



    X.append(df['month_trend_xielv'].values)
    names.append('month_trend_xielv')
    X.append(df['year_trend_xielv'].values)
    names.append('year_trend_xielv')
    X.append(df['year_trend_xielv_dx'].values)
    names.append('year_trend_xielv_dx')
    X.append(df['month_raw_xielv'].values)
    names.append('month_raw_xielv')
    X.append(df['year_raw_xielv'].values)
    names.append('year_raw_xielv')
    X.append(df['month_seasonal_std'].values)
    names.append('month_seasonal_std')
    X.append(df['year_seasonal_std'].values)
    names.append('year_seasonal_std')
    X.append(df['seasonal_std'].values)
    names.append('seasonal_std')
    X.append(df['seasonal_ratio_ave'].values)
    names.append('seasonal_ratio_ave')
    X.append(df['seasonal_ratio_geoave'].values)
    names.append('seasonal_ratio_geoave')

    X.append(df['p'].values)
    names.append('p')
    X.append(df['d'].values)
    names.append('d')
    X.append(df['q'].values)
    names.append('q')
    X.append(df['P'].values)
    names.append('P')
    X.append(df['D'].values)
    names.append('D')
    X.append(df['Q'].values)
    names.append('Q')
    X.append(df['s'].values)
    names.append('s')

    X.append(df['Sviirs_sum_whs'].values)
    names.append('Sviirs_sum_whs')

    X.append(df['IR_sum_whs_1'].values)
    names.append('IR_sum_whs_1')

    X.append(df['IR_sum_whs_2'].values)
    names.append('IR_sum_whs_2')

    X.append(df['IR_sum_whs_3'].values)
    names.append('IR_sum_whs_3')
    X.append(df['IR_sum_whs_4'].values)
    names.append('IR_sum_whs_4')
    X.append(df['IR_sum_whs_5'].values)
    names.append('IR_sum_whs_5')
    X.append(df['IR_sum_whs_6'].values)
    names.append('IR_sum_whs_6')
    X.append(df['IR_sum_whs_7'].values)
    names.append('IR_sum_whs_7')
    X.append(df['IR_sum_whs_8'].values)
    names.append('IR_sum_whs_8')
    X.append(df['IR_sum_whs_9'].values)
    names.append('IR_sum_whs_9')
    X.append(df['IR_sum_whs_10'].values)
    names.append('IR_sum_whs_10')
    X.append(df['IR_sum_whs_11'].values)
    names.append('IR_sum_whs_11')
    X.append(df['IR_sum_whs_12'].values)
    names.append('IR_sum_whs_12')
    X.append(df['google_country_var'].values) #google trends 国家方差
    names.append('google_country_var')
    X.append(df['per_GDP_LRR_max'].values)  #
    names.append('per_GDP_LRR_max')
    X.append(df['per_GDP_LRR_min'].values)  #
    names.append('per_GDP_LRR_min')
    X.append(df['per_GDP_LRR_ave'].values)  #
    names.append('per_GDP_LRR_ave')
    # X.append(df['war_LRR'].values)  #
    # names.append('war_LRR')
    X.append(df['per_area_conflict'].values)  #
    names.append('per_area_conflict')

    # X.append(df['conflict'].values)
    # names.append('conflict')

    # X.append(df['google_sight_ave'].values) #google trends  sight
    # names.append('google_sight_ave')
    # X.append(df['google_country_ave'].values)  # google trends 国家ave
    # names.append('google_country_ave')
    #
    # X.append(df['google_country_min'].values)  # google trends 国家min
    # names.append('google_country_min')





    # X.append(df['warning_soc'].values)
    # names.append('warning_soc')
    #X.append(df['country_pra'].values)
    #names.append('country_pra')
    for num0 in range(10):
        X.append(df.iloc[:, 10 +insert_num+ num0].values)
        names.append(pd_col_names[10 +insert_num+ num0])
    for num0 in range(14):
        X.append(df.iloc[:, 22 + insert_num + num0].values)
        names.append(pd_col_names[22 + insert_num + num0])
    for num0 in range(6):
        X.append(df.iloc[:, 37 + insert_num + num0].values)
        names.append(pd_col_names[37 + insert_num + num0])
    print(names)
    Y = df['label'].values

    X = np.array(X).T
    # print(np.where(np.isnan(X)))
    where_are_nan = np.isnan(X)
    X[where_are_nan] = 1
    Y = np.array(Y)

    ranks = {}
    ridge = Ridge(alpha=7)
    ridge.fit(X, Y)
    print("[INFO] Ridge data...")
    ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)

    rf = RandomForestRegressor()
    rf.fit(X, Y)
    print("[INFO] RF data...")
    ranks["RF"] = rank_to_dict(rf.feature_importances_, names)
    f, pval = f_regression(X, Y, center=True)
    print("[INFO] Corr. data...")
    ranks["Corr."] = rank_to_dict(f, names)
    mine = MINE()
    mic_scores = []
    #添加平衡X，Y函数

    for i in range(X.shape[1]):
        mine.compute_score(X[:, i], Y)
        m = mine.mic()
        mic_scores.append(m)
    print("[INFO] MIC data...")
    ranks["MIC"] = rank_to_dict(mic_scores, names)
    r = {}
    for name in names:
        r[name] = round(np.mean([ranks[method][name]
                                 for method in ranks.keys()]), 2)
    methods = sorted(ranks.keys())
    print("[INFO] Mean data...")
    ranks["Mean"] = r
    methods.append("Mean")
    print("\t%s" % "\t".join(methods))
    temp = sorted(ranks['Mean'].items(), key=lambda x: x[1], reverse=True)
    return_list=[]
    for name in temp:
        print("%s\t%s" % (name[0], "\t".join(map(str,[ranks[method][name[0]] for method in methods]))))
        if threshold>0:
            if name[1]>threshold:
                return_list.append(name[0])
            else:
                break
        else:
            if feature_num>0:
                return_list.append(name[0])
                if len(return_list)>=feature_num:
                    break
            else:
                return_list.append(name[0])
    return return_list


def xlsx_primaries_df(xlsx_dir, df):
    prab_list = [0.004892924, 0.000343289, 5.36E-05, 2.21E-09, 2.57E-12, 4.78E-06, 2.05E-10, 1.94E-11, 9.14E-09,
                 7.71E-11, 0.004893637, 8.87E-05, 1.89E-06, 9.77E-11, 2.93E-09, 6.47E-09, 0.999999857, 0.999953198,
                 7.84E-08, 1.43E-09, 0.599154765, 7.11E-12, 0.788927093, 0.998856744, 0.999981067, 0.060459821,
                 0.917245513, 0.125014699, 2.00E-08, 0.999985889, 2.92E-09, 1.12E-07, 8.36E-08, 2.69E-10, 0.112070189,
                 0.606144904, 0.802615374, 1.41E-08, 0.694742459, 0.918471116, 0.999998188, 0.003849512, 4.48E-07,
                 1.49E-12, 0.398401127, 0.649997304, 0.807286904, 0.999999994, 2.28E-09, 0.997372648, 0.667356647,
                 0.366476882, 0.999999887, 3.55E-07, 1, 7.50E-07, 0.436902273, 0.949921137, 0.303883362, 0.736213061,
                 0.798245135, 2.67E-13, 0.173163199, 9.06E-13, 0.03854537, 5.63E-10, 0.218658966, 0.200621496,
                 0.999999988, 2.02E-08, 0.999997526, 0.160232631, 0.158921811, 0.916189438, 0.999998498, 0.025017626,
                 0.319172382, 0.963176169, 0.211061769, 0.994021514, 0.981383196, 0.882869317]
    txt_file = xlsx_dir + 'primary threats.txt'
    pri_dict = primaryList(txt_file)
    heritage_id2num={}
    for i in range(len(df)):
        heritage_id2num[df.loc[i,'id_no']]=i
    df['warning_soc']=[0 for i in range(len(df))]
    for i in os.listdir(xlsx_dir):
        if i.endswith('.xlsx'):
            print(i)
            xls_file=xlsx_dir+i
            data_rs= xlrd.open_workbook(xls_file)
            table = data_rs.sheets()[0]
            #data_rs=readXLS(xls_file)
            #data_nrows=data_rs.nrows
            nrows = table.nrows
            warning_name=re.sub(' */ *', ' / ',table.cell_value(1,7))
            pri_soc_num=0
            warning_pra=0
            for pri0 in pri_dict:
                for soc_pri0 in pri_dict[pri0]['details']:
                    if warning_name==soc_pri0:
                        warning_pra=prab_list[pri_soc_num]
                        break
                    pri_soc_num=pri_soc_num+1
                if warning_pra!=0:
                    break

            if warning_pra>=WARNINGYUZHI:
                for row0 in range(1,nrows):
                    row=table.row_values(row0)
                    try:
                        df.loc[heritage_id2num[row[2]],'warning_soc']=1
                    except:
                        pass
                    # print(row)
                    # print(df.loc[heritage_id2num[row[2]],:])


    return df




def country_danger_pra(pd_heritage):
    country_dict={}
    for i in range(len(pd_heritage)):
        for country0 in pd_heritage.loc[i,'udnp_code'].split(','):
            country0=country0.strip()
            if country0 not in country_dict:
                country_dict[country0]={'danger':0,'not_danger':0}
            if type(pd_heritage.loc[i,'danger_list'])==type('str') :
                country_dict[country0]['danger']=country_dict[country0]['danger']+1
            else:
                country_dict[country0]['not_danger'] = country_dict[country0]['not_danger'] + 1
    for country0 in country_dict:
        country_dict[country0]['danger_pra']=country_dict[country0]['danger']/(country_dict[country0]['danger']+country_dict[country0]['not_danger'])


    #pd_heritage['country_pra']=''
   # for i in range(len(pd_heritage)):
     #   country_pra_list=[]
     #   for country0 in pd_heritage.loc[i,'udnp_code'].split(','):
     #       country0=country0.strip()
     #       country_pra_list.append(country_dict[country0]['danger_pra'])
     #   pd_heritage.loc[i,'country_pra']=np.mean(country_pra_list)

    pd_heritage.to_excel('H:\wh师兄\data\\whs_csv_flag_cl05.xlsx')
    print(pd_heritage)
    return pd_heritage



def load_data(path):
    print("[INFO] loading data...")
    pd_heritage = pd.read_excel(path)
    print("[INFO] counting soc data...")
    xlsx_dir = 'H:\\wh师兄\\xlsx\\'
    pd_heritage = xlsx_primaries_df(xlsx_dir, pd_heritage)
    #print("[INFO] resetting country_pra data...")
    pd_heritage = country_danger_pra(pd_heritage)
    print("[INFO] regulariting data...")
    pd_heritage=regularit_pd(pd_heritage)
    print("[INFO] inserting flag data...")
    pd_heritage=danger_year(pd_heritage)
    return pd_heritage

def pd_array(pd_heritage):
    print("[INFO] arraying data...")
    print(pd_heritage)
    # 筛选掉发达国家
    '''df1 = pd_heritage[~ pd_heritage['udnp_code'].str.contains('cyp')]
    df2 = df1[~ df1['udnp_code'].str.contains('isr')]
    df3 = df2[~ df2['udnp_code'].str.contains('jpn')]
    df4 = df3[~ df3['udnp_code'].str.contains('sgp')]
    df5 = df4[~ df4['udnp_code'].str.contains('aut')]
    df6 = df5[~ df5['udnp_code'].str.contains('bel')]
    df7 = df6[~ df6['udnp_code'].str.contains('cze')]
    df8 = df7[~ df7['udnp_code'].str.contains('dnk')]
    df9 = df8[~ df8['udnp_code'].str.contains('est')]
    df10 = df9[~ df9['udnp_code'].str.contains('fin')]
    df11 = df10[~ df10['udnp_code'].str.contains('fra')]
    df12= df11[~ df11['udnp_code'].str.contains('deu')]
    df13 = df12[~ df12['udnp_code'].str.contains('grc')]
    df14 = df13[~ df13['udnp_code'].str.contains('isl')]
    df15 = df14[~ df14['udnp_code'].str.contains('irl')]
    df16 = df15[~ df15['udnp_code'].str.contains('ita')]
    df17 = df16[~ df16['udnp_code'].str.contains('lva')]
    df18 = df17[~ df17['udnp_code'].str.contains('ltu')]
    df19 = df18[~ df18['udnp_code'].str.contains('lux')]
    df20 = df19[~ df19['udnp_code'].str.contains('mlt')]
    df21 = df20[~ df20['udnp_code'].str.contains('nld')]
    df22 = df21[~ df21['udnp_code'].str.contains('nor')]
    df23= df22[~ df22['udnp_code'].str.contains('prt')]
    df24 =df23[~ df23['udnp_code'].str.contains('smr')]
    df25= df24[~ df24['udnp_code'].str.contains('svk')]
    df26 = df25[~ df25['udnp_code'].str.contains('svn')]
    df27= df26[~ df26['udnp_code'].str.contains('esp')]
    df28 = df27[~ df27['udnp_code'].str.contains('swe')]
    df29 = df28[~ df28['udnp_code'].str.contains('che')]
    df30 = df29[~ df29['udnp_code'].str.contains('gbr')]
    df31 = df30[~df30['udnp_code'].str.contains('can')]
    df32 = df31[~ df31['udnp_code'].str.contains('usa')]
    df33 = df32[~ df32['udnp_code'].str.contains('aus')]
    df34 = df33[~ df33['udnp_code'].str.contains('nzl')]
    print(df34)'''
    data, labels, data_pred, labels_pred, num_pred=df2array(pd_heritage)
    global CLASS_NUM
    CLASS_NUM = len(Counter(labels))

    # 平衡不濒危与濒危
    #原本的位置在这
    smo = ADASYN(random_state=RANDOM_SEED)
    X_smo, y_smo = smo.fit_sample(data, labels)
    print(Counter(y_smo))
    y_smo = to_categorical(y_smo, num_classes=CLASS_NUM)
    index = np.arange(len(y_smo))
    np.random.seed(seed=RANDOM_SEED)
    np.random.shuffle(index)
    print(index[0:20])

    X_smo = X_smo[index, :]  # X_train是训练集，y_train是训练标签
    y_smo = y_smo[index]
    return X_smo, y_smo, data_pred, labels_pred, num_pred

def binary_recall_specificity(y_true, y_pred, recall_weight=0.9, spec_weight=0.1,e=0.1):
    print(y_true)
    loss1 = backend.categorical_crossentropy(y_true, y_pred)
    loss2 = backend.categorical_crossentropy(backend.ones_like(y_true) / CLASS_NUM, y_true)
    return (1 - e) * loss1 + e * loss2
    # TP = tf.reduce_sum(y_true * tf.round(y_pred))
    # TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    # FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    # FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    # # Converted as Keras Tensors
    # specificity = TN / (TN + FP + backend.epsilon())
    # recall = TP / (TP + FN + backend.epsilon())
    # return 1-(recall_weight*recall + spec_weight*specificity)

def train(X_smo, y_smo):
    # initialize the model
    print("[INFO] compiling model...")
    model = FulCon.build(input_dim=FEATURES,classes=CLASS_NUM)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    # model.compile(loss="mean_absolute_percentage_error", optimizer=opt,metrics=["accuracy"])
    # 0.7698992	0.5903411586534093	0.744809582233429	0.6446590551999999	0.6564285714285713	0.7820289855072463
    #0.787113	0.5496852423174379	0.7580266344547272	0.6130511328	0.5985714285714285	0.7804347826086955
    # model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])
    #0.8562557	5.52612901899015e-10	0.8308322823047638	6.431909218430519e-10	0.6514285714285714	0.7907246376811594
    #0.8655436	5.309874367484647e-10	0.8397204637527466	6.27687119692564e-10	0.6164285714285714	0.7884057971014491
    # model.compile(loss="mae", optimizer=opt, metrics=["accuracy"])
    #0.7699349	1.1802966524099338e-09	0.744725548028946	1.2889474570751192e-09	0.6557142857142856	0.7814492753623186
    #0.7878587	1.0953028803843095e-09	0.7597850048542023	1.2167194321751594e-09	0.5992857142857142	0.7805797101449273
    # model.compile(loss="mape", optimizer=opt, metrics=["accuracy"])
    #0.76981896	0.5904965948970982	0.744809582233429	0.6446412008	0.6564285714285713	0.7821739130434782
    #0.7898306	0.5428432834837961	0.761377044916153	0.6039350728	0.5921428571428571	0.7802898550724637
    #model.compile(loss="msle", optimizer=opt, metrics=["accuracy"])
    #0.8572488	2.708349865612197e-10	0.831934642791748	3.14457881078124e-10	0.6492857142857144	0.79231884057971
    #0.8631921	2.6356756061840223e-10	0.8358566534519195	3.1255646385252474e-10	0.6207142857142857	0.7881159420289854
    model.compile(loss="squared_hinge", optimizer=opt, metrics=["accuracy"])
    # 加上国家google trends 0.8211057	4.318869412070735e-09	0.7985638773441315	4.4895004570484155e-09	0.7714285714285714	0.7881159420289854
    #加上单个遗产点google trends 0.8482548	4.076611975789797e-09	0.8185173952579499	4.317152863740921e-09	0.7071428571428572	0.7807246376811594
   #google trends var,ave,min 0.8220534	4.323953403138605e-09	0.7966763508319855	4.521345168352127e-09	0.7621428571428569	0.7815942028985506
    # model.compile(loss="hinge", optimizer=opt, metrics=["accuracy"])
    #0.7702874	3.679237694829243e-09	0.744809582233429	3.78920401930809e-09	0.6557142857142856	0.7815942028985506
    #0.7888063	3.5916804799622084e-09	0.7602023422718048	3.715071648359298e-09	0.6014285714285713	0.7808695652173913
    # model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    #0.838593	1.79374748371813e-09	0.81676074385643	2.041033813357353e-09	0.6342857142857143	0.79
    #0.845472	1.7296915358901621e-09	0.8259347093105316	2.0009610548615456e-09	0.6064285714285715	0.7914492753623188
    # model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    #0.8374569	1.8011341022568153e-09	0.8176932871341706	2.0296005904674526e-09	0.6385714285714286	0.7894202898550725
    #0.8445423	1.7350462633339705e-09	0.8245830905437469	1.995832796394825e-09	0.6021428571428572	0.7910144927536231
    # model.compile(loss="sparse_categorical_crossentrop", optimizer=opt, metrics=["accuracy"])
    # model.compile(loss="kullback_leibler_divergence", optimizer=opt, metrics=["accuracy"])
    #0.8379685	1.8000511714045382e-09	0.8184506547451019	2.0315531462430956e-09	0.6371428571428571	0.7892753623188404
    #0.84476393	1.7368520434491257e-09	0.8254287302494049	2.004464499652386e-09	0.6021428571428572	0.7918840579710144
    # model.compile(loss="cosine_proximity", optimizer=opt, metrics=["accuracy"])
    #0.8607	-4.387037115712414e-09	0.8347891926765442	-4.284510356187821e-09	0.675	0.7915942028985506
    #0.87314475	-4.427237092758682e-09	0.8371231138706208	-4.297910493612289e-09	0.6107142857142858	0.7831884057971016
    # train the network
    early_stopping = EarlyStopping(monitor='loss', patience=20, verbose=1)
    print("[INFO] training network...")
    H = model.fit(x=X_smo,y=y_smo,
                            validation_split=0.25,validation_steps=1, steps_per_epoch=int(len(X_smo)/BS*0.75),
                            epochs=EPOCHS, verbose=1,shuffle=False,callbacks=[early_stopping])
    # save the model to disk
    print("[INFO] serializing network...")
    model.save('heritage_model.model')

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax2=ax.twinx()
    N = len(H.history["loss"])
    for i in range(N):
        # H.history["loss"][i] = H.history["loss"][i] / 200000000
        # H.history["val_loss"][i] = H.history["val_loss"][i] / 200000000
        H.history["loss"][i]=H.history["loss"][i]
        H.history["val_loss"][i] = H.history["val_loss"][i]
    acc_list.append(H.history["accuracy"][N-1])
    val_acc_list.append(H.history["val_accuracy"][N-1])
    val_loss_list.append(H.history["val_loss"][N-1])
    loss_list.append(H.history["loss"][N-1])

    font={'family' : 'Times New Roman'}

    ax.plot(np.arange(0, N), H.history["loss"], c='royalblue',label="train_loss")
    ax.plot(np.arange(0, N), H.history["val_loss"],c='tomato', label="test_loss")
    # plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    #plt.plot(np.arange(0, N), H.history["val_accuracy"], label="test_acc")
    ax2.plot(np.arange(0, N),H.history["accuracy"],c='lightgreen',label="train_acc")
    ax2.plot(np.arange(0, N),H.history["val_accuracy"], c='mediumorchid', label="test_acc")

    #plt.title("Training Loss and Accuracy on WHS classifier",font)
    ax.set_yticks([0.7+i*0.07 for i in range(10)])
    ax2.set_yticks([0.3+i*0.07 for i in range(10)])
    labels = ax.get_xticklabels() + ax.get_yticklabels()+ ax2.get_xticklabels()+ ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

   # plt.xlabel("Epoch #",font)
   #plt.ylabel("Loss/Accuracy",font)
   #plt.legend(loc="lower left",prop=font)
    plt.savefig('plot'+str(RANDOM_SEED)+'.png',dpi=600)
    plt.close()
    return model
def predict(data_pred, labels_pred, num_pred, model,df):

    df_new=df.set_index('id_no',inplace=False)
    matrix = np.zeros((CLASS_NUM, CLASS_NUM),dtype=int)
    # for i in range(1):

    FN_site=[]
    FP_site=[]
    TN_site = []
    TP_site = []
    for i in range(len(data_pred)):

        input_x=data_pred[i].reshape(1,FEATURES)
        y_pre=model.predict(input_x)
        if y_pre[0][0]<chengfa:
            y_pre0=1
        else:
            y_pre0=0
        # if df_new.loc[num_pred[i],'warning_soc']==1 and labels_pred[i]==1:
        #     y_pre0=1
        matrix[labels_pred[i]][y_pre0]=matrix[labels_pred[i]][y_pre0]+1
        if labels_pred[i]==1 and y_pre0==0:
            FN_site.append(num_pred[i])
        elif labels_pred[i] == 0 and y_pre0 == 1:
            FP_site.append(num_pred[i])
        elif labels_pred[i] == 0 and y_pre0 == 0:
            TN_site.append(num_pred[i])
        elif labels_pred[i] == 1 and y_pre0 == 1:
            TP_site.append(num_pred[i])
    for i in range(CLASS_NUM):
        print(matrix[i])
    return matrix, FN_site, FP_site,TN_site,TP_site
if __name__=='__main__':
    # 显示所有列(参数设置为None代表显示所有行，也可以自行设置数字)
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置数据的显示长度，默认为50
    pd.set_option('max_colwidth', 200)
    # 禁止自动换行(设置为Flase不自动换行，True反之)
    pd.set_option('expand_frame_repr', False)
    heritage_csv = 'H:\wh师兄\data\\筛选掉发达国家_csv.xlsx'
    pd_heritage = load_data(heritage_csv)
    global RANDOM_SEED, acc_list,loss_list,val_acc_list,val_loss_list
    RANDOM_SEED=50
    acc_list=[]
    loss_list=[]
    val_acc_list=[]
    val_loss_list=[]
    danger_acc_list=[]
    danger_recall_list=[]
    total_acc_list=[]
    FP_site_dict={}
    TP_site_dict={}
    while RANDOM_SEED:
        X_smo, y_smo, data_pred, labels_pred, num_pred=pd_array(pd_heritage)
        model = train(X_smo, y_smo)
        model = load_model("heritage_model.model")
        matrix, FN_site, FP_site, TN_site, TP_site=predict(data_pred, labels_pred, num_pred, model,pd_heritage)
        danger_acc_list.append(matrix[1][1]/(matrix[1][1]+matrix[1][0]))
        danger_recall_list.append(matrix[1][1]/(matrix[1][1]+matrix[0][1]))
        total_acc_list.append((matrix[0][0]+matrix[1][1]) / (matrix[0][0]+matrix[0][1]+matrix[1][1] + matrix[1][0]))
        for FP_site0 in FP_site:
            if FP_site0 not in FP_site_dict:
                FP_site_dict[FP_site0]=1
            else:
                FP_site_dict[FP_site0]=FP_site_dict[FP_site0]+1
        for TP_site0 in TP_site:
            if TP_site0 not in TP_site_dict:
                TP_site_dict[TP_site0]=1
            else:
                TP_site_dict[TP_site0]=TP_site_dict[TP_site0]+1
        RANDOM_SEED=RANDOM_SEED-1

    ave_acc=np.mean(acc_list)
    ave_loss = np.mean(loss_list)
    ave_val_acc = np.mean(val_acc_list)
    ave_val_loss = np.mean(val_loss_list)
    ave_danger_acc=np.mean(danger_acc_list)
    ave_recall_acc=np.mean(danger_recall_list)
    ave_total_acc=np.mean(total_acc_list)
    print(ave_acc,ave_loss,ave_val_acc,ave_val_loss,ave_danger_acc,ave_total_acc,ave_recall_acc,sep='\t')

    FP_site_dict=sorted(FP_site_dict.items(), key=lambda item: item[1],reverse=True)
    for i in FP_site_dict:
        print(i[0],end='\t')
        print(i[1]/50)
    print()

    TP_site_dict = sorted(TP_site_dict.items(), key=lambda item: item[1], reverse=True)
    for i in TP_site_dict:
        print(i[0], end='\t')
        print(i[1] / 50)
    print()

