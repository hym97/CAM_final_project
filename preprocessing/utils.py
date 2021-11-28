## author: hym97
## date: 2021/11/27

import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing


def preprocess_df(df, test=False):
    '''
    Used to drop columns, pad nans, and get dummies
    :param df (pd.DataFrame): full_dataset(with labels)
    :param test (bool): if it is test data set
    :return:
    df (pd.DataFrame): processed dataset
    '''
    to_be_droped = ['CURRUPB', 'LID', 'REMMNTHS', 'OLTV', 'STATE', 'ZIP', 'PRODUCT', 'CURRRATE']
    dummies = ['CHNL', 'PROP', 'PURPOSE', 'OCCSTAT', 'SELLER']
    df_dropped = df.drop(to_be_droped, axis=1)

    pad_NUMBO, pad_DTI, pad_CSCOREB, pad_CSCOREC, pad_OCLTV = df_dropped.NUMBO.median(), df_dropped.DTI.median(), \
                                                              df_dropped.CSCOREB.median(), df_dropped.CSCOREC.median(), df_dropped.OCLTV.median()
    fill_list = ['NUMBO', 'DTI', 'CSCOREB', 'CSCOREC', 'OCLTV']
    padding = [pad_NUMBO, pad_DTI, pad_CSCOREB, pad_CSCOREC, pad_OCLTV]
    for idx, column in enumerate(fill_list):
        df_dropped[column] = df_dropped[column].fillna(padding[idx])
    df_dropped.FIRSTFLAG = df_dropped.FIRSTFLAG.map({'N': 0, 'Y': 1}, na_action='ignore').fillna(method='bfill')
    df_dropped.IO = df_dropped.IO.map({'N': 1}, na_action='ignore').fillna(0)
    df_dropped.MIPCT = df_dropped.MIPCT.fillna(0)
    if not test:
        df_dropped.FORCLOSED = df_dropped.FORCLOSED.map({False: 0, True: 1})

    return pd.get_dummies(df_dropped, columns=dummies)


def get_labeled_data(df):
    '''
    split train and label
    :param df (pd.DataFrame): full dataset
    :return: train, labels
    '''
    labels = ['NMONTHS', 'FORCLOSED']
    return df.drop(labels, axis=1), df[labels]


def get_formatted_data(X):
    '''
    scale all data into [0,1]
    :param X (pd.DataFrame): train data
    :return:
    X (numpy.ndarray)
    '''
    sclaer = preprocessing.MinMaxScaler()
    X = np.c_[sclaer.fit_transform(X.iloc[:, :18]), X.iloc[:, 18:]]

    return X


def pipeline(df):
    '''
    pipeline for training data set
    :param df (pd.DataFrame): training dataset
    :return: train (numpy.ndarray), labels(pd.DataFrame)
    '''
    df = preprocess_df(df)
    X, Y = get_labeled_data(df)
    X = get_formatted_data(X)

    return X, Y


def pipeline_test(df):
    '''
    pipeline for testing data set
    :param df (pd.DataFrame): testing dataset
    :return: test (numpy.ndarray)
    '''
    df = preprocess_df(df, True)
    X = get_formatted_data(df)

    return X