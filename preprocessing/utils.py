import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing


def preprocess_df(df):
    to_be_droped = ['CURRUPB', 'LID', 'REMMNTHS', 'OLTV', 'STATE', 'ZIP', 'PRODUCT', 'CURRRATE']
    #     labeled = ['NMONTHS', 'FORCLOSED']
    #     df_labeled = df[labeled[cate]]
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
    df_dropped.FORCLOSED = df_dropped.FORCLOSED.map({False: 0, True: 1})

    return pd.get_dummies(df_dropped, columns=dummies)


def get_labeled_data(df):
    labels = ['NMONTHS', 'FORCLOSED']
    return df.drop(labels, axis=1), df[labels]


def get_formatted_data(X):
    sclaer = preprocessing.MinMaxScaler()
    X = np.c_[sclaer.fit_transform(X.iloc[:, :18]), X.iloc[:, 18:]]

    return X


def pipeline(df):
    df = preprocess_df(df)
    X, Y = get_labeled_data(df)
    X = get_formatted_data(X)

    return X, Y
