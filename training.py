import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


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
    return df.drop(labels, axis = 1), df[labels]


def get_formatted_data(X, Y):
    sclaer = preprocessing.MinMaxScaler()
    X = np.c_[sclaer.fit_transform(X.iloc[:, :18]), X.iloc[:, 18:]]

    Y = pd.get_dummies(Y, columns = ['FORCLOSED'])
    return X, Y.values


class FFNN_classifer(nn.Module):
    def __init__(self, input_size):
        super(FFNN_classifer, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 10)
        self.layer4 = nn.Linear(10, 1)
    def forward(self, input_data):
        input_data =input_data.float()
        output = self.layer1(input_data)
        output = F.relu(output)

        output = self.layer2(output)
        output = F.relu(output)

        output = self.layer3(output)
        output = F.relu(output)

        output = self.layer4(output)

        return output


def train_model(input_data, input_labels, optimizer, model,loss_func):
    optimizer.zero_grad()

    output = model(input_data)
    loss = loss_func(output.squeeze(1), input_labels.float())

    loss.backward()
    optimizer.step()

    return loss


def mini_batch(batch_size, input_data, label):
    length = len(input_data)
    batch_num = math.ceil(length / batch_size)

    for i in range(batch_num):
        input_batch, input_label = input_data[batch_size*i:batch_size * (i + 1), :], \
                                   label[batch_size*i:batch_size * (i + 1)]
        yield input_batch, input_label


def evaluate_model(dev_input, dev_label, model, loss_func):
    dev_input, dev_label = torch.tensor(dev_input), torch.tensor(dev_label)
    model.eval()
    dev_output = model(dev_input)
    loss = loss_func(dev_output.squeeze(1), dev_label.float())
    print('Performance on Dev set: {:.2f}'.format(loss.detach().numpy()))
    model.train()
    return

def main():
    df = pd.read_csv('TrainingData.csv')
    df = preprocess_df(df)

    train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

    train_X, train_Y = get_labeled_data(train)
    train_X, train_Y = get_formatted_data(train_X, train_Y)

    val_X, val_Y = get_labeled_data(validate)
    val_X, val_Y = get_formatted_data(val_X, val_Y)

    test_X, test_Y = get_labeled_data(test)
    test_X, test_Y = get_formatted_data(test_X, test_Y)

    epoch, N_epoch = 0, 256
    batch_size = 32


    n_MONTH = np.log(train_Y[:,0])
    model = FFNN_classifer(49)
    optimizer = optim.Adam(model.parameters())
    loss_func = nn.MSELoss()
    while epoch < N_epoch:
        for input_batch, input_label in mini_batch(batch_size, train_X, n_MONTH):
            input_batch, input_label = torch.tensor(input_batch), torch.tensor(input_label)
            loss = train_model(input_batch, input_label, optimizer, model, loss_func)


        evaluate_model(val_X, val_Y[:,0], model, loss_func)

        epoch += 1

    evaluate_model(test_X, test_Y[:,1:], model, loss_func)
    torch.save(model.state_dict(),'./state'  + '.pth')
if __name__ == '__main__':
    main()
