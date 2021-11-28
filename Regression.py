## author: hym97
## date: 2021/11/27

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import preprocessing.utils as utils

class FFNN_classifer(nn.Module):
    def __init__(self, input_size):
        '''
        Args:
        input_size (int): dimensions of train_data
        '''
        super(FFNN_classifer, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 10)
        self.layer4 = nn.Linear(10, 1)
        self.dropout = nn.Dropout(.2)

    def forward(self, input_data):

        input_data =input_data.float()
        output = self.layer1(input_data)
        output = F.relu(output)

        output = self.layer2(output)
        output = F.relu(output)
        output = self.dropout(output)

        output = self.layer3(output)
        output = F.relu(output)

        output = self.layer4(output)

        return output

def cast_data_types(input_data, input_labels = None):
    '''
    Args:
    input_data (numpy.ndarray): traning data without labels

    input_labels (numpy.ndarray): labels correspond to training data (optional)

    Return:
        input_data(torch.tensor), input_labels(torch.tensor) or None
    '''
    if type(input_data) != torch.tensor:
        input_data = torch.tensor(input_data)
    if input_labels is not None and type(input_labels) != torch.tensor:
        input_labels = torch.tensor(input_labels).float()

    return input_data, input_labels

def train_model(input_data, input_labels, optimizer, model,loss_func):
    optimizer.zero_grad()

    output = model(input_data)
    loss = loss_func(output.squeeze(1), input_labels.float())

    loss.backward()
    optimizer.step()

    return loss


def mini_batch(batch_size, input_data, input_labels):
    '''
    get mini-batch data, no shuffle included
    :param batch_size: batch_size needed
    :param input_data: training_dataset(all)
    :param input_labels: training_labels(all)
    :return: input_data(batch_size, dims), input_labels(batch_size)
    '''
    length = len(input_data)
    batch_num = math.ceil(length / batch_size)

    for i in range(batch_num):
        input_batch, input_label = input_data[batch_size*i:batch_size * (i + 1), :], \
                                   input_labels[batch_size*i:batch_size * (i + 1)]
        yield input_batch, input_label


def evaluate_model(dev_input, dev_label, model, loss_func):
    model.eval()
    dev_output = model(dev_input)
    loss = loss_func(dev_output.squeeze(1), dev_label)
    print('Performance on Dev set: {:.2f}'.format(loss.item()))
    model.train()
    return

def main():
    df = pd.read_csv('./data/TrainingData.csv')
    df = utils.preprocess_df(df)

    train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

    train_X, train_Y = utils.get_labeled_data(train)
    train_X = utils.get_formatted_data(train_X)

    val_X, val_Y = utils.get_labeled_data(validate)
    val_X = utils.get_formatted_data(val_X)

    test_X, test_Y = utils.get_labeled_data(test)
    test_X = utils.get_formatted_data(test_X)

    train_Y, val_Y, test_Y = train_Y.values, val_Y.values, test_Y.values
    ## Global Params
    N_epoch = 256
    batch_size = 32

    epoch = 0
    n_MONTH = np.log(train_Y[:,0])
    model = FFNN_classifer(49)
    optimizer = optim.Adam(model.parameters())
    loss_func = nn.MSELoss()
    while epoch < N_epoch:
        for input_batch, input_label in mini_batch(batch_size, train_X, n_MONTH):
            input_batch, input_label = cast_data_types(input_batch, input_label)
            loss = train_model(input_batch, input_label, optimizer, model, loss_func)


        val_X, val_Y = cast_data_types(val_X, np.log(val_Y[:,0]))
        evaluate_model(val_X, val_Y, model, loss_func)

        epoch += 1

    test_X, test_Y = cast_data_types(test_X, test_Y[:,0])
    evaluate_model(test_X, np.log(test_Y), model, loss_func)
    torch.save(model.state_dict(),'./state'  + '.pth')

if __name__ == '__main__':
    main()
