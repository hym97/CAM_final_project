import torch
import training as t
import pandas as pd
import numpy as np
from training import FFNN_classifer

model =FFNN_classifer(49)
model.state_dict(torch.load('state.pth'))
model.eval()
df = pd.read_csv('TrainingData.csv')
df = t.preprocess_df(df)

X, Y = t.get_labeled_data(df)
X, Y = t.get_formatted_data(X, Y)

X,Y = torch.tensor(X), torch.tensor(Y)
a = model(X)
print(a)