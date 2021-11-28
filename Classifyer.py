## author: hym97
## date: 2021/11/27

from matplotlib import pyplot as plt
from imblearn.ensemble import BalancedRandomForestClassifier
import preprocessing.utils as utils
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix

sns.set_context("poster")
def main():
    df = pd.read_csv('./data/TrainingData.csv')
    X, Y = utils.pipeline(df)

    Y = Y.values[:,1]
    data = np.c_[X,Y]
    np.random.shuffle(data)
    train, validate, test = np.split(data,
                                     [int(.6 * data.shape[0]), int(.8 * data.shape[0])])

    train_X, train_Y = train[:,:-1], train[:,-1]
    validate_X, validate_Y = validate[:,:-1], validate[:,-1]
    brf = BalancedRandomForestClassifier(n_estimators=75, random_state=37)
    brf.fit(train_X,train_Y)
    predict_Y = brf.predict(validate_X)

    fig, axs = plt.subplots(ncols=1, figsize=(10, 5))
    plot_confusion_matrix(brf, validate_X, validate_Y, ax=axs, colorbar=False)
    axs.set_title("Balanced random forest")
    plt.show()


if __name__ == '__main__':
    main()
