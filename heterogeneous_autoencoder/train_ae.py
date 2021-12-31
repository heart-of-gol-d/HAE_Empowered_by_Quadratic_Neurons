from __future__ import division
from __future__ import print_function

import os
import random
import sys
# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
import torch
from heterogeneous_autoencoder.Model.HAutoEncoder import AutoEncoder
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from heterogeneous_autoencoder.Model.HAutoEncoder import AutoEncoder

def random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random_state = np.random.RandomState(seed)
    return random_state




if __name__ == "__main__":

    # Define data file and read X and y
    mat_file_list = [
        'arrhythmia.mat',
        'cardio.mat',
        'glass.mat',
        'lympho.mat',
        'musk.mat',
        'optdigits.mat',
        'pendigits.mat',
        'pima.mat',
        'satimage-2.mat',
        'vertebral.mat',
        'vowels.mat',
        'wbc.mat',
    ]
    seed = 42
    random_state = random_seed(seed)
    ae_model = 'AE'  # Available parameter:  AE, QAE, HAE_X, HAE_Y, HAE_I
    # Default hyperparameters, all need to careful design for different data sets.
    learning_rate = 0.001
    batch_size = 32
    epochs = 200
    sub_learning_rate = 0.1  # In AE models, sub_learning_rate is not used, but we recommend to set 0.1 for a mark.
    # In default setting, it will generate this autoencoder:
    # (input, 64, ReLU)
    # (64, 32, ReLU)
    # (32, 64, ReLU)
    # (64, output, Sigmoid)
    hidden_neurons = [64, 32]

    df_columns = ['Datasets', 'AUC', 'PRN', 'PRE', 'ReCall', 'F1']
    ae_result_df = pd.DataFrame(columns=df_columns)
    for mat_file in mat_file_list:
        print("\n... Processing", mat_file, '...')
        mat = loadmat(os.path.join('../data', mat_file))
        X = mat['X']
        y = mat['y'].ravel()
        # 0-normal 1-abnormal
        outliers_fraction = np.count_nonzero(y) / len(y)
        outliers_percentage = round(outliers_fraction * 100, ndigits=4)  # 返回浮点数四舍五入的值

        classifiers = {
            'AE': AutoEncoder(hidden_neurons=hidden_neurons, epochs=epochs, preprocessing=False, contamination=outliers_fraction,
                              learning_rate=learning_rate, batch_norm=True, dropout_rate=0.5, quadratic=True,
                              sub_learning_rate=sub_learning_rate, batch_size=batch_size, hybird=False,
                              hybird_style='X'),
            'QAE': AutoEncoder(hidden_neurons=hidden_neurons, epochs=epochs, preprocessing=False, contamination=outliers_fraction,
                               learning_rate=learning_rate, batch_norm=True, dropout_rate=0.5, quadratic=True,
                               sub_learning_rate=sub_learning_rate, batch_size=batch_size, hybird=False,
                               hybird_style='X'),
            'HAE_X': AutoEncoder(hidden_neurons=hidden_neurons, epochs=epochs, preprocessing=False, contamination=outliers_fraction,
                                  learning_rate=learning_rate, batch_norm=True, dropout_rate=0.5, quadratic=True,
                                  sub_learning_rate=sub_learning_rate, batch_size=batch_size, hybird=True,
                                  hybird_style='X'),
            'HAE_Y': AutoEncoder(hidden_neurons=hidden_neurons, epochs=epochs, preprocessing=False, contamination=outliers_fraction,
                                  learning_rate=learning_rate, batch_norm=True, dropout_rate=0.5, quadratic=True,
                                  sub_learning_rate=sub_learning_rate, batch_size=batch_size, hybird=True,
                                  hybird_style='Y'),
            'HAE_I': AutoEncoder(hidden_neurons=hidden_neurons, epochs=epochs, preprocessing=False, contamination=outliers_fraction,
                                  learning_rate=learning_rate, batch_norm=True, dropout_rate=0.5, quadratic=True,
                                  sub_learning_rate=sub_learning_rate, batch_size=batch_size, hybird=True,
                                  hybird_style='I')
        }

        clf = classifiers[ae_model]  # choose an autoencoder

        # 60% data for training and 40% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                            random_state=random_state)
        # standardizing data for processing
        X_train_norm, X_test_norm = standardizer(X_train, X_test)

        clf.fit(X_train_norm)
        test_scores = clf.decision_function(X_test_norm)
        y_predict = clf.predict(X_test_norm)

        # we only use AUCs for comparison in the paper, however, the program will compute other metrics.
        auc = round(roc_auc_score(y_test, test_scores), ndigits=4)
        prn = round(precision_n_scores(y_test, test_scores), ndigits=4)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_predict, average='binary')
        precision = round(precision, ndigits=4)
        recall = round(recall, ndigits=4)
        f1 = round(f1, ndigits=4)

        ae_result = [mat_file[:-4], auc, prn, precision, recall, f1]
        ae_result_np = np.array(ae_result).reshape(1, -1)
        temp_df = pd.DataFrame(ae_result_np, columns=df_columns)
        ae_result_df = pd.concat([ae_result_df, temp_df], axis=0)
        if not os.path.exists('results'):
            os.mkdir('results')
        ae_result_df.to_csv('results/%s_bs%d_lr%f_slr_%f_seed%d.csv' % (ae_model, batch_size, learning_rate,
                                                                         sub_learning_rate, seed), index=False)
