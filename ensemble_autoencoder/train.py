import os
import pickle
import random

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# generate the dataset
import torch
from pyod.models.combination import average, maximization, median, aom, moa
from pyod.utils import standardizer, evaluate_print
from scipy.io import loadmat
from sklearn.datasets import make_blobs

# PCA
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from Model.ensemble_autoencoder import EnsembleRandomAutoencoder

def load_variables(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

def random_select(scores, nb_models, random_rate):
    replace_num = int(random_rate * nb_models)
    ind = np.random.choice(np.arange(nb_models), replace_num, replace=False)
    ind2 = np.setdiff1d(np.arange(nb_models), ind)
    temp1 = scores[:, ind]
    temp2 = scores[:, ind2]
    return temp1, temp2

def compute_combination(test_scores_norm):
    from pyod.models.combination import average, maximization, median, aom, moa
    avg = average(test_scores_norm)
    max = maximization(test_scores_norm)
    median = median(test_scores_norm)
    aom = aom(test_scores_norm, n_buckets=5)
    moa = moa(test_scores_norm, n_buckets=5)
    return avg, max, median, aom, moa


def generate_random_dataset():
    # generate
    X1, y1 = make_blobs(n_samples=10000, centers=1, center_box=(10, 15), n_features=15)
    X2, y2 = make_blobs(n_samples=100, centers=1, center_box=(-4, -3), n_features=15)

    # stack the data and shuffle
    X = np.vstack((X1, X2))
    np.random.shuffle(X)

    # stack the data and shuffle
    X = np.vstack((X1, X2))
    np.random.shuffle(X)
    return X

def save_variables(path: str, variables):
    with open(path, 'wb') as f:
        pickle.dump(variables, f)
        f.close()

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

def scores_process(scores):
    scores = scores.detach().numpy()
    return scores.transpose()

def load_datamat(chosen_mat, random_state):
    mat = loadmat(os.path.join('..\\data', chosen_mat))
    X = mat['X']
    y = mat['y'].ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test

def fit(X_train, X_test, random_state, batchs_size, nb_models, epochs, lr, slr):

    # standardizing data for processing
    X_train_norm, X_test_norm = standardizer(X_train, X_test)

    model_ensemble_ae = EnsembleRandomAutoencoder(batchs_size, nb_models, X_train_norm, random_state,
                                                  input_size=X_train_norm.shape[1],
                                                  nb_epochs=epochs, lr=lr, slr=slr)
    model_ensemble_ae.display_loaders_info()
    model_ensemble_ae.print_models()
    model_ensemble_ae.models_summary()
    model_ensemble_ae.fit()

    model_ensemble_ae_preds, model_ensemble_qae_preds, outputs1, outputs2 = model_ensemble_ae.outliers_scoring(
        X_test_norm)
    model_ensemble_ae_preds, model_ensemble_qae_preds = scores_process(model_ensemble_ae_preds), \
                                                        scores_process(model_ensemble_qae_preds)

    # save scores for ensemble
    model_ensemble_ae_preds, model_ensemble_qae_preds = standardizer(model_ensemble_ae_preds, model_ensemble_qae_preds)
    model_preds_scores = {'AE': model_ensemble_ae_preds,
                          'QAE': model_ensemble_qae_preds}
    path = chosen_mat[:-4] + '_' + str(seed)
    if not os.path.exists('scores'):
        os.mkdir('scores')
    path = os.path.join('scores', path)
    save_variables(path, model_preds_scores)
    return path

def inference(y_test, path, swap_rate):
    df_columns = ['Seed', 'AVG1', 'MAX1', 'MED1', 'AOM1', 'MOA1',
                  'AVG2', 'MAX2', 'MED2', 'AOM2', 'MOA2']

    if swap_rate != 0:
        method = 'random'
    else:
        method = 'pure'

    if nb_models == 1:
        method = 'AE'
        df_columns = ['Seed', 'AE', 'QAE']
    eae_result_df = pd.DataFrame(columns=df_columns)

    roc_list1 = []
    roc_list2 = []

    model_preds_scores = load_variables(path)
    model_ensemble_ae_preds = model_preds_scores['AE'][:, :nb_models]
    model_ensemble_qae_preds = model_preds_scores['QAE'][:, :nb_models]

    # random change
    if swap_rate != 0:
        ae_preds1, ae_preds2 = random_select(model_ensemble_ae_preds, nb_models, swap_rate)
        qae_preds1, qae_preds2 = random_select(model_ensemble_qae_preds, nb_models, swap_rate)
        model_ensemble_ae_preds = np.column_stack((ae_preds1, qae_preds2))
        model_ensemble_qae_preds = np.column_stack((qae_preds1, ae_preds2))

    if nb_models == 1:
        roc_list1.append(roc_auc_score(y_test, model_ensemble_ae_preds[:, 0]))
        roc_list2.append(roc_auc_score(y_test, model_ensemble_qae_preds[:, 0]))
    else:
        avg1, max1, med1, aom1, oma1 = compute_combination(model_ensemble_ae_preds)
        avg2, max2, med2, aom2, oma2 = compute_combination(model_ensemble_qae_preds)
        roc_list1.append(
            [roc_auc_score(y_test, avg1), roc_auc_score(y_test, max1), roc_auc_score(y_test, med1),
             roc_auc_score(y_test, aom1), roc_auc_score(y_test, oma1)])
        roc_list2.append(
            [roc_auc_score(y_test, avg2), roc_auc_score(y_test, max2), roc_auc_score(y_test, med2),
             roc_auc_score(y_test, aom2), roc_auc_score(y_test, oma2)])

        roc_np1 = np.array(roc_list1).reshape(-1)
        roc_np2 = np.array(roc_list2).reshape(-1)
        temp_np = np.concatenate((np.array(seed).reshape(-1), roc_np1, roc_np2), axis=0)
        temp_np = temp_np.reshape(1, len(temp_np))
        temp_df = pd.DataFrame(temp_np, columns=df_columns)
        eae_result_df = pd.concat([eae_result_df, temp_df], axis=0, ignore_index=True)


    save_path_name = 'Ensemble_' + method + '_' + chosen_mat[:-4] + '_' + str(swap_rate) \
                     + '_' + str(nb_models) + '.csv'
    if not os.path.exists('results'):
        os.mkdir('results')
    save_path = os.path.join('results', save_path_name)
    eae_result_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    # initial parameters
    dataset = ['arrhythmia.mat', 'cardio.mat', 'glass.mat', 'pendigits.mat']
    chosen_mat = dataset[0]
    seed = 42
    batchs_size = [16, 32, 64]
    nb_models = 10  # the number of base autoencoders
    epochs = 50
    lr = 0.001
    slr = 0.1  # In AE models, sub_learning_rate is not used, but we recommend to set 0.1 for a mark.
    swap_rate = 0  # An important parameter for ensemble mixed autoencoder
    # 0 - dont swap, the results will be an pure ensemble conventional and quddratic autoencoders.
    # 0.1~0.9 - swap rate, the results will be 2 ensemble mixed autoencoders.

    random_state = random_seed(seed)
    X_train, X_test, _, y_test = load_datamat(chosen_mat, random_state)

    path = fit(X_train, X_test, random_state, batchs_size, nb_models, epochs, lr, slr)
    inference(y_test, path, swap_rate)
