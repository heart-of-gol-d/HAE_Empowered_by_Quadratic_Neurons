import math
import os.path
import random
from itertools import product

import numpy as np
import torch
from matplotlib import gridspec
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from AEModel.QAE import QAE
from utils.DataSetLoader import *
from AEModel.AE import AutoEncoder
from utils.train_function import *
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pickle
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()


def load_model(mod_name: str, dataset_name: str, p: float):
    path = mod_name + '.pth'
    model_path = os.path.join('model', path)
    if use_gpu:
        save_info = torch.load(model_path)
    else:
        save_info = torch.load(model_path, map_location=torch.device('cpu'))
    img_size1, img_size2 = select_image_size(dataset_name)
    model_name_split = split_model_name(mod_name)
    model, _, loss_func = select_model(model_name_split, img_size1, img_size2, p)
    model.load_state_dict(save_info['model'])
    return model, loss_func


def save_model(model, batch_size, lr, model_name, loss):
    if not os.path.exists('model'):
        os.mkdir('model')
    train_model_name = model_name + '.pth'
    save_path = os.path.join('model', train_model_name)
    save_info = {'model': model.state_dict(),
                 'batch': batch_size,
                 'lr': lr,
                 'loss': loss}
    torch.save(save_info, save_path)


def select_image_size(dataset_name: str):
    if dataset_name in ('MNIST', 'FMNIST'):
        img_size1 = 28
        img_size2 = 28
    if dataset_name == 'YALEB':
        img_size1 = 192
        img_size2 = 168
    if dataset_name == 'olivetti':
        img_size1 = 64
        img_size2 = 64
    return img_size1, img_size2


def split_model_name(model_name_path: str):
    ind = model_name_path.index('_')
    return model_name_path[:ind]


def select_model(model_name_split: str, image_size1: int = 28, image_size2: int = 28, lr=0.001, slr=0.1, lambd=1e-3):
    if model_name_split in ('QAE', 'QCAE', 'QSAE'):
        net = QAE(image_size1, image_size2)
        group_r, group_g, group_b = group_parameters(net)
        optimizer = torch.optim.Adam([
            {"params": group_r},
            {"params": group_g, "lr": lr * slr},
            {"params": group_b, "lr": lr * slr},
        ], lr=lr)
        if model_name_split in ('QAE', 'QSAE'):
            loss_func = nn.MSELoss()
        if model_name_split == 'QCAE':
            loss_func = ContractiveLoss(ae=net, lambd=lambd, model_name=model_name_split)
    if model_name_split == 'CAE':
        net = AutoEncoder(image_size1, image_size2)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        loss_func = ContractiveLoss(ae=net, lambd=lambd)

    if model_name_split in ('SAE', 'AE'):
        net = AutoEncoder(image_size1, image_size2)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        loss_func = nn.MSELoss()
    return net, optimizer, loss_func


def train(lr, batch_size, epoch, dataloader, model_name_path: str, dataset_name: str, slr: float, p: float):
    image_size1, image_size2 = select_image_size(dataset_name)
    model_name_split = split_model_name(model_name_path)
    net, optimizer, loss_func = select_model(model_name_split, image_size1, image_size2, lr, slr, p)
    if use_gpu:
        net.cuda()
    loss_min = 100000
    loss_list = []
    for e in range(epoch):
        total_loss = 0
        net.train()
        loss = 0
        for step, (x, y) in enumerate(dataloader):
            if use_gpu:
                x, y = x.cuda(), y.cuda()
            b_x = x.view(-1, image_size1 * image_size2)
            b_y = x.view(-1, image_size1 * image_size2)
            if model_name_split in ('CAE', 'AE', 'QAE', 'QCAE'):
                encoded, decoded = net(b_x)
                loss = loss_func(decoded, b_y)
            elif model_name_split in ('SAE', 'QSAE'):
                encoded, decoded = net(b_x)
                rho_hat = torch.mean(encoded, dim=0, keepdim=True)
                rho = torch.FloatTensor([RHO for _ in range(encoded.shape[1])]).unsqueeze(0)
                if use_gpu:
                    rho, rho_hat = rho.cuda(), rho_hat.cuda()
                sparsity_penalty = p * kl_divergence(rho, rho_hat)
                loss = loss_func(decoded, b_y) + sparsity_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % 20 == 0:
                print('Epoch:%d, Step [%d/%d], Loss: %.4f'
                      % (e + 1, step + 1, len(data_loaders), loss.item()))
        loss_list.append(total_loss / len(data_loaders))
        if total_loss < loss_min and e > 5:
            loss_min = total_loss
            save_model(net, batch_size, lr, model_name_path, loss_min)
            print('model_saved')

    return loss_list


def inference(dataloader, model_name, dataset_name, p=1e-3):
    reconstruct = []
    image_size1, image_size2 = select_image_size(dataset_name)
    net, loss_fun = load_model(model_name, dataset_name, p)
    loss_list = []
    if use_gpu:
        net.cuda()
    net.eval()
    # endregion
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(dataloader):
            if use_gpu:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            batch_x = batch_x.view(-1, image_size1 * image_size2)
            _, result = net(batch_x)
            loss = loss_fun(result, batch_x)
            loss_list.append(math.sqrt(loss.item()))
            reconstruct.append(result)
        avg_loss = sum(loss_list) / len(dataloader)

        print('Inference Loss: %.4f' % avg_loss)
        return reconstruct, avg_loss



def random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def autoencoder(mod_name, dataset_name, dataloader, p, reduce_dimension):
    image_size1, image_size2 = select_image_size(dataset_name)
    net, _ = load_model(mod_name, dataset_name, p)
    net.eval()
    # change X size
    X = np.ones([1, reduce_dimension])
    y = np.ones([1, 1])
    for i, (batch_x, batch_y) in enumerate(dataloader):
        # if use_gpu:
        #     batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        batch_x = batch_x.view(-1, image_size1 * image_size2)
        encoded, _ = net(batch_x)
        # if use_gpu:
        #     encoded = encoded.detach().cpu().numpy()
        encoded = encoded.detach().numpy()
        batch_y = np.array(batch_y)
        batch_y = np.expand_dims(batch_y, axis=1)
        X = np.vstack([X, encoded])
        y = np.vstack([y, batch_y])
    X = X[1:]
    y = y[1:]
    y = y.ravel()
    return X, y


if __name__ == '__main__':
    # Available options
    dataset_name = ['YALEB', 'MNIST', 'FMNIST', 'olivetti']
    model_name = ['QAE', 'AE', 'QCAE', 'CAE', 'QSAE', 'SAE']
    # init parameters
    chosen_dataset = dataset_name[1]  # 0-yaleb,1-MNIST,2-FMNIST,3-olivetti
    chosen_model = model_name[3]  # 0-QAE,1-AE,2-QCAE,3-CAE,4-QSAE,5-SAE
    epoch = 10
    lr = 0.001
    bs = 128
    slr = 0.1  # In AE models, sub_learning_rate is not used, but we recommend to set 0.1 for a mark.
    RHO = 0.01  # Sparsity parameter for SAE and QSAE, we set it 0.01 in all experments
    p = 0.01  # Penalty Parameters, when training a CAE, QCAE, SAE, or QSAE, it denotes \lambda or \beta
    reduced_dimension = 32  # you need to change the file in AEModel (AE.py, or QAE.py) at the same time
    seed = 42

    random_seed(seed)
    test_rmse_list = []
    test_acc_list = []
    # region Unsupervised train and test
    data_set = DataSetLoader(chosen_dataset)
    train_dataset, vaild_dataset, test_dataset = data_set.data_set_loader()
    test_loader = DataLoader(test_dataset, batch_size=1)
    train_dataset.extend(vaild_dataset)
    data_loaders = DataLoader(train_dataset, batch_size=bs)
    if chosen_model in ('QAE', 'CAE', 'AE', 'QCAE'):
        model_name_path = chosen_model + '_4_nonoise_' + str(reduced_dimension) + '_' + chosen_dataset + '_lr%f_bs_%d_slr_%f_lambd_%f_seed_%d' % (
                              lr, bs, slr, p, seed)
        print(model_name_path)
    else:
        model_name_path = chosen_model + '_4_nonoise_' + str(reduced_dimension) + '_' + chosen_dataset + '_lr%f_bs_%d_slr_%f_beta_%f_seed_%d' % (
                              lr, bs, slr, p, seed)
        print(model_name_path)

    loss_list1 = train(lr, bs, epoch, data_loaders, model_name_path, chosen_dataset, slr, p)
    if np.any(np.isnan(loss_list1)):
        test_rmse_list.append(100000)
        test_acc_list.append(100000)
    else:
        _, avg_loss = inference(test_loader, model_name_path, chosen_dataset, p)
        test_rmse_list.append(avg_loss)
    # endregion

    # region Unsupervised dimension reduction and classification
    # Dimension reduction
    X_new, y = autoencoder(model_name_path, chosen_dataset, data_loaders, p, reduced_dimension)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_new = min_max_scaler.fit_transform(X_new)

    # Classifier training
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2', C=0.1, max_iter=500, solver='sag')
    model.fit(X_new, y)

    # test
    X_test, y_test = autoencoder(model_name_path, chosen_dataset, test_loader, p, reduced_dimension)
    X_test = min_max_scaler.fit_transform(X_test)
    y_pred = model.predict(X_test)
    test_acc_list.append(accuracy_score(y_pred, y_test))
    # endregion

    # save results
    rmse = np.array(test_rmse_list).reshape(-1, 1)
    acc = np.array(test_acc_list).reshape(-1, 1)
    results = np.concatenate((rmse, acc), axis=1)
    results_df = pd.DataFrame(results, columns=['rmse', 'acc'])

    save_path_name = chosen_model + '_4_nonoise_' + str(
        reduced_dimension) + '_' + chosen_dataset + '_lr%f_bs_%d_slr_%f_lambd_%f' % (
                         lr, bs, slr, p) + '.csv'
    if not os.path.exists('results'):
        os.mkdir('results')
    save_path = os.path.join('results', save_path_name)
    results_df.to_csv(save_path, index=False)
