import os
import sys

import numpy as np
import torch
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import TensorDataset, Dataset
from torchvision import transforms, datasets
from torchvision.datasets import mnist, cifar, imagenet
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
try:
    from PIL import Image
except ImportError:
    import Image

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

class DataSetLoader(object):
    def __init__(self, name: str):
        self.name = name

    def data_set_loader(self):


        if self.name == 'FMNIST':
            transform = transforms.Compose([transforms.ToTensor()])
            train_dataset = mnist.FashionMNIST(root='./data', train=True, transform=transform,
                                                     download=True)
            test_dataset = mnist.FashionMNIST(root='./data', train=False, transform=transform,
                                              download=True)
        if self.name == 'MNIST':
            transform = transforms.Compose([transforms.ToTensor()])
            train_dataset = mnist.MNIST(root='./data', train=True, transform=transform,
                                                     download=True)
            test_dataset = mnist.MNIST(root='./data', train=False, transform=transform,
                                              download=True)

        if self.name == 'YALEB':
            path_images = 'data/CroppedYale/'
            X_original, y_original = read_images(path_images)
            X, y = np.array(X_original) / 255., np.array(y_original)
            train_dataset = CustomTensorDataset(
                tensors=(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
                transform=None)
            train_dataset, test_dataset = train_test_split(train_dataset, test_size=0.5, random_state=42)

        if self.name == 'olivetti':
            X, y = fetch_olivetti_faces(data_home='/data/', return_X_y=True, shuffle=True,
                                                random_state=42)
            train_dataset = TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float))
            train_dataset, test_dataset = train_test_split(train_dataset, test_size=0.5, random_state=42)
        train_dataset, vaild_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
        return train_dataset, vaild_dataset, test_dataset


def read_images(path, sz=None, sz0=168, sz1=192):
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given) and check that it's the good size
                    if ((im.size[0] == sz0) & (im.size[1] == sz1)):
                        if (sz is not None):
                            im = im.resize(sz, Image.NEAREST)
                        X.append(np.asarray(im, dtype=np.uint8))
                        y.append(c)
                except IOError:
                    pass
                except:
                    # print("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c + 1

    return [X, y]