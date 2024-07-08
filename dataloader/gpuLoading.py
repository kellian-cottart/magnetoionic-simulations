import torch
import numpy as np
import idx2numpy
from torchvision.transforms import v2


class GPUTensorDataset(torch.utils.data.Dataset):
    """ TensorDataset which has a data and a targets tensor and allows to_dataset

    Args:
        data (torch.tensor): Data tensor
        targets (torch.tensor): Targets tensor
        device (str, optional): Device to use. Defaults to "cuda:0".
        transform (torchvision.transforms.Compose, optional): Data augmentation transform. Defaults to None.
    """

    def __init__(self, data, targets, device="cuda:0"):
        self.data = data.to("cpu")
        self.targets = targets.to("cpu")
        self.device = device

    def __getitem__(self, index):
        """ Return a (data, target) pair """
        return self.data[index].to(self.device), self.targets[index].to(self.device)

    def __len__(self):
        """ Return the number of samples """
        return len(self.data)

    def shuffle(self):
        """ Shuffle the data and targets tensors """
        perm = torch.randperm(len(self.data), device="cpu")
        self.data = self.data[perm]
        self.targets = self.targets[perm]


class GPUDataLoader():
    """ DataLoader which has a data and a targets tensor and allows to_dataset

    Args:
        dataset (GPUTensorDataset): Dataset to load
        batch_size (int): Batch size
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last batch if it is not full. Defaults to True.
        transform (torchvision.transforms.Compose, optional): Data augmentation transform. Defaults to None.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, transform=None, device="cuda:0", test=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transform = transform
        self.device = device
        self.test = test

    def __iter__(self):
        """ Return an iterator over the dataset """
        self.index = 0
        if self.shuffle:
            self.perm = torch.randperm(len(self.dataset))
        else:
            self.perm = torch.arange(len(self.dataset))
        return self

    def __next__(self):
        """ Return the next batch """
        if self.index >= len(self.dataset):
            raise StopIteration
        if self.index + self.batch_size > len(self.dataset):
            if self.drop_last == False:
                indexes = self.perm[self.index:]
            raise StopIteration
        else:
            indexes = self.perm[self.index:self.index+self.batch_size]
        self.index += self.batch_size
        data, targets = self.dataset.data[indexes], self.dataset.targets[indexes]
        if self.transform is not None:
            data = self.transform(data)
        return data.to(self.device), targets.to(self.device)

    def __len__(self):
        """ Return the number of batches """
        return len(self.dataset)//self.batch_size


class GPULoading:
    """ Load local datasets on GPU

    Args:
        batch_size (int): Batch size
        padding (int, optional): Padding to add to the images. Defaults to 0.
        as_dataset (bool, optional): If True, returns a TensorDataset, else returns a DataLoader. Defaults to False.
    """

    def __init__(self, padding=0, device="cuda:0", as_dataset=False, *args, **kwargs):
        self.padding = padding
        self.device = device
        self.as_dataset = as_dataset

    def to_dataset(self, train_x, train_y, test_x, test_y):
        """ Create a DataLoader to load the data in batches

        Args:
            train_x (torch.tensor): Training data
            train_y (torch.tensor): Training labels
            test_x (torch.tensor): Testing data
            test_y (torch.tensor): Testing labels
            batch_size (int): Batch size

        Returns:
            DataLoader, DataLoader: Training and testing DataLoader

        """
        train_dataset = GPUTensorDataset(
            train_x, torch.Tensor(train_y).type(
                torch.LongTensor), device=self.device)
        test_dataset = GPUTensorDataset(test_x, torch.Tensor(test_y).type(
            torch.LongTensor), device=self.device)
        return train_dataset, test_dataset

    def normalization(self, train_x, test_x):
        """ Normalize the pixels in train_x and test_x using transform

        Args:
            train_x (np.array): Training data
            test_x (np.array): Testing data

        Returns:
            torch.tensor, torch.tensor: Normalized training and testing data
        """

        # Completely convert train_x and test_x to float torch tensors
        # division by 255 is only scaling from uint to float
        train_x = torch.from_numpy(train_x).float() / 255
        test_x = torch.from_numpy(test_x).float() / 255

        if len(train_x.size()) == 3:
            train_x = train_x.unsqueeze(1)
            test_x = test_x.unsqueeze(1)

        # Normalize the pixels to 0, 1
        transform = v2.Compose(
            [v2.ToImage(),
             v2.ToDtype(torch.float32, scale=True),
             v2.Normalize((0,), (1,), inplace=True),
             v2.Pad(self.padding, fill=0, padding_mode='constant'),
             ])

        return transform(train_x), transform(test_x)

    def mnist(self, path_train_x, path_train_y, path_test_x, path_test_y, *args, **kwargs):
        """ Load a local dataset on GPU corresponding either to MNIST or FashionMNIST

        Args:
            batch_size (int): Batch size
            path_train_x (str): Path to the training data
            path_train_y (str): Path to the training labels
            path_test_x (str): Path to the testing data
            path_test_y (str): Path to the testing labels
        """
        # load ubyte dataset
        train_x = idx2numpy.convert_from_file(
            path_train_x).astype(np.float32)
        train_y = idx2numpy.convert_from_file(
            path_train_y).astype(np.float32)
        test_x = idx2numpy.convert_from_file(
            path_test_x).astype(np.float32)
        test_y = idx2numpy.convert_from_file(
            path_test_y).astype(np.float32)
        # Normalize and pad the data
        train_x, test_x = self.normalization(train_x, test_x)
        return self.to_dataset(train_x, train_y, test_x, test_y)
