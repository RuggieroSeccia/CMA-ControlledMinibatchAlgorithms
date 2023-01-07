import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

class Dataset:
    """
    Class to define the data set and its attributes
    """

    def __str__(self):

        description = 'Dataset: {}.\nNumber of samples: {}. Number of variables: {}.'.format(self.csv, self.x.shape[0], self.x.shape[1])
        description = description + '\nTraining size: {}. Test size: {}.'.format(self.x_train.shape[1], self.x_test.shape[1])
        return description

    def __init__(self,
                 csv_file: str,
                 scaling: bool = False,
                 perc_test: float = 0.25):
        """
        :param csv_file: path to the dataset to read
        :param scaling: if True/1 data are scaled
        :param perc_test: percentage of data used for testing
        """

        try:
            self.dataset = pd.read_csv(csv_file)
        except FileNotFoundError as e:
            print("data were not detected correctly. Check your cwd and the location of your dataset folder")
            print(f"cwd: {os.getcwd()}")
            raise e
        self.csv = csv_file.split('/')[-1]
        self.dataset = self.dataset.sample(frac=1, random_state=0)  # in this way we reshuffle the samples

        if scaling == 1:
            scaler = MinMaxScaler()
            self.dataset = scaler.fit_transform(self.dataset)

        # add the bias at the first layer
        self.dataset = np.hstack((np.ones((self.dataset.shape[0], 1)), self.dataset))

        # extract train and test
        self.x = self.dataset[:, :-1]
        self.y = self.dataset[:, -1].reshape(-1, 1)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=perc_test, random_state=1994)

        self.x_train, self.x_test = self.x_train.T, self.x_test.T

        self.P = self.x_train.shape[1]
        self.n = self.x_train.shape[0]
        self.idx = None
        self.x_train_mb = None
        self.y_train_mb = None

    def minibatch(self, first: int, last: int):
        """Extract a minibatch of observations from the starting dataset"""
        if last > self.P:
            last = self.P
        self.x_train_mb = self.x_train[:, first:last]
        self.y_train_mb = self.y_train[first:last, 0].reshape(last - first, 1)

    def get_idx(self):
        '''
        define the order in which the sample will be considered within the minibatches
        '''
        self.idx = np.arange(self.x_train.shape[1])
        np.random.shuffle(self.idx)
