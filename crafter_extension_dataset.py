from torch.utils.data import Dataset
from crafter_extension_utils import choose, interpolate_simple
import torch

import numpy as np


class CrafterCriticDataset(Dataset):

    def __init__(self, X, Y, oversample=False, dataset_size=50000, interpolate_real=False):
        if not interpolate_real:
            Y = torch.tensor[Y]
            if oversample:
                X_positive = X[Y == 1]
                X_negative = X[Y == 0]

                del X

                no_positive = len(X_positive)
                no_negative = len(X_negative)

                if no_negative > dataset_size / 2:
                    no_negative = int(dataset_size / 2)
                    X_negative = choose(X_negative, no_negative, replace=False)

                X_positive = choose(X_positive, no_negative)

                X = torch.vstack((X_positive, X_negative))
                Y = torch.cat((torch.ones(len(X_positive)), torch.zeros(len(X_negative))))

            self.X = X
            self.Y = Y
        else:
            if oversample:
                # get interpolated reward
                self.Y = interpolate_simple(Y)
                self.Y=torch.tensor(Y)

                # get indices by reward value

                ix_low = np.where(Y<=0.3)[0]
                ix_high = np.where(Y>=0.7)[0]
                ix_med = np.where((Y>0.3) & (Y<0.7))[0]



                # get 1/3 per sample category
                low_samples_ix = np.random.choice(ix_low,int(dataset_size/3))
                med_samples_ix = np.random.choice(ix_med,int(dataset_size/3))
                high_samples_ix = np.random.choice(ix_high,int(dataset_size/3))



                samples_ix = np.concatenate((low_samples_ix,med_samples_ix,high_samples_ix))

                self.X = X[samples_ix]
                self.Y=Y[samples_ix]


                del X, Y


            else:
                self.X = X
                self.Y = interpolate_simple(Y)
                self.Y = torch.tensor(Y)

        assert len(self.X) == len(self.Y), '?! error X shape does not match Y shape'

        self.len = len(self.X)



    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])
