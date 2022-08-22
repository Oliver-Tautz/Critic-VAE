from crafter_extension_utils import collect_data
from pathlib import Path
from crafter_extension_critic_model import Critic
import torch
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from vae_parameters import *
from crafter_extension_utils import load_crafter_pictures
import numpy as np

ncpu = multiprocessing.cpu_count()

torch.set_num_threads(ncpu)
torch.set_num_interop_threads(ncpu)


dataset_size=5000


critic = Critic()
critic.load_state_dict(torch.load(CRAFTER_CRITIC_PATH,map_location=torch.device('cpu')))

pictures = load_crafter_pictures('dataset')
pictures = torch.tensor(pictures).permute(0,3,1,2) / 255

Y = critic.evaluate(pictures,100)

ix_low = np.where(Y <= 0.25)[0]
ix_high = np.where(Y >= 0.7)[0]
ix_med = np.where((Y > 0.25) & (Y < 0.7))[0]

# get 1/3 per sample category
low_samples_ix = np.random.choice(ix_low, int(dataset_size / 3))
med_samples_ix = np.random.choice(ix_med, int(dataset_size / 3))
high_samples_ix = np.random.choice(ix_high, int(dataset_size / 3))

#print(len(ix_low),len(ix_high),len(ix_med))

samples_ix = np.concatenate((low_samples_ix, med_samples_ix, high_samples_ix))

#print(samples_ix)

