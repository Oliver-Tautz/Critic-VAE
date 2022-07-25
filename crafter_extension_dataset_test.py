from crafter_extension_utils import collect_data
from pathlib import Path
from crafter_extension_model import NewCritic
import torch
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
ncpu = multiprocessing.cpu_count()

torch.set_num_threads(ncpu)
torch.set_num_interop_threads(ncpu)

replay_dir = Path('/home/olli/gits/Critic-VAE/dataset')
X,Y,_ = collect_data(replay_dir,download=False,interpolate_to_float=False)

X = torch.tensor(X).permute(0,3,1,2) / 255




critic = NewCritic()
history = critic.fit_on_crafter(X,Y,batch_size = 32,epochs=30,dataset_size=30000,real=True,lossF=nn.MSELoss(),oversample=False)
history['train_acc'] = [a['accuracy'] for a in history['train_acc']]
history['val_acc'] = [a['accuracy'] for a in history['val_acc']]
pd.DataFrame(history)[['train_loss','val_loss']].plot()
plt.show()