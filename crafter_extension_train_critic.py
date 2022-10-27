from crafter_extension_utils import collect_data
from crafter_extension_critic_model import Critic
from pathlib import Path
import pandas as pd
import multiprocessing
import torch
import os

replay_dir = Path('dataset')
savepath = Path('crafter_models')
# use multiple cores with torch
ncpu = multiprocessing.cpu_count()
torch.set_num_threads(ncpu)

# colab can handle ~45000-50000
dataset_size = 45000

epochs = 2

# push this higher to be faster, but works well with 32
batchsize = 32
batch_size = batchsize

windowsize = 20

# get data
X, Y, _ = collect_data(replay_dir, download=True, windowsize=windowsize)
X = torch.tensor(X).permute(0, 3, 1, 2) / 255
X = X[:, :, 0:49]

# train critic
critic = Critic()
history = critic.fit_on_crafter(X, Y, batch_size=batchsize, epochs=epochs, dataset_size=dataset_size)

history['train_acc'] = [x['accuracy'] for x in history['train_acc']]
history['val_acc'] = [x['accuracy'] for x in history['val_acc']]

# save history
pd.DataFrame(history).to_csv('crafter_critic_training_log.csv')

# save model
modelname = f"critic-windowsize={windowsize}-batch_size={batch_size}-dataset_size={dataset_size}-epochs={epochs}"

os.makedirs(savepath / modelname, exist_ok=True)
torch.save(critic.state_dict(),
           savepath / modelname / 'critic.pt')
