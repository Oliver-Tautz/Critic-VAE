import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # maybe cuda instead of cuda:0
#device = 'cpu'
### IMAGE DATA ###


CRAFTER_DATASET_SIZE=50000

CRAFTER_IMAGE_SHAPE = (64,48,3)
CRAFTER_USE_MMSSIM = True

w = CRAFTER_IMAGE_SHAPE[0] # original image width
h = CRAFTER_IMAGE_SHAPE[1] # original image height
ch = CRAFTER_IMAGE_SHAPE[2] # image channels

MAX_CHANNELS = 64#256



### TRAINING DATA ###
epochs = 4
batch_size = 256
lr = 0.00005
#k=5
k = 3 # kernel size

#p=2
p = 1 # padding
step = 1
bottleneck = MAX_CHANNELS*int(CRAFTER_IMAGE_SHAPE[1]/4)* int(CRAFTER_IMAGE_SHAPE[0]/4 )
latent_dim = 32 # fully-connected layer, from 4096 to 64 dim
kld_weight = 0.002 # note: https://github.com/AntixK/PyTorch-VAE/issues/11 OR https://github.com/AntixK/PyTorch-VAE/issues/35

total_images = 50000

log_n = batch_size * 30  # data is logged every "log_n"-step
inject_n = 6

### PATHS ###
ENCODER_PATH = 'saved-networks/vae_encoder.pt'
DECODER_PATH = 'saved-networks/vae_decoder.pt'

SOURCE_IMAGES_PATH = 'source-images/'
SAVE_PATH = 'images/'
INJECT_PATH = 'inject/'
VIDEO_PATH = 'videos/'
SAVE_DATASET_PATH = 'recon-dataset.pickle'
MINERL_EPISODE_PATH = 'minerl-episode/'

SECOND_ENCODER_PATH = 'vae2_encoder.pt'
SECOND_DECODER_PATH = 'vae2_decoder.pt'

CRITIC_PATH = 'saved-networks/critic-rewidx=1-cepochs=15-datamode=trunk-datasize=99999-shift=12-chfak=1-dropout=0.3.pt'
CRAFTER_CRITIC_PATH_REAL = f"crafter_models/critic-batch_size=32-dataset_size=45000-epochs=50_real/critic.pt"
#CRAFTER_CRITIC_PATH = f"crafter_models/critic-windowsize=25-batch_size=32-dataset_size=45000-epochs=400/critic.pt"
CRAFTER_CRITIC_PATH = f"crafter_models/critic-windowsize=20-batch_size=32-dataset_size=45000-epochs=50/critic.pt"
SECOND_CRITIC_PATH = 'saved-networks/critic-rewidx=1-cepochs=15-datamode=trunk-datasize=100000-shift=12-chfak=1-dropout=0.3.pt'

MINERL_DATA_ROOT_PATH = '/homes/lcicek/anaconda3/envs/vae/lib/python3.6/site-packages/minerl'




