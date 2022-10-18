import numpy as np
from pathlib import Path
import wget
import zipfile
import os
from tqdm import tqdm, trange
import pandas as pd
import torch
from  torch import nn
from vae_parameters import *
from PIL import Image
from torch import Tensor, t
import statistics
from vae_utility import adjust_values, get_injected_img, get_diff_image, prepare_diff,get_final_frame,log_info
from os.path import exists
from collections import OrderedDict
from os import makedirs

import multiprocessing
ncpu = multiprocessing.cpu_count()
import torch
import os


torch.set_num_threads(ncpu)
torch.set_num_interop_threads(ncpu)

def remove_inventory(povs):
    "remove rows 49-64 (inventory)"
    povs[:, :, 49:, ] = 0
    return povs


def train_on_crafter(autoencoder,critic, dset, logger=None):
    #frames, gt_frames = load_textured_minerl()
    dset = np.stack(dset).squeeze()
    opt = torch.optim.AdamW(autoencoder.parameters())
    num_samples = dset.shape[0]

    # Start training

    training_data = {'total_loss':[],
                     'recon_loss':[],
                     'KLD':[]}

    for ep in trange(epochs, desc='train_epochs'):  # change
        epoch_indices = np.arange(num_samples)
        np.random.shuffle(epoch_indices)

        epoch_data = {'total_loss':[],
                     'recon_loss':[],
                     'KLD':[]}

        for batch_i in trange(0, num_samples, batch_size, desc='train_batches'):
            # NOTE: this will cut off incomplete batches from end of the random indices
            batch_indices = epoch_indices[batch_i:batch_i + batch_size]
            images = dset[batch_indices]


            images = Tensor(images).to(device)



            preds = critic.evaluate(images)



            opt.zero_grad()


            #print('preds:', preds.shape)
            #print('preds:', preds.shape)

            out = autoencoder(images, preds)


            #print(out[0].shape,out[1].shape)

            losses = autoencoder.vae_loss(out[0], out[1], out[2], out[3])



            epoch_data['total_loss'].append(losses['total_loss'].cpu().detach())
            epoch_data['recon_loss'].append(losses['recon_loss'].cpu().detach())
            epoch_data['KLD'].append(losses['KLD'].cpu().detach())

            loss = losses['total_loss']
            loss.backward()
            opt.step()


            if batch_i % log_n == 0:
                print(f'    ep:{ep}, imgs:{num_samples * ep + (batch_i + 1)}', end='\r')

                if logger is not None:
                    log_info(losses, logger, batch_i, ep, num_samples)

        training_data['total_loss'].append(np.mean(epoch_data['total_loss']))
        training_data['recon_loss'].append(np.mean(epoch_data['recon_loss']))
        training_data['KLD'].append(np.mean(epoch_data['KLD']))
        pd.DataFrame(training_data).to_csv('log.csv')
        print(epoch_data)


    return autoencoder


def choose(X, no_choices, replace=True):
    choices = np.array(len(X))

    choices = np.random.choice(choices, no_choices,replace=replace)
    return X[choices]


def plot_side_by_side(filename,in_im,out_im):
    import matplotlib.pyplot as plt

    in_im = in_im.detach().numpy()
    out_im = out_im.detach().numpy()


    in_im = in_im.squeeze().transpose(1,2,0)*255
    out_im = out_im.squeeze().transpose(1,2,0)*255
    f, (ax1, ax2) = plt.subplots(1,2)

    ax1.imshow(in_im.astype(np.uint8))
    ax2.imshow(out_im.astype(np.uint8))
    ax1.axis('off')
    ax2.axis('off')
    plt.savefig(filename,bbox_inches='tight')
    plt.close(f)
    del f

def crafter_image_evaluate(autoencoder, critic,crafter_povs=None,inject=False,no_samples=1000,remove_inv_for_vae=True,windowsize=None):
    """
    Batch processing could really speed this up i think :O

    :param autoencoder:
    :param critic:
    :param inject:
    :return:
    """

    print('evaluating source images...')

    if crafter_povs==None:
        crafter_povs = load_crafter_pictures('dataset',download=False,windowsize=windowsize)[0:100]
        print(crafter_povs.shape)

    #print(crafter_povs.shape)


    if no_samples and no_samples<len(crafter_povs):
        crafter_povs = choose(crafter_povs,no_choices=no_samples,replace=False)

    imgs = []

    for i,crafter_pov in enumerate(crafter_povs):

        crafter_pov = crafter_pov.permute(2, 0, 1)
        crafter_pov = crafter_pov.unsqueeze(0)/255
        crafter_pov = crafter_pov.to(device)


        x, mu, logvar, recon= autoencoder(crafter_pov,critic(crafter_pov))
        os.makedirs('my_images',exist_ok=True)
        plot_side_by_side(f'my_images/pov_recon_{i}.jpg',crafter_pov, recon)

    diff_max_values = []
    for i, crafter_pov in tqdm(enumerate(crafter_povs),desc='evaluate_dataset_step1',total=len(crafter_povs)):
        ### LOAD IMAGES AND PREPROCESS ###
        orig_img = crafter_pov
        img_array = adjust_values(orig_img)
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW for critic

        img_array = img_array[np.newaxis, ...]  # add batch_size = 1 to make it BCHW
        img_array = img_array[:, :, 0:48]
        img_tensor = Tensor(img_array).to(device)

        pred = critic.evaluate(img_tensor)


        if inject:
            img = get_injected_img(autoencoder, img_tensor, pred[0])
            os.makedirs(INJECT_PATH,exist_ok=True)
            img.save(f'{INJECT_PATH}image-{i:03d}.png', format="png")
        else:
            ro, rz, diff, max_value = get_diff_image(autoencoder, img_tensor, pred[0])
            imgs.append([img_tensor, ro, rz, diff, pred[0]])
            diff_max_values.append(max_value)

    if not inject:
        mean_max = statistics.mean(diff_max_values)
        diff_factor = 1 / mean_max if mean_max != 0 else 0

        for i, img in tqdm(enumerate(imgs),desc='evaluate_dataset_step2',total=len(imgs)):
            diff_img = prepare_diff(img[3], diff_factor, mean_max)
            diff_img = (diff_img * 255).astype(np.uint8)
            diff_img = Image.fromarray(diff_img)
            save_img = get_final_frame(img[0], img[1], img[2], diff_img, img[4])
            os.makedirs(SAVE_PATH,exist_ok=True)
            save_img.save(f'{SAVE_PATH}/image-{i:03d}.png', format="png")


def load_crafter_data(critic, recon_dset=False, vae=None,dataset_size=45000,windowsize=None):
    print("loading minerl-data...")

    ### Initialize mineRL dataset ###
    # os.environ['MINERL_DATA_ROOT'] = MINERL_DATA_ROOT_PATH
    pictures = load_crafter_pictures('dataset',windowsize=windowsize)

    pictures = torch.tensor(pictures).permute(0, 3, 1, 2) / 255
    pictures = pictures[:, :, 0:48]

    critic_values = nn.Sigmoid()(critic.evaluate(pictures,batchsize=100))
    critic_values = critic_values.cpu()

    #print(critic_values)

    ix_low = np.where(critic_values <= 0.25)[0]
    ix_high = np.where(critic_values >= 0.7)[0]
    ix_med = np.where((critic_values > 0.25) & (critic_values < 0.7))[0]
    print(len(ix_low),len(ix_med),len(ix_high))

    # get 1/3 per sample category
    low_samples_ix = np.random.choice(ix_low, int(dataset_size / 3))
    med_samples_ix = np.random.choice(ix_med, int(dataset_size / 3))
    high_samples_ix = np.random.choice(ix_high, int(dataset_size / 3))


    samples_ix = np.concatenate((low_samples_ix, med_samples_ix, high_samples_ix))

    dset=pictures[samples_ix]


    return dset

def load_crafter_pictures(replay_dir, target_inventory_item='inventory_wood', download=True, interpolate_to_float=False,windowsize=None):
    X, _ ,_ = collect_data(replay_dir, target_inventory_item, download, interpolate_to_float,windowsize)
    return X

def collect_data(replay_dir='./dataset', target_inventory_item='inventory_wood', download=True, interpolate_to_float=False,windowsize=None):
    # download and extract dataset
    from os import makedirs

    replay_dir = Path(os.path.expanduser(replay_dir))
    makedirs(replay_dir,exist_ok=True)

    dataset_folder = 'dataset'
    if windowsize:
        dataset_folder += f"_windowsize={windowsize}"

    if download:
        if not exists(replay_dir / dataset_folder/ '1CKFwmfLeb5MzlgFRaIF7M.npz') :
            print("Downloading raw replay dataset ...")
            human_replay_dataset_url = "https://archive.org/download/crafter_human_dataset/dataset.zip"
            print('downloaded ', wget.download(human_replay_dataset_url, out=str((replay_dir / 'dataset.zip').resolve())))
            print("\nUnzipping ...")
            with zipfile.ZipFile(replay_dir / 'dataset.zip', 'r') as zip_ref:
                zip_ref.extractall(replay_dir)
            os.remove(replay_dir / 'dataset.zip')

            if windowsize:
                save_dataset_with_windowsize(replay_dir,windowsize)

        else:
            print('Dataset already downloaded :)')

    # convert to dictionaries
    replay_list = []

    for replay_name in os.listdir(replay_dir / dataset_folder):
        if replay_name.endswith('.npz'):
            replay = np.load(replay_dir / dataset_folder / replay_name, allow_pickle=True)
            # replay.keys()
            replay_dict = dict()
            # for key in replay.keys():
            #   print(key,replay[key])
            for key in (replay.files):
                replay_dict[key] = replay[key]
            # print(replay_dict)
            replay_list.append(replay_dict)

    # convert to dataframes
    Xs = []
    Ys = []
    Is = []

    for i, replay_dict in enumerate(tqdm(replay_list, desc='loading episodes ...')):
        replay_list[i] = get_df(replay_dict)
        Xs.extend(([np.array(x) for x in replay_list[i]['image']]))
        Is.extend(list(range(len(replay_list[i]))))
        for j in range(len(replay_list[i])):
            current_replay = replay_list[i]
            current_inventory = current_replay[target_inventory_item][j]

            if j > 0:
                older_inventory = current_replay[target_inventory_item][j - 1]
                if current_inventory > older_inventory:
                    Ys.append(current_inventory - older_inventory)
                else:
                    Ys.append(0)
            else:
                Ys.append(current_inventory)

    if interpolate_to_float:
        Ys = interpolate_simple(np.array(Ys).astype(float))
    else:
        Ys = np.array(Ys)

    return np.array(Xs), Ys.astype(float), np.array(Is)

def interpolate_simple(Y_,windowsize=5):
    i=0


    before = None
    current = None

    while i < len(Y_):
        if Y_[i] == 1:
            current = i
            if not before:
                sublist_len = len(Y_[max(0,current-windowsize+1):current+1])
                Y_[max(0,current-windowsize+1):current+1] = linear_interpolate((sublist_len))

            elif current - before <= windowsize:
                sublist_len = len(Y_[before+1:current+1])
                Y_[before+1:current+1] = linear_interpolate((sublist_len))
            else:
                sublist_len = len(Y_[current-windowsize+1:current+1])
                Y_[current-windowsize+1:current+1] = linear_interpolate((sublist_len))
            before = current

        i+=1
    return Y_



def get_df(replay_dict,col_list = ['image', 'action', 'reward', 'done', 'discount', 'semantic',
       'player_pos', 'inventory_health', 'inventory_food', 'inventory_drink',
       'inventory_energy', 'inventory_sapling', 'inventory_wood',
       'inventory_stone', 'inventory_coal', 'inventory_iron',
       'inventory_diamond', 'inventory_wood_pickaxe',
       'inventory_stone_pickaxe', 'inventory_iron_pickaxe',
       'inventory_wood_sword', 'inventory_stone_sword', 'inventory_iron_sword',
]):
    df = pd.DataFrame()

    for key in replay_dict.keys():
        if len(replay_dict[key].shape) > 1 :
            df[key]=list(replay_dict[key])
        else:
            df[key] = replay_dict[key]
    return df[col_list]

def linear_interpolate(l):
    return [(1*i/l) for i in range((l+1))][1:]


def save_dataset_with_windowsize(replay_dir,windowsize):
    for replay_name in os.listdir(replay_dir / 'dataset'):
        if replay_name.endswith('.npz'):
            replay = np.load(replay_dir / 'dataset' / replay_name, allow_pickle=True)
            # replay.keys()
            replay_dict = OrderedDict(replay)

            ### only use pics close to the reward

            images = replay_dict['image']
            wood = replay_dict['inventory_wood']

            ## gather indices where a new woodlog is gathered
            reward_ix = []
            current_wood = 0

            for i, w in enumerate(wood):
                if w > current_wood:
                    reward_ix.append(i)
                    current_wood = w
                if w < current_wood:
                    current_wood = w

            ## get cleaned data
            for key in replay_dict.keys():

                cleaned = [replay_dict[key][max(0, i - windowsize):i] for i in reward_ix]

                if len(cleaned[0].shape) > 1:
                    replay_dict[key] = np.vstack([replay_dict[key][max(0, i - windowsize):i] for i in reward_ix])
                else:
                    replay_dict[key] = np.concatenate([replay_dict[key][max(0, i - windowsize):i] for i in reward_ix])

            ## save to new folders
            makedirs(replay_dir / f'dataset_windowsize={windowsize}', exist_ok=True)
            np.savez_compressed(replay_dir / f'dataset_windowsize={windowsize}' / replay_name, **replay_dict, )