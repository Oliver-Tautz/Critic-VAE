# sources:
# basic structure: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
# how to load minerl data: https://github.com/KarolisRam/MineRL2021-Research-baselines/blob/main/standalone/Behavioural_cloning.py#L38

import torch;

torch.manual_seed(0)
from torch import Tensor, t
import torch.utils
import torch.distributions
import numpy as np
import os
from PIL import Image
from time import time
import argparse
import os
import pickle
import statistics

from logger import Logger
from vae_parameters import *
from vae_nets import *
from vae_utility import *

from tqdm import trange
from crafter_extension_utils import load_crafter_data, crafter_image_evaluate, train_on_crafter
from crafter_extension_vae import CrafterVariationalAutoencoder

parser = argparse.ArgumentParser()
parser.add_argument('-train-crafter', action='store_true')  # train on crafter
parser.add_argument('-train-crafter-real', action='store_true')  # train on crafter
parser.add_argument('-eval-crafter', action='store_true')  # train on crafter
parser.add_argument('-train', action='store_true')  # train
parser.add_argument('-crafter-windowsize', type=int, default=None)
parser.add_argument('-crafter-dataset-size', type=int, default=10000)
parser.add_argument('-inject', action='store_true')  # show recons of samples
parser.add_argument('-dataset', action='store_true')  # save recons as dataset
parser.add_argument('-second', action='store_true')  # train second VAE
parser.add_argument('-evalsecond', action='store_true')
parser.add_argument('-video', action='store_true')
parser.add_argument('-thresh', action='store_true')  # test threshold
parser.add_argument('-crafter-epochs', type=int, default=400)
args = parser.parse_args()

epochs = args.crafter_epochs


def train(autoencoder, dset, logger=None):
    frames, gt_frames = load_textured_minerl()
    dset = np.stack(dset).squeeze()
    opt = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    num_samples = dset.shape[0]

    # Start training
    for ep in trange(epochs, desc='train_epochs'):  # change
        epoch_indices = np.arange(num_samples)
        np.random.shuffle(epoch_indices)

        for batch_i in trange(0, num_samples, batch_size, desc='train_batches'):
            # NOTE: this will cut off incomplete batches from end of the random indices
            batch_indices = epoch_indices[batch_i:batch_i + batch_size]
            images = dset[batch_indices]
            images = Tensor(images).to(device)

            preds = critic.evaluate(images)
            opt.zero_grad()

            out = autoencoder(images, preds)

            losses = autoencoder.vae_loss(out[0], out[1], out[2], out[3])

            loss = losses['total_loss']

            print('loss:', loss)
            loss.backward()
            opt.step()

            if batch_i % log_n == 0:
                print(f'    ep:{ep}, imgs:{num_samples * ep + (batch_i + 1)}', end='\r')

                if logger is not None:
                    log_info(losses, logger, batch_i, ep, num_samples)

    return autoencoder


def image_evaluate(autoencoder, critic):
    print('evaluating source images...')
    folder = os.listdir(SOURCE_IMAGES_PATH)

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    if args.inject and not os.path.exists(INJECT_PATH):
        os.mkdir(INJECT_PATH)

    imgs = []
    diff_max_values = []
    for i, img_file in enumerate(folder):
        ### LOAD IMAGES AND PREPROCESS ###
        orig_img = Image.open(f'{SOURCE_IMAGES_PATH}/{img_file}')
        img_array = adjust_values(orig_img)
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW for critic
        img_array = img_array[np.newaxis, ...]  # add batch_size = 1 to make it BCHW
        img_tensor = Tensor(img_array).to(device)

        pred = critic.evaluate(img_tensor)

        if args.inject:
            img = get_injected_img(autoencoder, img_tensor, pred[0])
            img.save(f'{INJECT_PATH}image-{i:03d}.png', format="png")
        else:
            ro, rz, diff, max_value = get_diff_image(autoencoder, img_tensor, pred[0])
            imgs.append([img_tensor, ro, rz, diff, pred[0]])
            diff_max_values.append(max_value)

    if not args.inject:
        mean_max = statistics.mean(diff_max_values)
        diff_factor = 1 / mean_max if mean_max != 0 else 0

        for i, img in enumerate(imgs):
            diff_img = prepare_diff(img[3], diff_factor, mean_max)
            diff_img = (diff_img * 255).astype(np.uint8)
            diff_img = Image.fromarray(diff_img)
            save_img = get_final_frame(img[0], img[1], img[2], diff_img, img[4])

            save_img.save(f'{SAVE_PATH}/image-{i:03d}.png', format="png")


vae = VariationalAutoencoder().to(device)  # GPU

if args.video:
    # get images from regular vae
    load_vae_network(vae)
    critic = load_critic(CRITIC_PATH)
    frames, gt_frames = load_textured_minerl()  # gt = ground truth of tree trunk

    if args.thresh:
        print('testing thresholds (thr):')
        for t in range(0, 130, 10):
            vae_frames, thr_iou, crf_iou = eval_textured_frames(frames, vae, critic, gt_frames, t=t)
            print(f'thr={t}, thr_iou={thr_iou}, crf_iou={crf_iou}')
    else:
        vae_frames, thr_iou, crf_iou = eval_textured_frames(frames, vae, critic, gt_frames)
        print(f'thr_iou={thr_iou}')
        print(f'crf_iou={crf_iou}')

    create_video(vae_frames)
elif args.dataset:
    load_vae_network(vae)
    critic = load_critic(CRITIC_PATH)
    dset = load_minerl_data(critic, recon_dset=True, vae=vae)

    with open(SAVE_DATASET_PATH, 'wb') as file:
        pickle.dump(dset, file)
elif args.second:

    print('training second vae...')
    critic = load_critic(CRITIC_PATH)

    print('preparing dataset...')
    with open(SAVE_DATASET_PATH, 'rb') as file:
        recon_dset = pickle.load(file)

    # logger = Logger('./logs/vae' + str(time())[-5::])
    vae = train(vae, recon_dset)

    torch.save(vae.encoder.state_dict(), SECOND_ENCODER_PATH)
    torch.save(vae.decoder.state_dict(), SECOND_DECODER_PATH)
elif args.evalsecond:
    critic = load_critic(CRITIC_PATH)
    load_vae_network(vae, second_vae=True)
    image_evaluate(vae, critic)




else:  # REGULAR VAE

    if args.train:
        critic = load_critic(CRITIC_PATH)
        logger = Logger('./logs/vae' + str(time())[-5::])
        dset = load_minerl_data(critic)
        vae = train(vae, dset, logger=logger)

        torch.save(vae.encoder.state_dict(), ENCODER_PATH)
        torch.save(vae.decoder.state_dict(), DECODER_PATH)

    elif args.train_crafter or args.train_crafter_real:
        print("Training on crafter dataset :)")
        vae = CrafterVariationalAutoencoder().to(device)

        if args.train_crafter_real:
            critic = load_critic(CRAFTER_CRITIC_PATH_REAL, crafter=True)
        else:
            critic = load_critic(CRAFTER_CRITIC_PATH, crafter=True)
        logger = Logger('./logs/vae' + str(time())[-5::])

        # print(args.crafter_dataset_size)
        dset, dset_test = load_crafter_data(critic, dataset_size=args.crafter_dataset_size,
                                            windowsize=args.crafter_windowsize)
        # print(dset.shape)

        vae = train_on_crafter(vae, critic, dset, logger=logger, epochs=epochs, test_data=dset_test)

        torch.save(vae.encoder.state_dict(), ENCODER_PATH)
        torch.save(vae.decoder.state_dict(), DECODER_PATH)

        crafter_image_evaluate(vae, critic, inject=args.inject,
                               crafter_train_povs=torch.permute(dset, (0, 2, 3, 1)) * 255, crafter_test_povs=torch.permute(dset_test, (0, 2, 3, 1)) * 255)

    elif args.eval_crafter:

        critic = load_critic(CRAFTER_CRITIC_PATH, crafter=True)
        vae = CrafterVariationalAutoencoder().to(device)
        load_vae_network(vae, second_vae=False)

        dset, dset_test = load_crafter_data(critic, dataset_size=args.crafter_dataset_size,
                                            windowsize=args.crafter_windowsize)
        crafter_image_evaluate(vae, critic, inject=args.inject,
                               crafter_train_povs=torch.permute(dset, (0, 2, 3, 1)) * 255, crafter_test_povs=dset_test)



    else:  # EVALUATE
        critic = load_critic(CRITIC_PATH)
        load_vae_network(vae)

        image_evaluate(vae, critic)
