import numpy as np
from pathlib import Path
import wget
import zipfile
import os
from tqdm import tqdm
import pandas as pd


def choose(X, no_choices, replace=True):
    choices = np.array(len(X))

    choices = np.random.choice(choices, no_choices)
    return X[choices]


def load_crafter_pictures(replay_dir, target_inventory_item='inventory_wood', download=True, interpolate_to_float=False):
    X, _ ,_ = collect_data(replay_dir, target_inventory_item, download, interpolate_to_float)
    return X

def collect_data(replay_dir, target_inventory_item='inventory_wood', download=True, interpolate_to_float=False):
    # download and extract dataset
    print(replay_dir)
    replay_dir = Path(os.path.expanduser(replay_dir))

    if download:
        print("Downloading raw replay dataset ...")
        human_replay_dataset_url = "https://archive.org/download/crafter_human_dataset/dataset.zip"
        print('downloaded ', wget.download(human_replay_dataset_url, out=str((replay_dir / 'dataset.zip').resolve())))
        print("\nUnzipping ...")
        with zipfile.ZipFile(replay_dir / 'dataset.zip', 'r') as zip_ref:
            zip_ref.extractall(replay_dir)
        os.remove(replay_dir / 'dataset.zip')

    # convert to dictionaries
    replay_list = []

    for replay_name in os.listdir(replay_dir / 'dataset'):
        if replay_name.endswith('.npz'):
            replay = np.load(replay_dir / 'dataset' / replay_name, allow_pickle=True)
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

def interpolate_simple(Y_,windowsize=50):
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
