import os
import random
import re

import pandas as pd
import numpy as np
# from sklearn import preprocessing as pre
# from matplotlib import pyplot as plt
# plt.rc('font', family='AppleGothic')
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from utils import date_to_int
BASE="2023-01-01"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

set_seed(42)
train_df = pd.read_csv('./data/train/train.csv')

if not os.path.exists("store"):
    os.mkdir("store")
for store_menu, group in tqdm(list(train_df.groupby("영업장명_메뉴명"))):
    print(f"store/{store_menu.strip()}.npz")
    date = group["영업일자"].tolist()
    date = torch.tensor(list(map(lambda x : date_to_int(x, BASE), date)))
    freq = torch.tensor(group["매출수량"].tolist())

    # print(f"store/{store_menu.strip()}.npz", group[["영업일자", "매출수량"]])

    print(date, freq)
