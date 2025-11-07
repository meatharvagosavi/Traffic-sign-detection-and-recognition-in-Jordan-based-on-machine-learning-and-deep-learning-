import os
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def find_images_and_labels(root_dir, exts={'.jpg','.jpeg','.png','.bmp'}):
    """
    Expect folders:
      root_dir/
        class0/
          img1.jpg
          img2.jpg
        class1/
          imgA.png
    Returns: list of (path, label_index), class_names
    """
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
    paths = []
    labels = []
    for idx, cname in enumerate(class_names):
        cdir = os.path.join(root_dir, cname)
        for fname in os.listdir(cdir):
            if os.path.splitext(fname)[1].lower() in exts:
                paths.append(os.path.join(cdir, fname))
                labels.append(idx)
    return paths, labels, class_names

def train_val_test_split(paths, labels, val_size=0.15, test_size=0.15, stratify=True, seed=42):
    # First split out test
    if stratify:
        strat = labels
    else:
        strat = None
    p_train_val, p_test, l_train_val, l_test = train_test_split(paths, labels, test_size=test_size, random_state=seed, stratify=strat)
    # then split train and val
    if stratify:
        strat2 = l_train_val
    else:
        strat2 = None
    val_relative = val_size / (1 - test_size)
    p_train, p_val, l_train, l_val = train_test_split(p_train_val, l_train_val, test_size=val_relative, random_state=seed, stratify=strat2)
    return (p_train, l_train), (p_val, l_val), (p_test, l_test)

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
