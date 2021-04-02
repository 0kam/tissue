from glob import glob
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision.datasets import DatasetFolder
from torchvision.transforms.functional import to_tensor
import pandas as pd
import json
import cv2
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import shutil
import pylab as pl
from matplotlib import pyplot as plt

from matplotlib.font_manager import FontProperties
font_path = "/usr/share/fonts/truetype/migmix/migmix-1p-regular.ttf"
font_prop = FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

# Set up patches
def read_sses(label_dir, image_size):
    """
    Read multiple json files created with Semantic Segmentation Editor
    Parameters
    ----------
    label_dir : str
        A path for a directory which stores json files.
    image_size : tuple of int
        (width, height) of the specified image

    Returns
    -------
    image : np.array
        A masked image.
    labels : pd.DataFrame
        
    """
    objects = []
    for p in glob(label_dir + "/*"):
        with open(p) as js:
            obj = json.load(js)["objects"]
            obj = pd.json_normalize(obj)
            objects.append(obj)
    objects = pd.concat(objects)
    polygons = list(objects["polygon"])
    labels = objects["classIndex"]
    image = np.zeros([image_size[1], image_size[0], 3])
    for p, l in zip(polygons, labels):
        polygon = []
        for point in p:
            point = list(point.values())
            polygon.append(point)
        polygon = np.array(polygon).reshape(-1, 1, 2).astype("int32")
        image = cv2.fillPoly(image, [polygon], color = (l, l, l))
    labels = objects[["classIndex", "label"]].drop_duplicates()
    return image, labels


def set_patches(label_dir, image_dir, out_dir, kernel_size, batch_size=5000):
    """
    Setting up data folders for semi-supervised image segmentation of time-lapse photographs.
    
    Parameters
    ----------
    label_dir : str
        A directory that has json files created with Semantic Segmentation Editor.
    image_dir : str
        A directory that has images for training models.
    out_dir : str
        An output directory. Subdirectories "labelled" and "unlabelled" will be created.
    kernel_size : tuple of int
        A kernel size.
    batch_size : int
        A length of each .npy file created with this function. 
        This value must be smaller than pix_smallest / 2, where pix_smallest stands for the smallest pixel number of a label in teacher data.
    """
    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)
        os.makedirs(out_dir + "/labelled")
        os.makedirs(out_dir + "/unlabelled")
    image = Image.open(glob(image_dir + "/*")[0])
    data_name = Path(image_dir).stem
    mask, labels_list = read_sses(label_dir, image.size)
    mask = mask.astype(int)
    kw = int((kernel_size[0] - 1) / 2)
    kh = int((kernel_size[1] - 1) / 2)
    w, h = image.size

    tensors = {}
    labels_list = labels_list.append({"label":"unlabelled", "classIndex":0}, ignore_index = True)
    for label in labels_list["classIndex"].values:
        tensors[str(label)] = []
        if os.path.exists(out_dir + "/labelled/" + str(label)) is False:
            os.mkdir(out_dir + "/labelled/" + str(label))
    images = torch.stack([to_tensor(Image.open(f)) for f in glob(image_dir + "/*")], dim = 0)
    i = 0
    for v in tqdm(range(h)):
        for u in range(w):
            patch = images[:, :, max(0, v-kh):(min(h, v+kh)+1), max(0, u-kw):(min(w, u+kh)+1)]
            patch = F.interpolate(patch, [kernel_size[1], kernel_size[0]])

            patch = torch.reshape(patch, (patch.shape[0], -1))
            label = mask[v,u][0]
            tensors[str(label)].append(patch)

            for label in labels_list["classIndex"]:
                if len(tensors[str(label)]) == batch_size:
                    out_path = out_dir + "/labelled/" + str(label) + "/" + data_name + "_" + str(i) + ".npy"
                    np.save(out_path, np.stack(tensors[str(label)], axis=0))
                    tensors[str(label)] = []
                i += 1
    shutil.move(out_dir + "/labelled/0/", out_dir + "/unlabelled/0/")

# Data Loaders
def load_npy(path):
    ts = torch.tensor(np.load(path))
    return ts

class LabelledDS(Dataset):
    def __init__(self, patch_dir):
        self.dataset = DatasetFolder(patch_dir, load_npy, "npy")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        y = torch.tensor(y)
        y = y.repeat(1, x.shape[0]).view(-1)
        return x, y

class UnlabelledDS(Dataset):
    def __init__(self, patch_dir):
        self.dataset = DatasetFolder(patch_dir, load_npy, "npy")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x

def cf_labelled(batch):
    x, y = list(zip(*batch))
    x = torch.stack(x)
    y = torch.stack(y)
    x = x.view(-1, x.shape[2], x.shape[3])
    y = y.view(-1)
    return x, y

def cf_unlabelled(batch):
    x = list(batch)
    x = torch.stack(x)
    x = x.view(-1, x.shape[2], x.shape[3])
    return(x)

class DrawDS(Dataset):
    """
    A Dataset that returns time series patches of images with a given kernel_size.
    Attributes
    ----------
    image_dir : str
        An input data directory that has a set of time-series images. All images must be same size.
    kernel_size : tuple of int
        A tuple (width, height) of the kernel.
    """
    def __init__(self, image_dir, kernel_size):
        self.image_dir = image_dir
        self.kernel_size = kernel_size
        f = glob(self.image_dir + "/*")[0]
        with Image.open(f) as img:
            self.size = img.size
            self.data_length = img.width*img.height
        self.target_images = torch.stack([to_tensor(Image.open(f)) for f in glob(self.image_dir + "/*")], dim = 0)
    
    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        center = (idx % self.size[0], idx // self.size[0])
        kw = int((self.kernel_size[0]-1) / 2)
        kh = int((self.kernel_size[1]-1) / 2)
        left = max(center[0] - kw, 0)
        upper = max(center[1] - kh, 0)
        right = min(center[0] + kw, self.size[0])
        lower = min(center[1] + kh, self.size[1])
        patch = self.target_images[:,:,upper:lower+1,left:right+1]
        patch = F.interpolate(patch, [self.kernel_size[1], self.kernel_size[0]])
        patch = torch.reshape(patch, (patch.shape[0], -1))
        patch = patch
        return patch


# Drawing functions
def draw_teacher(out_path, label_dir, class_to_idx, image_size):
    img, labels = read_sses(label_dir, image_size)
    mask = img.copy()
    cmap = plt.get_cmap("tab20", len(labels))
    for label in labels["classIndex"]:
        l = class_to_idx[str(label)]
        img[mask == label] = l
    plt.imsave(out_path, img[:,:,0], cmap = cmap)
    img = cv2.imread(out_path)
    img[mask == 0] = 0
    cv2.imwrite(out_path, img)

def draw_legend(out_path, label_dir, class_to_idx):
    img, labels = read_sses(label_dir, (9999,9999))
    labels = labels.sort_values("classIndex")
    cmap = plt.get_cmap("tab20", len(labels))
    for _, row in labels.iterrows():
        index = row[0]
        name = row[1]
        color = cmap(class_to_idx[str(index)])
        pl.plot(0, 0, "-", c = color, label = name, linewidth = 10)
    pl.legend(loc = "center", prop = {"family": "MigMix 1P"})
    pl.savefig(out_path)
    pl.cla()

def plot_latent(f, q, x, y, cmap):
    f.eval()
    q.eval()
    with torch.no_grad():
        label = torch.argmax(y, dim = 1).detach().cpu().numpy()
        _y = f.sample_mean({"x":x})
        z = q.sample_mean({"x":x, "y":_y}).detach().cpu().numpy()
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(z[:, 0], z[:, 1], c=label, marker='o', edgecolor='none', cmap=cmap)
        plt.colorbar(ticks=range(cmap.N))
        plt.grid(True)
        fig.canvas.draw()
        image = fig.canvas.renderer._renderer
        image = np.array(image).transpose(2, 0, 1)
        image = np.expand_dims(image, 0)
        return image