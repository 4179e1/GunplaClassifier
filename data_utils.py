
from typing import List, Union, Optional

import pathlib
from pathlib import Path

from PIL import Image
import pandas as pd
import torch
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.transforms.functional as F


def get_random_image(image_dir: str, pattern: str="*.jpg") -> pathlib.Path:
    """
    Get a random image path from given directory.
    
    Args:
        image_dir: the direcory to look at
        pattern: the pattern of the filename
    
    Return:
        The path of the random selected image
        
    Note: 
        The lookup is recursive, for example, withe image_dir="./" and pattern ="*.jpg",
        it could match "./abc.jpg" or "./subdir/def.jpg" 
        
    """
    path = pathlib.Path(image_dir)

    # 1. Get all image paths
    image_path_list = list(path.rglob(pattern))
    
    # 2. Get random image path
    img_path = random.choice (image_path_list)
    return img_path

def get_random_images(image_dir: str, k: int=3, pattern: str="*.jpg") -> List[pathlib.Path]:
    """
    Get k random image path from given directory.
    
    Args:
        image_dir: the direcory to look at
        k: the number of image to select
        pattern: the pattern of the filename
    
    Return:
        The path of the random selected image
        
    Note: 
        The lookup is recursive, for example, withe image_dir="./" and pattern ="*.jpg",
        it could match "./abc.jpg" or "./subdir/def.jpg" 
        
    """
    path = pathlib.Path(image_dir)

    # 1. Get all image paths
    image_path_list = list(path.rglob(pattern))
    
    # 2. Get random image path
    pathes = random.choices (image_path_list, k=k)
    
    return pathes

def get_image_class(img_path: Union[pathlib.Path, str]) -> str:
    """
    Given a path to image inside a standard Pytorch ImageFolder 
    (https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html),
    return the class of that image.
    
    Args:
        img_path: the path to the image, for example "root/dog/xxx.png"
        
    Returns:
        the class of that image, it returns "dog" in the example above
    """
    
    # Make sure it's an instance of pathlib.Path
    p = pathlib.Path (img_path)
    
    return p.parent.stem
    
    
def display_image(img_path: Union[pathlib.Path, str]) -> None:
    """
    Dispaly the specified image in IPython console, 
    also print it's path, height, width
    
    Args:
        image_path: the path of the image
    
    Return:
        None
           
    Note:
        This works only in IPython console.
    """

    img = Image.open(img_path)

    # 5. Print metadata
    print(f"Image path: {img_path}")
    print(f"Image height: {img.height}") 
    print(f"Image width: {img.width}")
    
    
    display(img)
        
def _get_image_data (
        image: Union[pathlib.Path, str, Image.Image, torch.Tensor]
    ) -> np.ndarray:

    if isinstance (image, str) or isinstance (image, pathlib.Path):
        image = Image.open(image)

    if isinstance(image, Image.Image):
        image = np.asarray(image)
        return image
    elif isinstance(image, torch.Tensor):
        # converting Tensor to numpy is not necessary, but make sure it always return a numpy.ndarray
        image = image.permute(1, 2, 0).numpy()
        return image
    else:
        raise TypeError(f"Unsupported type {type(image)}")

def plot_image(image: Union[pathlib.Path, str, Image.Image, torch.Tensor], title: Optional[str]=None) -> None:
    """
    Plot image with matplotlib
    
    Args:
        image: either the path of image, 
               or the image opened as PIL.Image.Image (via PIL.Image.Open),
               or the image transformed to pytorch tensor
    
    Return:
        None
    """

    data = _get_image_data(image)

    plt.imshow(data)
    if title:
        plt.title(title)

    plt.show()

    
def plot_images(
        images: List[Union[pathlib.Path, str, Image.Image, torch.Tensor]], 
        titles: Optional[List[str]]=None
    ) -> None:
    """
    Plot image with matplotlib, all images are show in a figure horizontally
    
    Args:
        images: a list containing the image data
        titles: a list containint the title of each image
    
    Return:
        None
        
    Note:
        If not None, the length of titles must match the length of img_paths
    """
    if titles:
        assert len(images) == len(titles)
    else:
        titles = [None] * len(images)
    
    ncols = len(images)
    
    fig, axs = plt.subplots(ncols=ncols, figsize=(ncols*6,6))
    
    for ax, i, t in zip (axs, images, titles):
        image = _get_image_data (i)
        ax.imshow(image)
        if t:
            ax.title.set_text(t)
    
    plt.show()

        
def show_random_image(image_dir: str, pattern: str="*.jpg") -> None:
    """
    Show an image randomly selected from given directory.
    
    Args:
        image_dir: the direcory to look at
        pattern: the pattern of the filename
    
    Return:
        None
        
    Note: 
        The lookup is recursive, for example, withe image_dir="./" and pattern ="*.jpg",
        it could match "./abc.jpg" or "./subdir/def.jpg" 
        
    """
    
    img_path = get_random_image(image_dir, pattern)
    label = get_image_class (img_path)
    plot_image(img_path, label)
    
def show_random_images(image_dir: str, k: int=3, pattern="*.jpg"):
    """
    Show k images randomly selected from given directory.
    
    Args:
        image_dir: the direcory to look at
        k: the number of images to show
        pattern: the pattern of the filename
    
    Return:
        None
        
    Note: 
        The lookup is recursive, for example, withe image_dir="./" and pattern ="*.jpg",
        it could match "./abc.jpg" or "./subdir/def.jpg" 
        
    """
    img_paths = get_random_images(image_dir, k, pattern)
    labels = [get_image_class(x) for x in img_paths]
    plot_images(img_paths, labels)
    
def plot_patched_image (image: Union[pathlib.Path, str, Image.Image, torch.Tensor], patch_size: int=16):
    
    img = _get_image_data(image)
    assert img.shape[0] == img.shape[1], "Image must be square"
    img_size = img.shape[1]
    
    assert img_size % patch_size == 0, "Image size must be divisible by patch size" 
    num_patches = img_size/patch_size
    print(f"Image size: {img.shape[0]} x {img.shape[1]}\
        \nNumber of patches per row: {num_patches}\
        \nNumber of patches per column: {num_patches}\
        \nTotal patches: {num_patches*num_patches}\
        \nPatch size: {patch_size} pixels x {patch_size} pixels")
    
    fig, axs = plt.subplots(nrows=img_size//patch_size, # need int
            ncols=img_size//patch_size,
            figsize=(num_patches, num_patches),
            sharex=True,
            sharey=True,
    )
    
    for i, h in enumerate(range(0, img_size, patch_size)):
        for j, w in enumerate(range(0, img_size, patch_size)):
            ax = axs[i, j]
            ax.imshow(img[h:h+patch_size, w:w+patch_size, :])
            
            ax.set_ylabel(i+1, 
                        rotation="horizontal", 
                        horizontalalignment="right",
                        verticalalignment="center")
            ax.set_xlabel(j+1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.label_outer()
            
    fig.suptitle("Image Patchified", fontsize=16)
    plt.show()
    
def show_image_statistics(image_dir, pattern="*.jpg"):
    """
    Show statistics of all matched images under image_dir
    
    Args:
        image_dir: the direcory to look at
        pattern: the pattern of the filename
    
    Return:
        None
        
    Note: 
        The lookup is recursive, for example, withe image_dir="./" and pattern ="*.jpg",
        it could match "./abc.jpg" or "./subdir/def.jpg" 
        
    """
    path = Path(image_dir)
    
    # 1. Get all image paths
    image_path_list = list(path.rglob(pattern))
    
    c = Counter()
    wh=[]
    for x in image_path_list:
        img = Image.open(x)
        c[x.parent.stem] += 1

        wh.append([img.width, img.height])
        del img
    
    t=torch.tensor(wh, dtype=torch.float)
    mean = t.mean(dim=0)
    stddev = t.std(dim=0)
    median,_ = t.median(dim=0)
    minimal, _ = t.min(dim=0) # return value,index
    maximal, _ = t.max(dim=0)
    
    print (f"# of images: {len(image_path_list)}")
    
    for k in c:
        print (f"\t{k:20}: {c[k]}")

    print("Avg ", mean)
    print("Std ", stddev)
    print("Med ", median)

    print('Min ', minimal)
    print('Max ', maximal)
    

    plt.xlabel('width')
    plt.ylabel('height')
    # WTF, see https://stackoverflow.com/questions/41248767/splitting-a-2-dimensional-array-or-a-list-into-two-1-dimensional-lists-in-python
    w,h = zip(*wh)
    plt.scatter(w,h)
    plt.plot(*mean.tolist(), marker="o", markersize=10, markerfacecolor="green", label="mean")
    plt.plot(*median.tolist(), marker="o", markersize=10, markerfacecolor="yellow", label="median")

    plt.legend(loc='best')
    plt.show()
