import os
import torch
import PIL
import json

from typing import List, Optional, Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torchvision.transforms.functional as F


NUM_WORKERS = os.cpu_count()

DEFAULT_TRAIN_DIR="./data/gunpla/train"
DEFAULT_TEST_DIR="./data/gunpla/test"
DEFAULT_VALIDATE_DIR="./data/gunpla/validate"
DEFAULT_METADATA="./data/gunpla/metadata.json"

def create_dataset(
        train_dir: str = DEFAULT_TRAIN_DIR,
        test_dir: str = DEFAULT_TEST_DIR,
        validate_dir: str = DEFAULT_VALIDATE_DIR,

        train_transform: Optional[transforms.Compose] = None,
        test_transform: Optional[transforms.Compose] = None,
        validate_transform: Optional[transforms.Compose] = None,
    
        metadata: str = DEFAULT_METADATA,

    ) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder, List[str]]:
    """Creates training and testing dataset.
    
    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets
    
    Args:
      train_dir: Path to training directory.
      test_dir: Path to testing directory.
      train_transform: optional torchvision transforms to perform on training data.
      test_transform: optional torchvision transforms to perform on testing data.

    
    Returns:
      A tuple of (train_dataset, test_dataset, class_names).
      Where class_names is a list of the target classes.
    """
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    validate_data = datasets.ImageFolder(validate_dir, transform=validate_transform)

    
    class_names = train_data.classes
    
    with open(metadata) as f:
        meta = json.load(f)
        
    items=[]
    for x in class_names:
        m=meta[x]
        item=x
        code,name = m["code"], m["name"]
        if code:
            item += f" {code}"
        if name:
            item += f" {name}"
            
        items.append(item)
        
    return train_data, test_data, validate_data, items

def create_dataloaders(
        train_dir: str = DEFAULT_TRAIN_DIR,
        test_dir: str = DEFAULT_TEST_DIR,
        validate_dir: str = DEFAULT_VALIDATE_DIR,

        train_transform: Optional[transforms.Compose] = None,
        test_transform: Optional[transforms.Compose] = None,
        validate_transform: Optional[transforms.Compose] = None,
        batch_size: Optional[int]=32,
        num_workers: Optional[int]=NUM_WORKERS,
        metadata: str=DEFAULT_METADATA,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:

    """Creates training and testing dataloader.
    
    Takes in a training directory and testing directory path and turns
    them into PyTorch DataloLoader
    
    Args:
      train_dir: Path to training directory.
      test_dir: Path to testing directory.
      train_transform: optional torchvision transforms to perform on training data.
      test_transform: optional torchvision transforms to perform on testing data.

    
    Returns:
      A tuple of (train_dataloader, test_dataloader, class_names).
      Where class_names is a list of the target classes.
    """
    
    train_data, test_data, validate_data, class_names = create_dataset(train_dir, test_dir, validate_dir, train_transform, test_transform, validate_transform)
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    validate_dataloader = DataLoader(
        validate_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_dataloader, test_dataloader, validate_dataloader, class_names

class SquarePad(torch.nn.Module):
    """
    Taken from https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/10
    With modification
    """
    def __init__(self, fill: int=0):
        super().__init__()
        self.fill = fill
        
    def __call__(self, image: PIL.Image.Image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, fill=self.fill, padding_mode='constant')
