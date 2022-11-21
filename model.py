
import torch
import torchvision
import dataset

from torch import nn
from torchvision import transforms

def create_vit_model(model_path: str,
                     device: str,
                     num_classes:int=3
                    ):
    model = torchvision.models.vit_b_16()
    model.heads = nn.Linear(in_features=768, out_features=num_classes) 

    model.load_state_dict(torch.load(f=model_path, map_location=device))
    
    transform = transforms.Compose([
        dataset.SquarePad(fill=255),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
    return model, transform


def create_efficientnet_model(model_path: str,
                     device: str,
                     num_classes:int=3
                    ):
    model = torchvision.models.efficientnet_b2()
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1408,         # note: this is different from b0
                    out_features=num_classes, # same number of output units as our number of classes
                    bias=True))

    model.load_state_dict(torch.load(f=model_path, map_location=device))
    
    transform = transforms.Compose([
        dataset.SquarePad(fill=255),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
    return model, transform
