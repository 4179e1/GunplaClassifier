
from typing import List
import torch
import torchvision
import matplotlib.pyplot as plt
import data_utils
import PIL

default_device = "cuda" if torch.cuda.is_available() else "cpu"



def predict(model: torch.nn.Module, 
                        image_path: str, 
                        class_names: List[str] = None, 
                        device: torch.device = default_device,
                        transform=None,
                        topk = 3,
           ):
    """Makes a prediction on a target image and plots the image with its prediction."""
    
    # 1. Load in image
    pil_image = PIL.Image.open(image_path)
        
    # 2. Transform with the same transformer
    if transform:
        target_image = transform(pil_image)
        
    
    # 3. Make sure the model is on the target device
    model.to(device)
    
    # 4. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
    
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))
        
    # 5. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 6. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)    
    
    probs, indexes = target_image_pred_probs.data[0].topk(topk,sorted=True, largest=True)
    probs = probs.tolist()
    indexes = indexes.tolist()
    
    prob = probs[0]
    pred = indexes[0]
    

    # 7. Plot the image alongside the prediction and prediction probability
    if class_names:
        title = f"Pred: {class_names[pred]} | Prob: {prob:.3f}"

        for p, i in zip(probs, indexes):
            print (f"{class_names[i]: <20} : {p:.3f}")
    else: 
        title = f"Pred: {pred} | Prob: {prob:.3f}"
        
    data_utils.plot_images([pil_image, target_image.squeeze(dim=0)], [title, title])    
