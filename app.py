
import gradio as gr
import os
import torch

import model

from timeit import default_timer as timer
from typing import Tuple, Dict

class_names=['RG01 RX-78-2 Gundam', "RG02 MS-06S Char's Zaku II", 'RG03 GAT-X-105 Aile Strike Gundam', 'RG04 MS-06F Zaku II', 'RG05 ZGMF-X10A Freedom Gundam', 'RG06 FX-550 Sky Grasper', 'RG07 RX-178 Gundam Mk-II Titans', 'RG08 RX-178 Gundam Mk-II A.E.U.G.', 'RG09 ZGMF-X09A Justice Gundam', 'RG10 MSZ-006 Zeta Gundam', 'RG11 ZGMF-X42S Destiny Gundam', 'RG12 RX-78GP01 Zephyranthes', 'RG13 RX-78GP01fb Full Burnern', 'RG14 ZGMF-X20A Strike Freedom Gundam', 'RG15 GN-001 Gundam Exia', "RG16 MSM-07S Char's Z'gok", 'RG17 XXXG-00W0 Wing Gundam Zero EW', 'RG18 GN-0000-GNR-010 OO Raiser', 'RG19 MBF-P02 Gundam Astray Red Frame', 'RG20 XXXG-01W Wing Gundam EW', 'RG21 GNT-0000 OO Qan[T]', 'RG22 MSN-06S Sinanju', 'RG23 Build Strike Gundam Full Package', 'RG24 Gundam Astray Gold Frame Amatsu Mina', 'RG25 RX-0 Unicorn Gundam', "RG26 MS-06R-2 Johnny Ridden's Zaku II", 'RG27 RX-0[N] Unicorn Gundam 02 Banshee Norn', 'RG28 OZ-00MS Tallgeese EW', 'RG29 MSN-04 Sazabi', 'RG30 RX-0 Full Armor Unicorn Gundam', 'RG31 XM-X1 Crossbone Gundam X1', 'RG32 RX-93 Nu Gundam', 'RG33 ZGMF-X56S_α Force Impulse Gundam', 'RG34 MSN-02 Zeong', 'RG35 XXXG-01W Wing Gundam', 'RG36 RX-93-υ2 Hi-Nu Gundam', 'RG37 GF13-017NJII God Gundam']

device = "cuda" if torch.cuda.is_available() else "cpu"

vit, vit_transform = model.create_vit_model("./models/pretrained_vit2.pt", device, len(class_names))
vit=vit.to(device)

efficientnet, efficientnet_transform = model.create_efficientnet_model("./models/efficentnet_b2_argument.pt", device, len(class_names))
efficientnet=efficientnet.to(device)


def predict_func(model, transform, names, device) -> Tuple[Dict, float]:
    def inner_func(img):

    	"""Transforms and performs a prediction on img and returns prediction and time taken.
    	"""
    	# Start the timer
    	start_time = timer()
    	
    	# Transform the target image and add a batch dimension
    	img = transform(img).unsqueeze(0).to(device)
    	
    	# Put model into evaluation mode and turn on inference mode
    	model.eval()
    	with torch.inference_mode():
    	    # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
    	    pred_probs = torch.softmax(model(img), dim=1)
    	
    	# Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    	pred_labels_and_probs = {names[i]: float(pred_probs[0][i]) for i in range(len(names))}
    	
    	# Calculate the prediction time
    	pred_time = round(timer() - start_time, 5)
    	
    	# Return the prediction dictionary and prediction time 
    	return pred_labels_and_probs, pred_time

    return inner_func

vit_predict=predict_func(vit, vit_transform, class_names, device)
efficientnet_predict=predict_func(efficientnet, efficientnet_transform, class_names, device)


def predict(img, model="EfficientNet"):
    pf = vit_predict if model == "ViT" else efficientnet_predict
    
    return pf(img)

# Gradio app

title="Gunpla Classifier"
description="Which gunpla is this?"
example_list = [["./data/playground/" + example] for example in os.listdir("./data/playground/")]

demo = gr.Interface(
            fn=predict, 
            inputs=[
                gr.Image(type='pil', label="Upload Image"),
                gr.inputs.Dropdown(["EfficientNet", "ViT"], default="EfficientNet", label="Select Model"),
            ],
            outputs=[
                gr.Label(num_top_classes=3, label="Predictions"),
                gr.Number(label="Prediction time (s)"),
            ],
            examples=example_list,
            title=title,
            description=description,
        )


if __name__ == "__main__":
    demo.launch()
