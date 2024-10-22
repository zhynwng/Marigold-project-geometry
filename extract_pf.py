import cv2
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from perspective2d import PerspectiveFields
from perspective2d.utils import draw_perspective_fields, draw_from_r_p_f_cx_cy
from tqdm import tqdm
import time


# Load filenames
lines = os.listdir("/share/data/p2p/zhiyanw/train_marigold/visualization/iter_029250/mini/image/")

# Select a model for Perspective Field inference
version = 'Paramnet-360Cities-edina-centered' # Trained on 360cities. Predicts perspective fields.
# Initialize the model and set it to evaluation mode and move it to GPU for faster processing
pf_model = PerspectiveFields(version).eval().cuda()

# Load an image
for fn in tqdm(lines):
    img_path = "/share/data/p2p/zhiyanw/train_marigold/visualization/iter_029250/mini/image/" + fn
    img_bgr = cv2.imread(img_path)

    # Perform inference to get predictions
    predictions = pf_model.inference(img_bgr=img_bgr)
    torch.save(predictions, "/share/data/p2p/zhiyanw/field/"+fn.split('.')[0]+"_added.pt")
    

    # Draw and visualize the perspective fields based on the predictions
    pred_field = draw_perspective_fields(
        img_bgr[...,::-1], 
        predictions["pred_gravity_original"].cpu().detach(), 
        torch.deg2rad(predictions["pred_latitude_original"].cpu().detach()), 
        color=(0,1,0),
    )

    plt.axis('off')
    plt.imshow(pred_field)

    plt.savefig("/share/data/p2p/zhiyanw/vis/"+fn.split('.')[0]+"_added.png")
