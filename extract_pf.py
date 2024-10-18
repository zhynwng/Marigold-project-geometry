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


# # Load filenames
# filename_ls_path = "/share/data/p2p/zhiyanw/output.txt"
# with open(filename_ls_path, "r") as f:
#     content = f.readlines()
#     lines = [l.strip() for l in content]

# Select a model for Perspective Field inference
version = 'Paramnet-360Cities-edina-centered' # Trained on 360cities. Predicts perspective fields.
# Initialize the model and set it to evaluation mode and move it to GPU for faster processing
pf_model = PerspectiveFields(version).eval().cuda()

# Load an image
# for fn in tqdm(lines):
# img_path = "/share/data/p2p/zhiyanw/model_output/" + fn
img_path = "/share/data/p2p/zhiyanw/train_marigold/visualization/iter_024000/mini/image/1_pred.jpg"
img_bgr = cv2.imread(img_path)

# Perform inference to get predictions
predictions = pf_model.inference(img_bgr=img_bgr)
torch.save(predictions, "/share/data/p2p/yz5880/iter_024000_1_pred_field.pt")#+fn.split('.')[0]+".pt")


# if (fn.split(".")[0] == "100_pred"):
    # Draw and visualize the perspective fields based on the predictions
pred_field = draw_perspective_fields(
    img_bgr[...,::-1], 
    predictions["pred_gravity_original"].cpu().detach(), 
    torch.deg2rad(predictions["pred_latitude_original"].cpu().detach()), 
    color=(0,1,0),
)

plt.axis('off')
plt.imshow(pred_field)

plt.savefig("PF.png")
