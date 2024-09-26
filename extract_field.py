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
filename_ls_path = "/share/data/p2p/yz5880/eval_mini/original_image_list.txt"
with open(filename_ls_path, "r") as f:
    content = f.readlines()
    lines = [l.strip() for l in content]

# Select a model for Perspective Field inference
version = 'PersNet-360Cities' # Trained on 360cities. Predicts perspective fields.
# Initialize the model and set it to evaluation mode and move it to GPU for faster processing
pf_model = PerspectiveFields(version).eval().cuda()

# Load an image
keys_to_extract = ['pred_gravity_original', 'pred_latitude_original']
for fn in tqdm(lines):
    img_path = "/share/data/p2p/yz5880/eval_mini/original_image/image/" + fn
    img_bgr = cv2.imread(img_path)

    # Perform inference to get predictions
    predictions = pf_model.inference(img_bgr=img_bgr)    
    device = 'cuda:0'
    

    # Print the keys of the predictions dictionary
    # print("pred_gravity_original", predictions['pred_gravity_original'].shape)
    # print("pred_latitude_original", prediciotns['pred_latitude_original'].shape)
    torch.save({key: predictions[key] for key in keys_to_extract}, "/share/data/p2p/yz5880/eval_mini/original_image/field/"+fn.split('.')[0]+".pt")
    # torch.save(predictions, "/share/data/p2p/yz5880/eval_mini/"+fn.split('.')[0]+".pt")

    # Draw and visualize the perspective fields based on the predictions
    # pred_field = draw_perspective_fields(
    #     img_bgr[...,::-1], 
    #     predictions["pred_gravity_original"].cpu().detach(), 
    #     torch.deg2rad(predictions["pred_latitude_original"].cpu().detach()), 
    #     color=(0,1,0),
    # )

    # plt.axis('off')
    # plt.imshow(pred_field)

    # plt.savefig("PF.png")

    # sys.exit()