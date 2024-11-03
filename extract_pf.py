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
filename_ls_path = "/scratch/yz5880/Projective-Geometry-1024/Indoor_Real_1024/40K_filenames.txt"
with open(filename_ls_path, "r") as f:
    content = f.readlines()
    lines = [l.strip() for l in content]
print("Finished reading filenames!")

# Select a model for Perspective Field inference
version = 'PersNet-360Cities' # Trained on 360cities. Predicts perspective fields.
# Initialize the model and set it to evaluation mode and move it to GPU for faster processing
pf_model = PerspectiveFields(version).eval().cuda()

print("extracting perspective fields...")
keys_to_extract = ['pred_gravity_original', 'pred_latitude_original']
# Load an image
for fn in tqdm(lines):
    img_path = os.path.basename(fn)
    if torch.load(img_path.split('.')[0]+".pt"):
        pass
    img_bgr = cv2.imread(fn)

    # Perform inference to get predictions
    predictions = pf_model.inference(img_bgr=img_bgr)
    new_dict = {key: predictions[key] for key in keys_to_extract}
    torch.save(new_dict, "/scratch/yz5880/Projective-Geometry-1024/Indoor_Real_1024/field/"+img_path.split('.')[0]+".pt")
    

    # if (fn.split(".")[0] == "100_pred"):
    #     # Draw and visualize the perspective fields based on the predictions
    #     pred_field = draw_perspective_fields(
    #         img_bgr[...,::-1], 
    #         predictions["pred_gravity_original"].cpu().detach(), 
    #         torch.deg2rad(predictions["pred_latitude_original"].cpu().detach()), 
    #         color=(0,1,0),
    #     )

    #     plt.axis('off')
    #     plt.imshow(pred_field)

    #     plt.savefig("PF.png")
