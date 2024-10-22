import cv2
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from torch.nn.functional import cosine_similarity as cosine_similarity

filename_ls_path = "/share/data/p2p/yz5880/sorted_fields.txt"

# Load filenames
with open(filename_ls_path, "r") as f:
    filenames = [
        s.strip() for s in f.readlines()
    ]

name_to_write = []
for f in tqdm(filenames):
    field_og = torch.load(f+"/original.pt")
    field_og_squeeze = field_og.reshape(-1).unsqueeze(0)
    worst_pf = None
    best_pf = None
    worst_distance = 0
    best_distance = 1e10
    
    for i in range(5):
        field = torch.load(f+f"/{i}_pred.pt").permute(1, 2, 0).cpu()
        field_squeeze = field.reshape(-1).unsqueeze(0)

        # if torch.norm(field-field_og, p=2) > worst_distance:
            # worst_distance = torch.norm(field-field_og, p=2)
        if cosine_similarity(field_squeeze, field_og_squeeze) > worst_distance:
            worst_distance = cosine_similarity(field_squeeze, field_og_squeeze)            
            worst_pf = f+f"/{i}_pred.pt"
        # if torch.norm(field-field_og, p=2) < best_distance:
            # best_distance = torch.norm(field-field_og, p=2)
        if cosine_similarity(field_squeeze, field_og_squeeze) < best_distance:
            best_distance = cosine_similarity(field_squeeze, field_og_squeeze)            
            best_pf = f+f"/{i}_pred.pt"
        
    name_to_write.append(f + "/original.pt " + best_pf + " " + worst_pf)

    with open("contrastive_fields.txt", "w") as file:
        for item in name_to_write:
            file.write(f"{item}\n")