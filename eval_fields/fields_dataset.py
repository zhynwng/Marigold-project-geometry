import torch
from torch.utils.data import Dataset, DataLoader
import sys


class PerspectiveMapDataset(Dataset):
    def __init__(self, image_paths, class_to_idx, category):
        self.image_paths = image_paths
        self.class_to_idx = class_to_idx
        self.category = category
    
    def __len__(self):
        return len(self.image_paths)
    
    def transform_maps(self, latitude_map, gravity_maps):
        latitude_map = latitude_map / 90.0
        joined_maps = torch.cat([latitude_map.unsqueeze(0), gravity_maps], dim = 0)
        return joined_maps
    
    def image_path_to_field_path(self, image_filepath):
        # identifier = image_filepath.split("/")[-2]
        # number = (image_filepath.split("/")[-1]).split('.')[0]
        field_path = image_filepath # f'/share/data/p2p/yz5880/eval_mini/test_gen/field/{identifier}_{number}.pt'
        return field_path
    
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        
        field_path = self.image_path_to_field_path(image_filepath)
        
        field = torch.load(field_path)
        
        latitude_map = field['pred_latitude_original']
        gravity_maps = field['pred_gravity_original']
        
        joined_maps = self.transform_maps(latitude_map,  gravity_maps)
        
        if self.category == "real":
            label = "real"
        elif self.category in ["generated", "generated_before_finetuning"]:
            label = "gen"
        else:
            print("category not implemented!")
        label = self.class_to_idx[label]
        
        return joined_maps, label
    
    
def get_train_dataloaders(train_image_paths, val_image_paths, class_to_idx):
    train_dataset = PerspectiveMapDataset(train_image_paths, class_to_idx)
    val_dataset = PerspectiveMapDataset(val_image_paths, class_to_idx)

    train_dataloader = DataLoader(train_dataset, batch_size = 256, shuffle = True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size = 256, shuffle = True, num_workers=6)

    return train_dataloader, val_dataloader


def get_test_dataloaders(test_image_paths, class_to_idx, category):

    field_dataset = PerspectiveMapDataset(test_image_paths, class_to_idx, category)
    field_dataloader = DataLoader(field_dataset, batch_size=256, shuffle=True)
    
    return field_dataloader 