import matplotlib.pyplot as plt
from perspective2d.utils import draw_perspective_fields
import torch
from PIL import Image
import numpy as np

def visualize_image(image_path, field_path):
    # Read the image and field file
    image = Image.open(image_path)  # [H, W, rgb]
    image = np.array(image)

    field = torch.load(field_path)
    lat = field['pred_latitude_original']
    grav = field['pred_gravity_original']

    print(image.shape)
    # draw Perspective field 
    img = draw_perspective_fields(image, grav, torch.deg2rad(lat))

    # Display the image
    plt.imshow(img)
    plt.savefig('/home-nfs/zhiyanw/vis.jpg')
    plt.close()

# Example usage
image_path = '/home-nfs/zhiyanw/mini/test/image/real/396461.jpg'
# Replace with your image file path
field_path = '/home-nfs/zhiyanw/mini/test/fields/real/real_396461.pt'
visualize_image(image_path, field_path)