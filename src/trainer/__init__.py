# Author: Bingxin Ke
# Last modified: 2024-05-17

from .marigold_trainer import MarigoldTrainer
from .finetune_trainer import FinetuneTrainer
from .marigold_trainer_predict_image import MarigoldTrainerImage
from .marigold_trainer_sdxl import SDXLTrainer


trainer_cls_name_dict = {
    "MarigoldTrainer": MarigoldTrainer,
    "FinetuneTrainer": FinetuneTrainer,
    "MarigoldTrainerImage": MarigoldTrainerImage,
    "SDXLTrainer": SDXLTrainer,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
