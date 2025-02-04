import os
from flask import current_app
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision

def generation():
    gdrive_path = current_app.config["GDRIVE_PATH"]
    sam2_path = current_app.config["SAM2_PATH"]

    sam2_dir = sam2_path
    sys.path.append(sam2_dir)
    sam2_checkpoint_dir = os.path.join(sam2_dir, "sam2_hiera_small")
    os.makedirs(sam2_checkpoint_dir, exist_ok=True)

    # чекпоинт для small модели
    # !wget -P "{sam2_checkpoint_dir}" https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt

    from sam2.build_sam import build_sam2_video_predictor

    sam2_checkpoint = os.path.join(sam2_checkpoint_dir, "sam2_hiera_small.pt")
    model_cfg = "sam2_hiera_s.yaml"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    print('Predictor SAM2 загружен')