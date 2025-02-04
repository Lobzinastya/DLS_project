import os
import cv2
from flask import current_app
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import json

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

    frame_dir = current_app.config['FRAME_FOLDER']
    annotation_path = os.path.join(current_app.config['ANNOTATIONS_FOLDER'], 'annotations.json')

    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"❌ Файл аннотаций не найден: {annotation_path}")
    with open(annotation_path, "r", encoding="utf-8") as f:
        annotation = f.read().strip()


    predict_alpha_mask(predictor, frame_dir, annotation)







def predict_alpha_mask(predictor, output_dir, annotation):
    import cv2

    torch.cuda.empty_cache()

    data = json.loads(annotation)
    points = [[entry["x"], entry["y"]] for entry in data]
    labels = [int(entry["label"]) for entry in data]

    ann_frame_idx = 0
    ann_obj_id = 0

    inference_state = predictor.init_state(video_path=output_dir)
    predictor.reset_state(inference_state)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                                      inference_state=inference_state,
                                                                      frame_idx=ann_frame_idx,
                                                                      obj_id=ann_obj_id,
                                                                      points=points,
                                                                      labels=labels,
                                                                      )

    mask_logits = []
    epsilon = 0.3  # Граница для бинаризации маски

    # Папка с кадрами
    frame_folder = current_app.config['FRAME_FOLDER']
    output_folder = current_app.config['OUTPUT_FOLDER']
    os.makedirs(output_folder, exist_ok=True)  # Создаём папку для выхода, если её нет

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # Получаем mask logits
        # Заглушка применяем сигмоиду + сглаживаем значения
        mask_pred = torch.sigmoid(out_mask_logits[out_obj_ids[0]])
        mask_pred = torch.clip(mask_pred, min=epsilon, max=1 - epsilon)
        mask_pred[mask_pred == epsilon] = 0
        mask_pred[mask_pred == 1 - epsilon] = 1

        mask_np = mask_pred.squeeze().cpu().numpy()

        frame_path = os.path.join(frame_folder, f"{out_frame_idx:04d}.jpg")
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)

            frame_float = frame.astype(np.float32) / 255.0
            masked_frame = frame_float * mask_np[:, :, np.newaxis]  # Применяем маску ко всем каналам - можно этого не делать, так как png сохраняется с альфа каналом?
            #masked_frame = frame_float
            masked_frame_uint8 = (masked_frame * 255).astype(np.uint8)

            alpha_channel = (mask_np * 255).astype(np.uint8)
            rgba_image = np.dstack((masked_frame_uint8, alpha_channel))  # Теперь (H, W, 4)

            output_path = os.path.join(output_folder, f"masked_{out_frame_idx:04d}.png")
            cv2.imwrite(output_path, rgba_image)
