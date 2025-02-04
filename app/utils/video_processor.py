import cv2
import os


# def process_video(video_path, output_folder, resolution):
#     # Добавим resolution в параметры
#     os.makedirs(output_folder, exist_ok=True)
#
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError("Error opening video file")
#
#     success, frame = cap.read()
#     if success:
#         frame = cv2.resize(frame, resolution)
#         first_frame_path = os.path.join(output_folder, '0000.jpg')
#
#         # Конвертируем BGR в RGB
#       #  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         success = cv2.imwrite(first_frame_path, frame)
#
#         if not success:
#             raise RuntimeError("Failed to save frame")
#
#     cap.release()
#     return first_frame_path




def process_video(video_path, output_folder, resolution):
    """
    Обрабатывает видео: сохраняет все кадры в `output_folder` с именами 0000.jpg, 0001.jpg и т. д.
    Возвращает путь к первому кадру.

    :return: Путь к первому кадру (0000.jpg)
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    frame_idx = 0
    first_frame_path = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, resolution)
        frame_filename = f"{frame_idx:04d}.jpg"  # 0000.jpg, 0001.jpg, 0002.jpg ...
        frame_path = os.path.join(output_folder, frame_filename)

        success = cv2.imwrite(frame_path, frame)
        if not success:
            raise RuntimeError(f"Failed to save frame {frame_filename}")

        if frame_idx == 0:
            first_frame_path = frame_path

        frame_idx += 1

    cap.release()
    return first_frame_path



