import cv2
import os


def process_video(video_path, output_folder, resolution):
    # Добавим resolution в параметры
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame, resolution)
        first_frame_path = os.path.join(output_folder, 'first_frame.jpg')

        # Конвертируем BGR в RGB
      #  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        success = cv2.imwrite(first_frame_path, frame)

        if not success:
            raise RuntimeError("Failed to save frame")

    cap.release()
    return first_frame_path