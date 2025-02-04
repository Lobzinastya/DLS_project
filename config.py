import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    ANNOTATIONS_FOLDER = os.path.join(BASE_DIR, 'annotations')

    SECRET_KEY = 'your-secret-key'
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads', 'videos')
    FRAME_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads', 'frames')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads', 'output')
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024