from flask import Flask
from config import Config
import os

import sys


# Получаем абсолютный путь к GDRIVE_PATH (один уровень выше от Flask)
GDRIVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Формируем путь к sam2
SAM2_PATH = os.path.join(GDRIVE_PATH, "sam2")
sys.path.append(SAM2_PATH)
print('GDRIVE_PATH', GDRIVE_PATH)
print('SAM2_PATH',SAM2_PATH)



app = Flask(__name__,
           template_folder='../templates',
           static_folder='../static')

app.config.from_object(Config)

app.config['GDRIVE_PATH'] = GDRIVE_PATH
app.config['SAM2_PATH'] = SAM2_PATH

# Создаем необходимые директории при запуске
with app.app_context():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FRAME_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

from app import routes