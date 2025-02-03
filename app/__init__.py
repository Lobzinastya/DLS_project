from flask import Flask
from config import Config
import os

app = Flask(__name__,
           template_folder='../templates',
           static_folder='../static')
app.config.from_object(Config)

# Создаем необходимые директории при запуске
with app.app_context():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FRAME_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

from app import routes