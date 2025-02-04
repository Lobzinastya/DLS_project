from flask import render_template, request, redirect, url_for, session
from app import app
import os
from werkzeug.utils import secure_filename
from app.utils.video_processor import process_video
from app.utils.generate_processor import generation

from flask import current_app, jsonify
import json
import shutil




def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clear_folders():
    """Удаляет все файлы внутри папок UPLOAD_FOLDER, FRAME_FOLDER, OUTPUT_FOLDER, ANNOTATIONS_FOLDER
    """
    folders = [
        current_app.config['UPLOAD_FOLDER'],
        current_app.config['FRAME_FOLDER'],
    #    current_app.config['OUTPUT_FOLDER'],
        current_app.config['ANNOTATIONS_FOLDER']
    ]

    for folder in folders:
        if os.path.exists(folder):  # папка существует
            for file_name in os.listdir(folder):
                file_path = os.path.join(folder, file_name)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Удаляем файлы и символические ссылки
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Удаляем вложенные папки
                except Exception as e:
                    current_app.logger.error(f"Ошибка при удалении {file_path}: {e}")




@app.route('/', methods=['GET', 'POST'])
def index():
    # удаление всего сохраненного в папках
    clear_folders()

    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)

        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            # Обработка видео
            frame_path = None
            try:
                frame_path = process_video(
                    video_path,
                    app.config['FRAME_FOLDER'],
                    (640, 360)  # width=640, height=360
                )

                relative_path = os.path.relpath(
                    frame_path,
                    start=current_app.config['BASE_DIR']  # Используем конфиг приложения
                ).replace('\\', '/')

            except Exception as e:
                app.logger.error(f"Video processing failed: {str(e)}")
                return "Error processing video", 500

            if not frame_path:
                return "Failed to extract first frame", 500

            session['video_path'] = video_path
            session['first_frame'] = relative_path



            print("Absolute frame path:", frame_path)
            print("Relative path for web:", relative_path)
            print("File exists:", os.path.exists(frame_path))

            return redirect(url_for('annotate'))

    return render_template('index.html')


@app.route('/annotate')
def annotate():
    frame_path = session.get('first_frame')

    if not frame_path or not os.path.exists(frame_path):
        return "First frame not found", 404

    # Используем current_app вместо Config
    base_dir = current_app.config['BASE_DIR']
    relative_path = os.path.relpath(frame_path, start=base_dir).replace('\\', '/').replace('static/', '', 1)

    return render_template('annotate.html', frame_path=relative_path)


@app.route('/save-annotations', methods=['POST'])
def save_annotations():
    try:
        data = request.json
        annotations = data.get('points', [])
        print('Получены аннотации', annotations)

        # Валидация данных
        for ann in annotations:
            if not all(key in ann for key in ('x', 'y', 'label')):
                return jsonify({"status": "error", "message": "Invalid annotation format"}), 400
            if ann['label'] not in ('0', '1'):
                return jsonify({"status": "error", "message": "Invalid label value"}), 400

        # Сохраняем в сессии
        session['annotations'] = annotations

        # Дополнительно: сохраняем в файл
        annotations_dir = os.path.join(current_app.config['BASE_DIR'], 'annotations')
        os.makedirs(annotations_dir, exist_ok=True)

        with open(os.path.join(annotations_dir, 'annotations.json'), 'w') as f:
            json.dump(annotations, f)

        return jsonify({"status": "success", "count": len(annotations)})

    except Exception as e:
        current_app.logger.error(f"Annotation error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/process', methods=['POST'])
# def process():
#     # Здесь будет обработка точек
#     return redirect(url_for('result'))


@app.route('/generate')
def result():
    import os
    import cv2
    import subprocess

    generation()
    print('Генерация отдельных png сделана')

    input_folder = current_app['OUTPUT_FOLDER']
    image_files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    if not image_files:
        raise ValueError("В папке нет рендеров PNG-файлов!")

    first_image_path = os.path.join(input_folder, image_files[0])
    first_image = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)

    if first_image is None:
        raise ValueError(f"Ошибка чтения изображения {first_image_path}")
    h, w = first_image.shape[:2]

    has_alpha = first_image.shape[2] == 4
    if not has_alpha:
        raise ValueError("PNG-файлы должны содержать альфа-канал (RGBA)")

    input_pattern = os.path.join(input_folder, "%04d.png")  # Ожидает файлы вида 0000.png, 0001.png...

    output_webm = os.path.join(input_folder,"final_result.webm")
    fps = 25

    # Команда FFmpeg для создания WebM с прозрачностью
    ffmpeg_cmd = [
        "ffmpeg",
        "-framerate", str(fps),  # FPS
        "-i", input_pattern,  # Шаблон входных файлов (ожидаются файлы с номерами)
        "-c:v", "libvpx-vp9",  # Кодек VP9 (WebM с прозрачностью)
        "-pix_fmt", "yuva420p",  # Поддержка прозрачности (alpha channel)
        "-b:v", "1M",  # Битрейт видео
        "-y", output_webm  # Перезаписать файл, если уже существует
    ]

    # Запускаем FFmpeg через subprocess
    subprocess.run(ffmpeg_cmd, check=True)



    static_webm_path = "static/uploads/output/sample_sticker.webm"
    return render_template('result.html', webm_path = static_webm_path)