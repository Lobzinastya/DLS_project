from flask import render_template, request, redirect, url_for, session
from app import app
import os
from werkzeug.utils import secure_filename
from app.utils.video_processor import process_video
from flask import current_app, jsonify
import json


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def index():
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

@app.route('/process', methods=['POST'])
def process():
    # Здесь будет обработка точек
    return redirect(url_for('result'))


@app.route('/result')
def result():
    return render_template('result.html')