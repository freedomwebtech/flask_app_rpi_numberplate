from flask import Flask, request, render_template, send_from_directory
import os
import pandas as pd
import sqlite3
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}


ocr = PaddleOCR()
model = YOLO('best.pt')  

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def perform_ocr(image_path):
    result = ocr.ocr(image_path, rec=True)
    detected_texts = []
    if result and result[0] is not None:
        for line in result[0]:
            text = line[1][0]
            detected_texts.append(text)
    return ' '.join(detected_texts)

def process_image_with_yolo(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Perform object detection
    results = model.predict(image, imgsz=240)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    cropped_images = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        crop = image[y1:y2, x1:x2]
        crop_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'crop_{index}_{os.path.basename(image_path)}')
        cv2.imwrite(crop_filename, crop)
        cropped_images.append(crop_filename)
    
    return cropped_images

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error='No selected file')
            if file and allowed_file(file.filename):
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                
                image = cv2.imread(filename)
                if image is None:
                    return render_template('index.html', error='Failed to process image')
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, timestamp, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(filename, image)
                
                cropped_images = process_image_with_yolo(filename)
                ocr_texts = [perform_ocr(cropped_image) for cropped_image in cropped_images]
                
                conn = sqlite3.connect('example.db')
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT,
                        timestamp TEXT,
                        ocr_text TEXT
                    )
                ''')
                cursor.execute("INSERT INTO images (filename, timestamp, ocr_text) VALUES (?, ?, ?)",
                               (file.filename, timestamp, ', '.join(ocr_texts)))
                conn.commit()
                conn.close()
                
                return render_template('index.html', filename=file.filename, ocr_texts=ocr_texts)
    
    search_query = request.args.get('search')
    search_time = request.args.get('time')
    filtered_images = []
    
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    
    if search_query:
        if search_time:
            search_date = search_query
            search_time = search_time
            query = '''
                SELECT filename, timestamp, ocr_text 
                FROM images 
                WHERE timestamp LIKE ? AND strftime('%H:%M', timestamp) >= ?
            '''
            cursor.execute(query, (f'{search_date}%', search_time))
        else:
            search_date = search_query
            query = '''
                SELECT filename, timestamp, ocr_text 
                FROM images 
                WHERE timestamp LIKE ?
            '''
            cursor.execute(query, (f'{search_date}%',))
        
        rows = cursor.fetchall()
        for row in rows:
            filename, timestamp, ocr_text = row
            filtered_images.append({
                'filename': filename,
                'timestamp': timestamp,
                'ocr_text': ocr_text
            })
    
    conn.close()
    
    return render_template('index.html', filtered_images=filtered_images, search_query=search_query, search_time=search_time)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
