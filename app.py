from flask import Flask, request, render_template, send_from_directory
import os
import pandas as pd
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime  # Import for timestamp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Initialize PaddleOCR and YOLOv8
ocr = PaddleOCR()
model = YOLO('best.pt')  # Replace with your YOLOv8 model path

# List to store metadata (filename, timestamp, OCR text)
image_metadata = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def perform_ocr(image_path):
    result = ocr.ocr(image_path, rec=True)
    detected_texts = []
    if result and result[0] is not None:
        for line in result[0]:
            text = line[1][0]
            print(text)
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
        d = int(row[5])
        crop = image[y1:y2, x1:x2]
#        crop = cv2.resize(crop, (110, 70))
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
                
                # Add current date and time on the uploaded image itself
                image = cv2.imread(filename)
                if image is None:
                    return render_template('index.html', error='Failed to process image')
                
                # Get the current date and time
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Add timestamp to the top-left corner of the uploaded image
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, timestamp, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Save the modified image (overwrite the original image with the timestamp)
                cv2.imwrite(filename, image)
                
                # Process image with YOLO and OCR
                cropped_images = process_image_with_yolo(filename)
                
                ocr_texts = []
                for cropped_image_path in cropped_images:
                    text = perform_ocr(cropped_image_path)
                    ocr_texts.append(text)
                
                # Store metadata (filename, timestamp, and OCR text)
                image_metadata.append({
                    'filename': file.filename, 
                    'timestamp': timestamp, 
                    'ocr_text': ', '.join([ocr[1] for ocr in ocr_texts])  # Save detected text from cropped images
                })
                
                return render_template('index.html', filename=file.filename, ocr_texts=ocr_texts)
    
    # Handle search query (GET request)
    search_query = request.args.get('search')
    filtered_images = []
    if search_query:
        for meta in image_metadata:
            if search_query in meta['timestamp'] or search_query in meta['ocr_text']:
                filtered_images.append(meta)
    
    return render_template('index.html', filtered_images=filtered_images, search_query=search_query)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='192.168.0.104', port=5000, debug=True)
