import os
import torch
from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict

app = Flask(__name__)

# โหลดโมเดล YOLO
model_path = "C:\\Users\\user\\Desktop\\MODEL_AI\\pt_model\\best.pt"
model = YOLO(model_path)

# ฟังก์ชันในการทำนาย
def predict_image(image):
    # แปลงภาพให้เป็น numpy array
    image = image.convert("RGB")
    image = np.array(image)

    # ใช้ YOLO ทำการทำนาย
    results = model.predict(source=image, imgsz=640, conf=0.2)

    result_data = []

    # ถ้าตรวจพบวัตถุในภาพ
    if results[0].boxes:
        for det in results[0].boxes:
            class_idx = int(det.cls)
            class_name = model.names[class_idx]
            confidence = det.conf.item()
            xmin, ymin, xmax, ymax = det.xyxy[0].tolist()
         

            result_data.append(OrderedDict([
                ("class", class_name),  # กำหนดให้ class แสดงก่อน
                ("confidence", confidence),
                ("bbox", [xmin, ymin, xmax, ymax])

                
            ]))
    else:
        # ถ้าไม่มีวัตถุในภาพ ให้คืนค่า NG
        result_data.append(OrderedDict([
            ("class", "NG"),
            ("confidence", 0),
            ("bbox", [0, 0, 0, 0])
        ]))

    return result_data



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('file')  # รับไฟล์หลายไฟล์
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    all_results = []

    for file in files:
        # อ่านไฟล์ภาพจาก request
        image = Image.open(io.BytesIO(file.read()))

        # ทำการทำนาย
        result_data = predict_image(image)

     #  เพิ่มผลลัพธ์ของแต่ละภาพลงใน all_results
        all_results.append(OrderedDict([
            ("filename", file.filename),
            ("predictions", result_data)
                    
                        
            ]))

    # # ส่งคืนผลการทำนายทั้งหมด
    return jsonify(all_results)

if __name__ == '__main__':
    app.run(debug=True)
