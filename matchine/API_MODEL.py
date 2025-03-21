import os
import torch
import cv2
import json
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict
from threading import Thread

app = Flask(__name__)

# โหลดโมเดล YOLO
model_path = 'pt_model/matchine.pt'
model = YOLO(model_path)

# ฟังก์ชันส่งผลลัพธ์กลับไปยังผู้ใช้
def send_results_to_user(results):
    print(f"📤 ส่งผลลัพธ์กลับไปยังผู้ใช้: {results}")
    # สามารถทำการส่งข้อมูลไปยังผู้ใช้ได้ที่นี่ (เช่น การส่งผ่าน WebSocket หรืออีเมล)


# ฟังก์ชันในการทำนาย
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    # ใช้ YOLO ทำการทำนาย
    results = model.predict(source=image, imgsz=640, conf=0.3)

    result_data = []
    
    if results and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
        #ถ้าไม่มีวัตถุ
        for det in results[0].boxes:
            class_idx = int(det.cls)
            class_name = model.model.names[class_idx]  # ป้องกัน KeyError
            confidence = det.conf.item()* 100
            xmin, ymin, xmax, ymax = det.xyxy[0].tolist()

            result_data.append(OrderedDict([
                ("class", class_name),
                ("confidence", f"{confidence:.2f}%"),
                ("bbox", [xmin, ymin, xmax, ymax])
            ]))
    else:
        result_data.append(OrderedDict([
            ("class", "NG"),
            ("confidence","0.00%"),
            ("bbox", [0, 0, 0, 0])
        ]))

    return result_data, results

# ฟังก์ชันหลักเพื่อประมวลผลการทำนายภาพ
def process_predictions(image_names, image_directory, save_directory):
    all_results = []

    for image_name in image_names:
        image_path = os.path.join(image_directory, image_name)

        # ทำนาย
        result_data, results = predict_image(image_path)


        save_image_with_prediction(image_name, results, save_directory, result_data)

        # เพิ่มผลลัพธ์
        all_results.append({
            "image": image_name,
            "results": result_data
        })

    # บันทึกผลลัพธ์ลงไฟล์ JSON
    result_file = os.path.join(save_directory, "prediction_results.json")
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"✅ ผลลัพธ์การทำนายถูกบันทึกที่: {result_file}")

    # ส่งผลลัพธ์กลับไปยังผู้ใช้
    send_results_to_user(all_results)



# ฟังก์ชันเพื่อบันทึกผลลัพธ์
def save_image_with_prediction(image_name, prediction, save_dir, result_data):
    jpg_dir = os.path.join(save_dir, 'jpg')
    json_dir = os.path.join(save_dir, 'json')
    txt_dir = os.path.join(save_dir, 'txt')

    for directory in [jpg_dir, json_dir, txt_dir]:
        os.makedirs(directory, exist_ok=True)

    # บันทึกภาพที่มีผลลัพธ์การทำนาย
    annotated_image = prediction[0].plot(line_width=1)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    jpg_save_path = os.path.join(jpg_dir, image_name)
    cv2.imwrite(jpg_save_path, annotated_image_rgb)

    # บันทึกผลลัพธ์เป็น TXT
    txt_save_path = os.path.join(txt_dir, image_name.replace('.jpg', '.txt'))
    with open(txt_save_path, 'w') as f:
        for item in result_data:
            f.write(f"{item}\n")

    # บันทึกผลลัพธ์เป็น JSON
    json_save_path = os.path.join(json_dir, image_name.replace('.jpg', '.json'))
    with open(json_save_path, 'w') as f:
        json.dump(result_data, f, indent=4)

    return jpg_save_path, txt_save_path, json_save_path


@app.route('/predict', methods=['POST'])
def predict():
    image_directory = r'\\10.63.85.5\image'
    save_directory = r'./predictions'

    try:
        data = request.json

        # ตรวจสอบว่า JSON ถูกต้องหรือไม่
        if 'image_names' not in data or not isinstance(data['image_names'], list):
            return jsonify({"error": "รูปแบบ JSON ไม่ถูกต้อง"}), 400

        image_names = data['image_names']
        print(f"📥 ได้รับข้อมูลแล้ว: {image_names}")

        # ตรวจสอบชื่อไฟล์ว่าใน 'image_names' มีไฟล์ที่มีอยู่ใน image_directory หรือไม่
        invalid_files = []
        for image_name in image_names:
            image_path = os.path.join(image_directory, image_name)
            if not os.path.exists(image_path):
                invalid_files.append(image_name)

        if invalid_files:
            # หากไม่พบไฟล์ ให้ส่งกลับรายชื่อไฟล์ที่ไม่พบ
            return jsonify({
                "error": "ไม่พบไฟล์ใน directory โปรดตรวจสอบอีกครั้ง",
                "missung_file": invalid_files
                }), 400


        # ส่งค่ากลับก่อนการทำนา
        response = jsonify({"message": f"📥 ได้รับข้อมูลแล้ว กำลังทำการทำนาย...."})
        response.status_code = 200

        # ทำนายใน Background
        thread = Thread(target=process_predictions, args=(image_names, image_directory, save_directory))
        thread.start()

        return response  # ส่งค่ากลับทันทีโดยไม่รอการทำนาย

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
