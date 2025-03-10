import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict  # เพิ่มการนำเข้า OrderedDict
from ultralytics import YOLO

# กำหนดพาธของโมเดล
model_path = "C:\\Users\\traineeit\\Desktop\\API_AI\\pt_model\\best.pt"
model = YOLO(model_path)

# ฟังก์ชันในการทำนาย
def predict_image(image):
    # แปลงภาพให้เป็น numpy array
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # ใช้ OpenCV แปลงภาพ
    image = np.array(image)

    # ใช้ YOLO ทำการทำนาย
    results = model.predict(source=image, imgsz=640, conf=0.5)

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

# ฟังก์ชันเพื่อบันทึกรูปภาพและผลลัพธ์
def save_image_with_prediction(image, prediction, save_dir, image_name, result_data):
    #สร้างโฟรเดอร์สำหรับเก็บค่าของนามสกุลไฟล์ต่างๆ
    jpg_dir = os.path.join(save_dir, 'jpg')
    json_dir = os.path.join(save_dir, 'json')
    txt_dir = os.path.join(save_dir, 'txt')

    for directory in [jpg_dir,json_dir,txt_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    #บันทึกภาพที่ทำนาย
    annotated_image = prediction[0].plot(line_width=1)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    jpg_save_path = os.path.join(jpg_dir, image_name)
    cv2.imwrite(jpg_save_path, annotated_image_rgb)
    print(f"✅ Image saved to {jpg_save_path}")

    txt_save_path = os.path.join(save_dir, image_name.replace('.jpg', '.txt'))
    with open(txt_save_path, 'w') as f:
        for item in result_data:
            f.write(f"{item}\n")
    print(f"✅ Results saved to {txt_save_path}")

    json_save_path = os.path.join(save_dir, image_name.replace('.jpg', '.json'))
    with open(json_save_path, 'w') as f:
        json.dump(result_data, f, indent=4)
    print(f"✅ Results saved to {json_save_path}")

def find_files_by_name(directory):
    files = {}
    for ext in ['jpg' , 'json','txt']:
        files[ext] = []
        search_path = os.path.join(directory, ext)
        if os.path.exists(search_path):
            for file in os.listdir(search_path):
                if name in file:
                    file[ext].append(os.path.join(search_path, file))
    return file 

if __name__ == '__main__':
    # กำหนดชื่อไฟล์ภาพที่ต้องการทดสอบ
    test_image_name = "20240313_232239.jpg"

    # กำหนดพาธของภาพและโฟลเดอร์บันทึกผล
    data_path = r'\\10.63.85.5\image'  # พาธเฉพาะของโฟลเดอร์ภาพ
    save_dir = 'C:\\Users\\traineeit\\Desktop\\API_AI\\test_results\\test'

    # สร้างพาธเต็มของภาพ
    image_path = os.path.join(data_path, test_image_name)

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
    else:
        print(f"✅ Found image: {image_path}")

        # โหลดภาพเพื่อทำการทำนาย
        image = cv2.imread(image_path)

        # ทำนายผลลัพธ์
        results = model.predict(source=image_path, imgsz=640, conf=0.5)

        # เตรียมข้อมูลสำหรับผลลัพธ์
        result_data = predict_image(image)

        # แสดงผลลัพธ์
        annotated_image = results[0].plot(line_width=1)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 8))
        plt.imshow(annotated_image_rgb)
        plt.axis('off')
        plt.title(f"Prediction for {test_image_name}")
        plt.show()

        # บันทึกผลลัพธ์
        save_image_with_prediction(image, results, save_dir, test_image_name, result_data)
