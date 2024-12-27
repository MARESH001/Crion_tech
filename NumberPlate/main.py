import json
import cv2
from ultralytics import YOLOv10
import numpy as np
import math
import re
import os
from datetime import datetime
from paddleocr import PaddleOCR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize the YOLOv10 Model
model = YOLOv10("weights/best.pt")
# Class Names
className = ["License"]
# Initialize the Paddle OCR
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

def paddle_ocr(frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1:x2]
    result = ocr.ocr(frame, det=False, rec=True, cls=False)
    text = ""
    for r in result:
        scores = r[0][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores * 100)
        if scores > 60:
            text = r[0][0]
    pattern = re.compile('[\W]')
    text = pattern.sub('', text)
    text = text.replace("???", "")
    text = text.replace("O", "0")
    text = text.replace("粤", "")
    return str(text)

def save_json(license_plates, image_name):
    # Generate JSON data for the image
    image_data = {
        "Image Name": image_name,
        "License Plates": list(license_plates)
    }
    for i in license_plates:
        print(i)
    # Save the JSON file in the same directory as the input image
    json_file_path = os.path.splitext(image_name)[0] + f"_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(json_file_path, 'w') as f:
        json.dump(image_data, f, indent=2)

# Process each image
image_file = r'/Users/lalitm/Downloads/22_0001_DSC09985.jpg'

if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
    print(f"Processing {image_file}")

    # Read the image
    frame = cv2.imread(image_file)
    license_plates = set()

    # Run YOLO model prediction
    results = model.predict(frame, conf=0.2)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            classNameInt = int(box.cls[0])
            clsName = className[classNameInt]
            conf = math.ceil(box.conf[0] * 100) / 100
            label = paddle_ocr(frame, x1, y1, x2, y2)
            if label:
                license_plates.add(label)
            textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
            c2 = x1 + textSize[0], y1 - textSize[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    # Save JSON and processed image in the same directory
    save_json(license_plates, image_file)
    output_image_path = os.path.splitext(image_file)[0] + "_processed.png"
    cv2.imwrite(output_image_path, frame)

    print(f"Processed {image_file} and saved results.")

print("Processing complete!")
