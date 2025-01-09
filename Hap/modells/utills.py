import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.colors import CSS4_COLORS, to_rgb
from easyocr import Reader
from paddleocr import PaddleOCR
import difflib
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from yolov5.detect import run

image_path = r"C:/CLONOS/Hap1 (4)/Hap1/Hap/output_image.jpg"

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from yolov5.detect import run

image_path = r"C:/CLONOS/Hap1 (4)/Hap1/Hap/output_image.jpg"

def detection_results(image_path):
    """
    Run YOLOv5 detection on the given image using the existing run function.
    """
    try:
        detection_opts = {
            'weights': r'C:\CLONOS\Hap1 (4)\Hap1\Hap\yolov5\runs\train\weights\best.pt',
            'source': image_path,
            'data': r'C:\CLONOS\Hap1 (4)\Hap1\Hap\yolov5\data\data.yaml',
            'imgsz': 640,
            'conf_thres': 0.25,
            'iou_thres': 0.45,
            'max_det': 1000,
            'device': 'cpu',
            'view_img': False,
            'save_txt': False,
            'save_format': 'jpg',
            'save_conf': False,
            'save_crop': False,
            'save_csv': False,
            'nosave': True,
            'classes': None,
            'agnostic_nms': False,
            'augment': False,
            'visualize': False,
            'update': False,
            'project': 'runs/detect',
            'name': 'exp',
            'exist_ok': False,
            'line_thickness': 3,
            'hide_labels': False,
            'hide_conf': False,
            'half': False,
            'dnn': False,
            'vid_stride': 1,
        }
        results = run(**detection_opts)
        
        # Log the results object to inspect its structure
        print(f"YOLOv5 raw results: {results}")
        
        if not results or len(results) == 0:
            print("No results returned from YOLOv5")
            return None
        
        return results
    except Exception as e:
        print(f"Error in detection: {str(e)}")
        return None

def process_type_of_milk(image_path):
    """
    Process the image to detect milk types and return the category (FCM/SM).
    Returns the category of the highest confidence detection.
    """
    try:
        # Get detections from YOLOv5
        results = detection_results(image_path)
        if not results or not results[0]:
            print("No detections found")
            return None

        # Class mapping to milk types
        class_mapping = {
            3: "500ml Orange",  # 500ml Orange
            1: "250ml Orange",  # 250ml Orange
            2: "500ml Green",   # 500ml Green
            0: "1L Green"       # 1L Green
        }

        # Category mapping based on milk type
        category_mapping = {
            "500ml Orange": "FCM",
            "250ml Orange": "FCM",
            "500ml Green": "SM",
            "1L Green": "SM"
        }

        # Iterate over the results (likely a list of detections)
        highest_conf = 0
        best_detection = None

        # Check YOLOv5 Results to ensure proper handling of data structure
        for detection in results:  # Each detection is a dictionary
            print(f"Detection Result: {detection}")  # Print each detection result
            conf = detection.get('confidence', 0)  # Get confidence from the detection
            class_name = detection.get('class_name', '')  # Get class name from the detection

            # If the detection has a confidence value and is a valid class name
            if conf > highest_conf and class_name:
                highest_conf = conf
                best_detection = class_name

        if best_detection:
            # Now map the best detection to milk type category
            category = category_mapping.get(best_detection, "Unknown")
            print(f"Selected {best_detection} with category {category} (confidence: {highest_conf:.2f})")
            return category  # Return just the category string (FCM or SM)

        print("No valid detections found")
        return None
    except Exception as e:
        print(f"Error in processing type of milk: {str(e)}")
        return None





def process_vehicle_number(image_np):
    reader = PaddleOCR(use_gpu=False)
    result = reader.ocr(image_np)
    if result:
        return result[0][1][0]  # Assuming this is the correct format
    return None

def process_batch_code(image_np):
    reader = PaddleOCR(use_gpu=False)
    result = reader.ocr(image_np)
    if result:
        return result[0][1][0]  # Assuming this is the correct format
    return None

def process_sku(image_np):
    reader = Reader(['en'])
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    print("Gray image shape:", gray_image.shape)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_image = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    # Binarize the image
    binarized_image = cv2.adaptiveThreshold(
        morph_image, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        15, 
        4 
    )
    
    print("Binarized image created.")

    # Save the binarized image for inspection
    cv2.imwrite("debug_binarized_image.jpg", binarized_image)
    print("Saved binarized image for inspection.")

    # Perform OCR 
    results = reader.readtext(binarized_image)
    
    print("OCR Results:", results)
    
    if not results:
        print("No text detected by OCR.")
    
    sku_mappings = {
        "500": ['500ml', '500', '5', '50'],
        "750": ['750ml', '750'],
        "1000": ['1L', '1', '1 L'],
        "2000": ['2L', '2', '2 L']
    }
    
    for _, text, _ in results:
        print(f"Detected Text: {text.strip()}")
        
        for key, values in sku_mappings.items():
            if any(val in text.strip() for val in values):  
                print(f"Matched SKU: {key}")
                return key
    
    print("No matching SKU found.")
    
    return None

async def process_hand_gesture(image_path):
    image_np = cv2.imread(image_path)
    
    print("Processing Hand Gesture")
    
    if image_np is None:
        print("No valid image data available.")
        return None
    
    try:
        img = image_np.copy()
        detector = HandDetector(maxHands=1)
        classifier = Classifier("C:/CLONOS/Hap1 (4)/Hap1/converted_keras/keras_model.h5", "C:/CLONOS/Hap1 (4)/Hap1/converted_keras/labels.txt")
        
        offset = 20 
        imgSize = 300 
        labels = ["pin_hole", "horizontal_leak", "vertical_leak"]
        
        hands, img = detector.findHands(img)

        if not hands:
            print("No hands detected")
            return None
        
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        try:
            imgCrop = img[max(0, y - offset):min(img.shape[0], y + h + offset), 
                         max(0, x - offset):min(img.shape[1], x + w + offset)]
        except Exception as e:
            print(f"Cropping error: {str(e)}")
            return None
        
        if imgCrop.size == 0:
            print("Invalid crop dimensions")
            return None
        
        aspectRatio = h / w
        
        try:
            if aspectRatio > 1:
                k = imgSize / h 
                wCal = math.ceil(k * w) 
                imgResize = cv2.resize(imgCrop, (wCal, imgSize)) 
                wGap = math.ceil((imgSize - wCal) / 2) 
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w 
                hCal = math.ceil(k * h) 
                imgResize = cv2.resize(imgCrop, (imgSize, hCal)) 
                hGap = math.ceil((imgSize - hCal) / 2) 
                imgWhite[hGap:hCal + hGap, :] = imgResize
        except Exception as e:
            print(f"Resize error: {str(e)}")
            return None
        
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        
        if index is None or index >= len(labels):
            print("Invalid prediction index")  
            return None
        
        result = labels[index]
        print(f"Predicted: {result}")
        
        result_mapping = {
            "vertical_leak": "Vertical",
            "horizontal_leak": "Horizontal",
            "pin_hole": "Pin Hole"
        }
        
        return result_mapping.get(result)

    except Exception as e:
        print(f"Error in process_hand_gesture: {str(e)}")
        return None

