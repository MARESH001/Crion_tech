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


''' def process_type_of_milk(image_np):
    def closest_color(requested_color):
        min_colors = {
            sum((int(255 * c) - rc) ** 2 for c, rc in zip(to_rgb(hex_val), requested_color)): name
            for name, hex_val in CSS4_COLORS.items()
        }
        return min_colors[min(min_colors.keys())]

    if image_np is None:
        print("No valid image data available.")
        return None

    # Resize image to improve processing speed
    image_resized = cv2.resize(image_np, (200, 200), interpolation=cv2.INTER_AREA)
    pixels = image_resized.reshape(-1, 3)

    # Perform KMeans clustering to find dominant colors
    k = 10  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    # Extract dominant colors
    dominant_colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    proportions = counts / counts.sum()

    # Print all detected colors and their proportions
    print("Detected Colors and Their Proportions:")
    for color, proportion in zip(dominant_colors, proportions):
        color_name = closest_color(color)
        print(f"Color: {color_name}, RGB: {color}, Proportion: {proportion:.4f}")

    # Sort dominant colors based on their proportion and return the top two
    sorted_indices = np.argsort(proportions)[::-1][:2]
    top_two_colors = dominant_colors[sorted_indices]
    top_two_color_names = [closest_color(color) for color in top_two_colors]

    # Color mapping dictionary
    color_mapping = {
        'orange': "FCM",
        'darkorange': "FCM",
        'orangered': "FCM",
	'saddlebrown': "FCM",
	'sienna': "FCM",
        'green': "SM",
        'darkgreen': "SM",
        'limegreen': "SM",
        'blue': "TM",
        'deepskyblue': "TM",
        "steelblue": "TM",
        'lightblue': "TM",
        'lightgreen': "CM",
        'palegreen': "CM"
    }

    # Match and return the color type based on dominant colors
    print("\nTop Two Dominant Colors:")
    for color, name in zip(top_two_colors, top_two_color_names):
        if name == 'white':
            continue
        matched_name = next((key for key in color_mapping if key == name), None)
        if matched_name:
            print(f"{name}: {color_mapping[matched_name]}")
            return color_mapping[matched_name]  # Return the detected color type

    return None '''


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
        morph_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
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
        return None

    # SKU mapping
    sku_mappings = {
        "500": ['500ml', '500', '5', '50'],
        "750": ['750ml', '750'],
        "1000": ['1L', '1', '1 L'],
        "2000": ['2L', '2', '2 L']
    }

    for _, text, _ in results:
        print(f"Detected Text: {text.strip()}")

        # Check for SKU matches
        for key, values in sku_mappings.items():
            if any(val in text.strip() for val in values):  # Check if any SKU value is in the detected text
                print(f"Matched SKU: {key}")
                return key

    print("No matching SKU found.")
    return None


'''async def process_hand_gesture(image_np=None):
    print("Processing Hand Gesture")
    if image_np is None:
        print("No valid image data available.")
        return None

    try:
        img = image_np.copy()
        imgOutput = img.copy()

        detector = HandDetector(maxHands=1)
        classifier = Classifier("converted_keras/keras_model.h5", "converted_keras/labels.txt")

        # Variables for image cropping and resizing
        offset = 20
        imgSize = 300
        labels = ["pin_hole", "horizontal_leak", "vertical_leak"]

        # Check for hand detection
        hands, img = detector.findHands(img)

        # Final prediction variable
        final_prediction = None

        # Check if hands are detected
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # Ensure valid cropping dimensions
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgCropShape = imgCrop.shape
                aspectRatio = h / w

                # If the aspect ratio is more vertical
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

                # Perform prediction
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                final_prediction = labels[index]

        print(final_prediction)

        # Print the final prediction
        if final_prediction == "vertical_leak":
            print("Vertical Leak")
            return "Vertical"
        elif final_prediction == "horizontal_leak":
            print("Horizontal Leak")
            return "Horizontal"
        elif final_prediction == "pin_hole":
            print("Pin Hole")
            return "Pin Hole"
        else:
            print("Take Again")
            return None

    except Exception as e:
        print(f"Error in process_hand_gesture: {str(e)}")
        return None'''


def process_type_of_milk(image_np):
    def closest_color(requested_color):
        min_colors = {
            sum((int(255 * c) - rc) ** 2 for c, rc in zip(to_rgb(hex_val), requested_color)): name
            for name, hex_val in CSS4_COLORS.items()
        }
        return min_colors[min(min_colors.keys())]

    if image_np is None:
        print("No valid image data available.")
        return None

    try:
        # Resize image to improve processing speed
        image_resized = cv2.resize(image_np, (200, 200), interpolation=cv2.INTER_AREA)
        pixels = image_resized.reshape(-1, 3)

        # Perform KMeans clustering to find dominant colors
        k = 10  # Number of clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pixels)

        # Extract dominant colors
        dominant_colors = kmeans.cluster_centers_.astype(int)
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        proportions = counts / counts.sum()

        # Print all detected colors and their proportions
        print("Detected Colors and Their Proportions:")
        for color, proportion in zip(dominant_colors, proportions):
            color_name = closest_color(color)
            print(f"Color: {color_name}, RGB: {color}, Proportion: {proportion:.4f}")

        # Sort dominant colors based on their proportion and return the top two
        sorted_indices = np.argsort(proportions)[::-1][:2]
        top_two_colors = dominant_colors[sorted_indices]
        top_two_color_names = [closest_color(color) for color in top_two_colors]

        # Updated color mapping dictionary with all possible options
        color_mapping = {
            'orange': "FCM",
            'darkorange': "FCM",
            'orangered': "FCM",
            'coral': "FCM",
            'green': "SM",
            'darkgreen': "SM",
            'limegreen': "SM",
            'forestgreen': "SM",
            'blue': "TM",
            'deepskyblue': "TM",
            'steelblue': "TM",
            'lightblue': "TM",
            'cornflowerblue': "TM",
            'lightgreen': "CM",
            'palegreen': "CM",
            'mediumspringgreen': "CM"
        }

        # Match and return the color type based on dominant colors
        print("\nTop Two Dominant Colors:")
        for color, name in zip(top_two_colors, top_two_color_names):
            if name == 'white':
                continue
            matched_name = next((key for key in color_mapping if key == name.lower()), None)
            if matched_name:
                result = color_mapping[matched_name]
                print(f"{name}: {result}")
                return result

        return None

    except Exception as e:
        print(f"Error in process_type_of_milk: {str(e)}")
        return None
image_path = r"C:/CLONOS/Hap1 (2)/Hap1/Hap/output_image.jpg"
async def process_hand_gesture(image_path):
    image_np = cv2.imread(image_path)
    print("Processing Hand Gesture")
    if image_np is None:
        print("No valid image data available.")
        return None

    try:
        img = image_np.copy()
        
        # Initialize detector and classifier
        detector = HandDetector(maxHands=1)
        classifier = Classifier("converted_keras/keras_model.h5", "converted_keras/labels.txt")

        # Variables for image cropping and resizing
        offset = 20
        imgSize = 300
        labels = ["pin_hole", "horizontal_leak", "vertical_leak"]

        # Check for hand detection
        hands, img = detector.findHands(img)
        
        if not hands:
            print("No hands detected")
            return None

        # Process detected hand
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Create white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Ensure safe cropping
        try:
            imgCrop = img[max(0, y - offset):min(img.shape[0], y + h + offset), 
                         max(0, x - offset):min(img.shape[1], x + w + offset)]
        except Exception as e:
            print(f"Cropping error: {str(e)}")
            return None

        if imgCrop.size == 0:
            print("Invalid crop dimensions")
            return None

        # Calculate aspect ratio and resize
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

        # Perform prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        
        if index is None or index >= len(labels):
            print("Invalid prediction index")
            return None
            
        result = labels[index]
        print(f"Predicted: {result}")

        # Map the result to the expected format
        result_mapping = {
            "vertical_leak": "Vertical",
            "horizontal_leak": "Horizontal",
            "pin_hole": "Pin Hole"
        }

        return result_mapping.get(result)

    except Exception as e:
        print(f"Error in process_hand_gesture: {str(e)}")
        return None


