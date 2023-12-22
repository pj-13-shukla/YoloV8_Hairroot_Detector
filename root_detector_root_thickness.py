from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image
import json
import random
import cv2
import numpy as np

app = Flask(__name__)

# Define the visualization function
def visualize_boxes_on_image(image, boxes, colors):
    for box in boxes:
        x1, y1, x2, y2, color = box
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[color], thickness=2)
    return image


@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler of /detect POST endpoint
    Receives uploaded file with a name "image_file", 
    passes it through YOLOv8 object detection 
    network and returns an array of bounding boxes.
    :return: a JSON array of objects bounding 
    boxes in format 
    [[x1, y1, x2, y2, color],..]
    """
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(buf)
    return jsonify(boxes)    


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format 
    [[x1, y1, x2, y2, color],..]
    """
    model = YOLO("best_root_thickness.pt")
    
    # Get names and colors for different objects
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = {
        'thick': (0, 255, 0),  # Green for 'thick'
        'medium': (255, 255, 0),  # Yellow for 'medium'
        'thin': (255, 0, 0)  # Red for 'thin'
    }

    # Read image using PIL
    img_pil = Image.open(buf)
    img_np = np.array(img_pil)

    results = model.predict(img_np)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        
        # Determine the object type based on class_id
        object_type = result.names[class_id]
        
        # Assign colors based on object type
        color = 'thin'  # Default color for 'thin'
        if object_type == 'thick':
            color = 'thick'
        elif object_type == 'medium':
            color = 'medium'
        
        output.append([
          x1, y1, x2, y2, color
        ])
    
    # Visualize the bounding boxes on the image
    image = visualize_boxes_on_image(img_np, output, colors)
    
    # Convert the image back to bytes and return it as a response
    _, img_encoded = cv2.imencode('.jpg', image)
    response = img_encoded.tobytes()
    return response

serve(app, host='0.0.0.0', port=8080)
