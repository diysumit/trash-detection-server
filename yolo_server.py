#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from ultralytics import YOLO
import cv2
import numpy as np
import os
import logging

app = Flask(__name__)
# use any random key
app.secret_key = "db4251506e4a2b47475a1b885ec53e81"
app.template_folder = "templates"

# Load the YOLO model "runs/detect/train/weights/best.pt"
# using default model the trained model won't work for now
model_path = os.getenv(
    "YOLO_MODEL_PATH", "./yolo11n.pt"
)  # Default to 'yolo11n.pt'
# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Log level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("app.log"),  # Log to file
    ],
)
logging.info(f"Loading YOLO model from: {model_path}")
model = YOLO(model_path)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect_objects():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["file"]
        np_img = np.frombuffer(file.read(), np.uint8)
        logging.info(f"The n-dim image array: {np_img}")
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img, (640, 640))  # Resize to 640x640
        results = model(resized_img, conf=0.25)[0] # setting confidence too low for model to detect
        logging.info(f"Results of model: {results}")
      #   # Extract image and bounding boxes (some c++ exception error on mac)
      #   for result in results:
      #       image = result.orig_img
      #       boxes = result.boxes  # Bounding box information
      #       names = result.names  # Class names

      #       # Draw boxes on the image
      #       if boxes is not None:
      #             for box in boxes.data:
      #                   x1, y1, x2, y2, confidence, class_idx = box[:6]
      #                   class_name = names[int(class_idx)]
      #                   cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
      #                   cv2.putText(image, f"{class_name} {confidence:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      #       # Display the image
      #       cv2.imshow("Detections", image)
      #       cv2.waitKey(0)
      #       cv2.destroyAllWindows()
        logging.info(f"Result Datatype: {type(results)}")
        logging.info(f"{results.boxes}")
        detections = results.boxes.data.cpu().numpy()
        logging.info(f"Detections: {detections}")
        output = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            output.append({
                "class": model.names[int(cls)],
                "confidence": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })
      
        logging.info(f"Loading YOLO model from: {model_path}")

        session["detections"] = output
        # Return JSON response
        # return jsonify({"detections": output})
        # Redirect to results
        return redirect(url_for("results"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/results", methods=["GET"])
def results():
    detections = session.get("detections", [])
    return render_template("results.html", detections=detections)


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(host="0.0.0.0", port=8000)
