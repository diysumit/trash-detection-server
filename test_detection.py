#!/usr/bin/env python3

from ultralytics import YOLO
import cv2

# Load the YOLO model "./runs/detect/train/weights/best.pt"
model_path = "./yolo11n.pt"  # Replace with your model's path if needed
model = YOLO(model_path)

# Path to the input image
image_path = "./test_image.jpg"  # Replace with your image path

# Load and prepare the image
image = cv2.imread(image_path)
if image is None:
    print(f"Failed to load image: {image_path}")
    exit(1)

# Perform object detection
resized_img = cv2.resize(image, (640, 640))
results = model(resized_img, conf=0.25)
print(f"Number of detections: {len(results[0].boxes.data) if results[0].boxes is not None else 0}")

# Process and print the results
for result in results:
    boxes = result.boxes  # Bounding box information
    names = result.names  # Class names

    print("Detected Objects:")
    if boxes is not None:
        for box in boxes.data:
            x1, y1, x2, y2, confidence, class_idx = box[:6]
            class_name = names[int(class_idx)]
            print(
                f"Class: {class_name}, Confidence: {confidence:.2f}, "
                f"BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"
            )
    else:
        print("No objects detected.")

# Optionally display the image with bounding boxes
for result in results:
    image = result.orig_img
    if result.boxes is not None:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, class_idx = box[:6]
            class_name = result.names[int(class_idx)]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{class_name} {confidence:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

# Display the image (optional)
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
