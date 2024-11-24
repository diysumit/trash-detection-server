import json
import os
import shutil
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Paths
annotations_path = "./TACO/data/annotations.json"  # Path to annotations file
images_root_dir = "./TACO/data/"  # Root folder containing batch_X directories
output_dir = "./data/"  # Where YOLO-formatted data will be saved

# Load annotations
with open(annotations_path, "r") as f:
    data = json.load(f)

# Create YOLO directories
os.makedirs(f"{output_dir}/images/train", exist_ok=True)
os.makedirs(f"{output_dir}/images/val", exist_ok=True)
os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
os.makedirs(f"{output_dir}/labels/val", exist_ok=True)

# Map categories to indices
categories = {cat["id"]: cat["name"] for cat in data["categories"]}
category_ids = {cat_id: idx for idx, cat_id in enumerate(categories)}


def convert_bbox(size, box):
    """Convert bounding box from COCO format to YOLO format."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return x, y, w, h


# Process images and annotations
for image in tqdm(data["images"], desc="Processing images"):
    split = "train" if image["id"] % 5 != 0 else "val"  # 80% train, 20% val

    # Image source path from file_name (already includes batch folder)
    src = os.path.join(images_root_dir, image["file_name"])
    dst = os.path.join(
        output_dir, f"images/{split}/{os.path.basename(image['file_name'])}"
    )

    if os.path.exists(src):
        shutil.copy(src, dst)
        logging.info(f"Copied image to {dst}")
    else:
        logging.warning(f"Image not found: {src}")
        continue

    # Process annotations for the image
    annotations = [ann for ann in data["annotations"] if ann["image_id"] == image["id"]]
    if not annotations:
        logging.warning(f"No annotations found for image: {image['file_name']}")
        continue

    label_path = os.path.join(
        output_dir,
        f"labels/{split}/{os.path.splitext(os.path.basename(image['file_name']))[0]}.txt",
    )
    with open(label_path, "w") as label_file:
        for ann in annotations:
            bbox = convert_bbox((image["width"], image["height"]), ann["bbox"])
            class_id = category_ids[ann["category_id"]]
            label_file.write(f"{class_id} {' '.join(map(str, bbox))}\n")
            logging.info(f"Wrote label for {image['file_name']} with bbox: {bbox}")
