import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

# Object Detection Model 
model = RFDETRBase()

url = r"C:/Users/lmgre/Documents/SIU/Senior Design/Hugging_Face/.venv/Scripts/testing_transformers/object_id.jpg"

image = Image.open(url)
detections = model.predict(image, threshold=0.5)

labels = []
for class_id, confidence, (xmin, ymin, xmax, ymax) in zip(
    detections.class_id, detections.confidence, detections.xyxy
):
    label = f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    labels.append(label)  # Append label to list
    print(f"Detected: {label} at [{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}]")


annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)

annotated_image.save("annotated_output.jpg")  # Change the filename as needed

# VLM Smol Model

