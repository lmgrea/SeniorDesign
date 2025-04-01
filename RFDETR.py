import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# Measure inference time
start_time = time.time()

# Object Detection Model 
model = RFDETRBase()

url = r"C:/Users/lmgre/Documents/SIU/Senior Design/Hugging_Face/.venv/Scripts/testing_transformers/object_id.jpg"

image = Image.open(url)
detections = model.predict(image, threshold=0.5)

labels = []
prompt_parts = []
for class_id, confidence, (xmin, ymin, xmax, ymax) in zip(
    detections.class_id, detections.confidence, detections.xyxy
):
    label = f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    labels.append(label)  # Append label to list
    # print(f"Detected: {label} at [{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}]")

    xcoord = (xmin + xmax) / 2
    ycoord = (ymin + ymax) / 2

    segment = f"a {COCO_CLASSES[class_id]} at ({xcoord:.2f}, {ycoord:.2f})"
    prompt_parts.append(segment)

prompt = f"\nIn the image, there is " + ", and ".join(prompt_parts) + "."
print(prompt)

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

annotated_image.save("annotated_output.jpg")  # Change the filename as needed

# VLM Smol Model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
image = load_image("object_id.jpg")

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    torch_dtype=torch.bfloat16,
).to(DEVICE)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt + " Where are my scissors? What color are they?"}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=100)

end_time = time.time()
elapsed_time = end_time - start_time

generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
print(f"Inference Time: {elapsed_time:.2f} seconds")