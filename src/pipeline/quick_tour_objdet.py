import requests
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

# Download an image with cute cats
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
image_data = requests.get(url, stream=True).raw
image = Image.open(image_data)

# Allocate a pipeline for object detection
object_detector = pipeline('object-detection')
result = object_detector(image)

# display result image
# Copy the original image
draw_image = image.copy()
draw = ImageDraw.Draw(draw_image)

# Draw each result on the image
for obj in result:
    # Extract bounding box and label
    box = obj['box']
    label = obj['label']

    # Draw rectangle
    draw.rectangle([(box['xmin'], box['ymin']),
                   (box['xmax'], box['ymax'])], outline="yellow", width=3)

    # Draw label
    fontsize = 20
    font = ImageFont.truetype("arial.ttf", fontsize)
    draw.text((box['xmin'] + 10, box['ymin'] + 10),
              f"{label}", fill="yellow", font=font)

# Display the image with detections
draw_image.show()
