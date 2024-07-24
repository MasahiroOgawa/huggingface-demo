# ref: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/OneFormer/Inference_with_OneFormer.ipynb

# 1. Load image
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from transformers import AutoModelForUniversalSegmentation
from transformers import AutoProcessor
from PIL import Image
import requests
import sys

# read argument as an input image name. If there is no argument, use default.
if len(sys.argv) > 1:
    image_path = sys.argv[1]
    image = Image.open(image_path)
else:
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

# 2. Load the model
# the Auto API loads a OneFormerProcessor for us, based on the checkpoint
processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")


# 3. panoptic segmentation
# prepare image for the model
panoptic_inputs = processor(images=image, task_inputs=[
                            "panoptic"], return_tensors="pt")
for k, v in panoptic_inputs.items():
    print(k, v.shape)
processor.tokenizer.batch_decode(panoptic_inputs.task_inputs)

model = AutoModelForUniversalSegmentation.from_pretrained(
    "shi-labs/oneformer_coco_swin_large")

# Forward pass
with torch.no_grad():
    outputs = model(**panoptic_inputs)

# Visualiz
panoptic_segmentation = processor.post_process_panoptic_segmentation(
    outputs, target_sizes=[image.size[::-1]])[0]
print(panoptic_segmentation.keys())


def draw_panoptic_segmentation(segmentation, segments_info):
    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    fig, ax = plt.subplots()
    ax.imshow(segmentation)
    instances_counter = defaultdict(int)
    handles = []
    # for each segment, draw its legend
    for segment in segments_info:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=handles)
    plt.savefig('panoptic.png')


draw_panoptic_segmentation(**panoptic_segmentation)


# 4. Inference: semantic segmentation
# prepare image for the model
semantic_inputs = processor(images=image, task_inputs=[
                            "semantic"], return_tensors="pt")
for k, v in semantic_inputs.items():
    print(k, v.shape)

# forward pass
with torch.no_grad():
    outputs = model(**semantic_inputs)

semantic_segmentation = processor.post_process_semantic_segmentation(outputs)[
    0]
semantic_segmentation.shape


def draw_semantic_segmentation(segmentation):
    # return if segmentation is empty
    if torch.max(segmentation) == 0:
        return

    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    # get all the unique numbers
    labels_ids = torch.unique(segmentation).tolist()
    fig, ax = plt.subplots()
    ax.imshow(segmentation)
    handles = []
    for label_id in labels_ids:
        label = model.config.id2label[label_id]
        color = viridis(label_id)
        handles.append(mpatches.Patch(color=color, label=label))
    ax.legend(handles=handles)
    plt.savefig('semantic.png')


draw_semantic_segmentation(semantic_segmentation)

# 5. Inference: instance segmentation
# prepare image for the model
instance_inputs = processor(images=image, task_inputs=[
                            "instance"], return_tensors="pt")
for k, v in instance_inputs.items():
    print(k, v.shape)

# forward pass
with torch.no_grad():
    outputs = model(**instance_inputs)
instance_segmentation = processor.post_process_instance_segmentation(outputs)[
    0]
instance_segmentation.keys()


def draw_instance_segmentation(segmentation, segments_info):
    # get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    fig, ax = plt.subplots()
    ax.imshow(segmentation)
    instances_counter = defaultdict(int)
    handles = []
    # for each segment, draw its legend
    for segment in segments_info:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=handles)
    plt.savefig('instance.png')


draw_instance_segmentation(**instance_segmentation)
