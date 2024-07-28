# ref: https://huggingface.co/learn/computer-vision-course/en/unit3/vision-transformers/oneformer
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch
import sys


def run_segmentation(image, task_type="panoptic", model_name="shi-labs/oneformer_ade20k_dinat_large"):
    """Performs image segmentation based on the given task type.

    Args:
        image (PIL.Image): The input image.
        task_type (str): The type of segmentation to perform ('semantic', 'instance', or 'panoptic').

    Returns:
        PIL.Image: The segmented image.

    Raises:
        ValueError: If the task type is invalid.
    """
    processor = OneFormerProcessor.from_pretrained(
        model_name
    )  # Load once here
    model = OneFormerForUniversalSegmentation.from_pretrained(
        model_name
    )

    if task_type == "semantic":
        inputs = processor(images=image, task_inputs=[
                           "semantic"], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

    elif task_type == "instance":
        inputs = processor(images=image, task_inputs=[
                           "instance"], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = processor.post_process_instance_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        predicted_map = prediction["segmentation"]
        segments_info = prediction["segments_info"]

    elif task_type == "panoptic":
        inputs = processor(images=image, task_inputs=[
                           "panoptic"], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        predicted_map = prediction["segmentation"]
        segments_info = prediction["segments_info"]

    else:
        raise ValueError(
            "Invalid task type. Choose from 'semantic', 'instance', or 'panoptic'"
        )

    # debug
    print(f"predicted_map = {predicted_map}")
    print(f"segments_info = {segments_info}")
    for segment in segments_info:
        print(f"segment id = {segment["id"]}")
        print(f"segment label id = {segment["label_id"]}")
        label = model.config.id2label[id['label_id']]
        print(f"label = {label}")

    return predicted_map


def show_image_comparison(image, predicted_map, segmentation_title):
    """Displays the original image and the segmented image side-by-side.

    Args:
        image (PIL.Image): The original image.
        predicted_map (PIL.Image): The segmented image.
        segmentation_title (str): The title for the segmented image.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_map)
    plt.title(segmentation_title + " Segmentation")
    plt.axis("off")
    plt.savefig("oneformer_segm.png")
    plt.show()


# run below sample if this file is called as main
if __name__ == "__main__":
    # read argument as an input image name. If there is no argument, use default.
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = Image.open(image_path)
    else:
        url = "https://huggingface.co/datasets/shi-labs/oneformer_demo/resolve/main/ade20k.jpeg"
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        image = Image.open(response.raw)

    # run segmentation
    model_name = "shi-labs/oneformer_ade20k_swin_tiny"
    # model_name = "shi-labs/oneformer_coco_swin_large"
    # model_name = "shi-labs/oneformer_ade20k_dinat_large"
    task_to_run = "panoptic"
    predicted_map = run_segmentation(image, task_to_run, model_name)
    show_image_comparison(image, predicted_map, task_to_run)
