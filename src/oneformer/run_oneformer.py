from oneformersegmentator import OneFormerSegmentator
import sys
from PIL import Image
import requests

# run below sample if this file is called as main
if __name__ == "__main__":
    # read argument as an input image name. If there is no argument, use default.
    if len(sys.argv) > 1:
        # print help message
        if sys.argv[1] == "-h" or sys.argv[1] == "--help":
            print("Usage: python run_oneformer.py [your image_path]")
            print("If no image_path is given, a default image is used.")
            sys.exit(0)
        else:
            image_path = sys.argv[1]
            image = Image.open(image_path)
    else:
        url = "https://huggingface.co/datasets/shi-labs/oneformer_demo/resolve/main/ade20k.jpeg"
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        image = Image.open(response.raw)

    # run segmentation
    model_name = "shi-labs/oneformer_coco_swin_large"
    task_type = "panoptic"
    oneformer = OneFormerSegmentator(model_name, task_type)
    predicted_map, segments_info = oneformer.inference(
        image)
    oneformer.print_debug_info()
    oneformer.show()
