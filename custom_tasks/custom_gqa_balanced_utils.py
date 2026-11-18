from datasets import load_dataset
import io
from PIL import Image

DS = load_dataset("lmms-lab-encoder/GQA", "val_balanced_images", split="val")
MAP = {imageId: idx for idx, imageId in enumerate(DS["id"])}


def doc_to_visual(doc):
    imageId = doc["imageId"]
    image = DS[MAP[imageId]]["image"]
    return [image.convert("RGB")]

def doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["question"].strip()
    return f"{pre_prompt}{question}{post_prompt}"
