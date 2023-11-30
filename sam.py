from contextlib import ContextDecorator
import inspect
import os
import torch
import io
import base64
import numpy as np
import cv2
import gc
from PIL import Image
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamPredictor
from mobile_sam.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

from modules import safe, devices
from modules.paths import models_path

from functools import wraps


def clear_cache():
    gc.collect()
    devices.torch_gc()


def clear_cache_decorator(func):
    @wraps(func)
    def yield_wrapper(*args, **kwargs):
        clear_cache()
        yield from func(*args, **kwargs)
        clear_cache()

    @wraps(func)
    def wrapper(*args, **kwargs):
        clear_cache()
        res = func(*args, **kwargs)
        clear_cache()
        return res

    if inspect.isgeneratorfunction(func):
        return yield_wrapper
    else:
        return wrapper
# from modules.paths import models_path
class torch_default_load_cd(ContextDecorator):
    def __init__(self):
        self.backup_load = safe.load

    def __enter__(self):
        self.backup_load = torch.load
        torch.load = safe.unsafe_torch_load
        return self

    def __exit__(self, *exc):
        torch.load = self.backup_load
        return False
    
@torch.no_grad()
@torch_default_load_cd()
# @clear_cache_decorator()
def image_predictions(image_base64):
  sam_models_dir = os.path.join(models_path, "SAM")

  checkpoint = f"{sam_models_dir}/sam_vit_h_4b8939.pth"
  model_type = "vit_h"

  # with torch_default_load_cd():
  sam = sam_model_registry[model_type](checkpoint=checkpoint)


  image_bytes = io.BytesIO(base64.b64decode(image_base64))
  image_nparray = np.array(Image.open(image_bytes))

  # image = cv2.imread(f'{sam_models_dir}/truck.jpg')
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  sam.to(device='cuda')
  predictor = SamPredictor(sam)

  print(f"START SAM")
  predictor.set_image(image_nparray)
  image_embedding = predictor.get_image_embedding().cpu().numpy()

  f = io.BytesIO()
  np.save(f, image_embedding)
  
  print(f"END SAM")
  clear_cache()
  return f

def Test():
  sam_models_dir = os.path.join("C:\stable-diffusion-webui\models", "SAM")

  checkpoint = f"{sam_models_dir}/sam_vit_h_4b8939.pth"
  model_type = "vit_h"

  with torch_default_load_cd():
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

  image = cv2.imread(f'{sam_models_dir}/truck.jpg')
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


  sam.to(device='cuda')
  predictor = SamPredictor(sam)

  predictor.set_image(image)
  image_embedding = predictor.get_image_embedding().cpu().numpy()

  return image_embedding.tolist()
