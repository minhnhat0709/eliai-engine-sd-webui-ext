from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules.api.models import *
from pydantic import BaseModel

EliAIEngineTxt2ImgProcessingAPI = PydanticModelGenerator(
    "EliAIEngineTxt2ImgProcessingAPI",
    StableDiffusionProcessingTxt2Img,
    [
        {"key": "sampler_index", "type": str, "default": "Euler"},
        {"key": "script_name", "type": str, "default": None},
        {"key": "script_args", "type": list, "default": []},
        {"key": "send_images", "type": bool, "default": True},
        {"key": "save_images", "type": bool, "default": False},
        {"key": "alwayson_scripts", "type": dict, "default": {}},
        {"key": "loras", "type": list, "default": []},
        {"key": "task_id", "type": str, "default": ""},
        {"key": "user_id", "type": str, "default": ""},
    ]
).generate_model()

EliAIEngineImg2ImgProcessingAPI = PydanticModelGenerator(
    "EliAIEngineImg2ImgProcessingAPI",
    StableDiffusionProcessingImg2Img,
    [
        {"key": "sampler_index", "type": str, "default": "Euler"},
        {"key": "init_images", "type": list, "default": None},
        {"key": "denoising_strength", "type": float, "default": 0.75},
        {"key": "mask", "type": str, "default": None},
        {"key": "include_init_images", "type": bool, "default": False, "exclude" : True},
        {"key": "script_name", "type": str, "default": None},
        {"key": "script_args", "type": list, "default": []},
        {"key": "send_images", "type": bool, "default": True},
        {"key": "save_images", "type": bool, "default": False},
        {"key": "alwayson_scripts", "type": dict, "default": {}},
        {"key": "loras", "type": list, "default": []},
        {"key": "task_id", "type": str, "default": ""},
        {"key": "user_id", "type": str, "default": ""},
    ]
).generate_model()


class EliAIEngineSAMPredictorAPI(BaseModel):
  image_base64: str