from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules.api.models import *
from pydantic import BaseModel

EliAIEngineTxt2ImgProcessingAPI = PydanticModelGenerator(
    "EliAIEngineTxt2ImgProcessingAPI",
    StableDiffusionProcessingTxt2Img,
    [
        {"key": "task_id", "type": str, "default": ""},
        {"key": "sampler_index", "type": str, "default": "Euler"},
        {"key": "script_name", "type": str, "default": None},
        {"key": "script_args", "type": list, "default": []},
        {"key": "send_images", "type": bool, "default": True},
        {"key": "save_images", "type": bool, "default": False},
        {"key": "alwayson_scripts", "type": dict, "default": {}},
    ]
).generate_model()

class EliAIEngineSAMPredictorAPI(BaseModel):
  image_base64: str