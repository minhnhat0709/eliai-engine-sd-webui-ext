import datetime
import threading
import time
import asyncio
import os
from typing import List

import requests

from supabase_client import supabase
import boto3

from automapper import mapper

import numpy as np
from fastapi import FastAPI, Body, BackgroundTasks, File, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.concurrency import run_in_threadpool
from PIL import Image

import gradio as gr

from modules.api.models import *
from modules.api import api, models
from modules import scripts
from modules.shared import opts, cmd_opts

from modules.api.api import Api
from modules.call_queue import queue_lock

from scripts.models import EliAIEngineImageCaptionAPI, EliAIEngineSAMPredictorAPI, EliAIEngineDownloadLocalLoraAPI, EliAIEngineTxt2ImgProcessingAPI, EliAIEngineImg2ImgProcessingAPI, EliAIEngineExtraAPI
from outside_lora_process import load_loras
from task_queue import runQueue
from sam import image_predictions
from blip import generate_captions

from PIL import Image

import base64
import io
import json
from pathlib import Path


lora_dir = Path(cmd_opts.lora_dir).resolve()

def save_base64_to_file(base64String, filename):
    # Decode the Base64 data
    binary_data = base64.b64decode(base64String)

    # Specify the file path and name where you want to save the image
    file_path = "./outputs/engine_temp/" + filename + ".png"

    # Save the binary data to a file
    with open(file_path, "wb") as image_file:
        image_file.write(binary_data)

def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return api.encode_pil_to_base64(image)
    elif type(image) is np.ndarray:
        return encode_np_to_base64(image)
    else:
        return ""


def encode_np_to_base64(image):
    pil = Image.fromarray(image)
    return api.encode_pil_to_base64(pil)


s3client = boto3.client('s3', endpoint_url= os.environ.get('AWS_ENDPOINT') or 'https://hn.ss.bfcplatform.vn',
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID') or "1XHPESY0IXGEMWYKN6PL",
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY') or "Kl8vL45TnD9KmVBcn2siOj9tno6rOimIxZlITSr1")
bucket_name = os.environ.get('BUCKET_NAME') or "eliai-server"
server_domain = os.environ.get('STORAGE_DOMAIN') or "https://eliai-server.hn.ss.bfcplatform.vn/"


def s3Storage_base64_upload(base64_image: str, task_id: str, index: int):
    image_binary = base64.b64decode(base64_image)
    object_key = f"images/{task_id}/{task_id}_{index}.jpg"

    s3client.upload_fileobj(
      Fileobj=io.BytesIO(image_binary),
      Bucket=bucket_name,
      Key=object_key,
      ExtraArgs={'ACL': 'public-read'}  # Optional: Set ACL to make the image public
    )

    return server_domain + object_key



def image_uploading(images: List[str], seed:int, task_id:   str, user_id: str):
    # time.sleep(10)
    result = []
    for index, image in enumerate(images):
        image_url = s3Storage_base64_upload(image, task_id, index)
        result.append(image_url)
        supabase.table("Images").insert({
            "image_url": image_url,
            "is_shared": False,
            "seed": seed,
            "task_id": task_id,
            "user_id": user_id
        }).execute()
    

    if len(result) == 0:
        supabase.table("Tasks").update({
            "status": "failed",
            "finished_at": datetime.datetime.utcnow().isoformat()
        }).eq("task_id", task_id).execute()
    else:
        supabase.table("Tasks").update({
            "status": "done",
            "finished_at": datetime.datetime.utcnow().isoformat()
        }).eq("task_id", task_id).execute()
    
def is_tile_controlnet(args):
   if "tile" in args.get("model", ""):
      return False
   return True

def eliai_engine_api(_: gr.Blocks, app: FastAPI):
    api = Api(app, queue_lock)
    opts.samples_format = 'jpg'
    
    @app.get("/ping", status_code=200)
    def ping():
        return
    @app.post("/upload-lora")
    def upload(url: EliAIEngineDownloadLocalLoraAPI):
        try:
            response = requests.get(url.lora_url)
            if response.status_code == 200:
              print(Path.joinpath(lora_dir, "Local", url.file_name))
              with open(Path.joinpath(lora_dir, "Local", url.file_name), 'wb') as f:
                  f.write(response.content)
        except Exception:
            return {"message": "There was an error uploading the file"}

        return {"message": f"Successfully Download lora {url.file_name}"}
    @app.post("/eliai_engine/image_caption")
    def blip_caption(captionReq: EliAIEngineImageCaptionAPI):
      response = requests.get(captionReq.image_url)

      img = response.content

      image_pil = Image.open(io.BytesIO(img))

      captions = generate_captions(image_pil)

      return captions
    
    @app.post("/eliai_engine/img_sam_prediction")
    def sam_prediction(samreq: EliAIEngineSAMPredictorAPI, user_id: str):
      image_base64 = samreq.image_base64 or ""

      print(f"START")
      # await asyncio.sleep(40)
      result = image_predictions(image_base64)
      # result = io.BytesIO()
      bytes = result.getvalue()
      result.close()

      print(f"END")
      return Response(bytes)
       

    @app.post("/eliai_engine/txt2img", status_code=204)
    def text2imgapi(txt2imgreq: EliAIEngineTxt2ImgProcessingAPI):

        try:
          loras = txt2imgreq.loras
          load_loras(loras)
          req = mapper.to(StableDiffusionTxt2ImgProcessingAPI).map(txt2imgreq)
          # return
          result = api.text2imgapi(req)

          controlnet_args = req.alwayson_scripts.get('controlnet', {}).get('args', [])
          controlnet_lenght = len(list(filter(is_tile_controlnet, controlnet_args)))

          if controlnet_lenght & controlnet_lenght > 0 :
            images = result.images[:-controlnet_lenght]
          else:
            images = result.images


          info = json.loads(result.info)
          seed = info.get('seed')
          # background_tasks.add_task(image_uploading, images, task_id, user_id)

          # Create a new thread to run the background task
          background_thread = threading.Thread(target=image_uploading, args=(images, seed, txt2imgreq.task_id, txt2imgreq.user_id))
          background_thread.start()

        except Exception as e:
          print(e)
          supabase.table("Tasks").update({
              "status": "failed",
              "finished_at": datetime.datetime.utcnow().isoformat()
          }).eq("task_id", txt2imgreq.task_id).execute()

          raise HTTPException(status_code=400, detail="Request failed") 

        return
    
    @app.post("/eliai_engine/img2img", status_code=204)
    def img2imgapi(img2imgreq: EliAIEngineImg2ImgProcessingAPI):
         

        try:
          loras = img2imgreq.loras  
          load_loras(loras)
          req = mapper.to(StableDiffusionImg2ImgProcessingAPI).map(img2imgreq)

          result = api.img2imgapi(req)

          controlnet_args = req.alwayson_scripts.get('controlnet', {}).get('args', [])
          controlnet_lenght = len(list(filter(is_tile_controlnet, controlnet_args)))

          if controlnet_lenght & controlnet_lenght > 0 :
            images = result.images[:-controlnet_lenght]
          else:
            images = result.images

          info = json.loads(result.info)
          seed = info.get('seed')
          # Create a new thread to run the background task
          background_thread = threading.Thread(target=image_uploading, args=(images, seed, img2imgreq.task_id, img2imgreq.user_id))
          background_thread.start()

        except Exception as e:
          print(e)
          supabase.table("Tasks").update({
              "status": "failed",
              "finished_at": datetime.datetime.utcnow().isoformat()
          }).eq("task_id", img2imgreq.task_id).execute()
          
          raise HTTPException(status_code=400, detail="Request failed") 

        return 
    
    @app.post("/eliai_engine/extra-single-image", status_code=204)
    def extra(extraReq: EliAIEngineExtraAPI):

        print(f"Tassk ID: {extraReq.task_id}")   

        try:
          req = mapper.to(ExtrasSingleImageRequest).map(extraReq)
          result = api.extras_single_image_api(req)
          images = [result.image]
          

          # Create a new thread to run the background task
          background_thread = threading.Thread(target=image_uploading, args=(images,1 , extraReq.task_id, extraReq.user_id))
          background_thread.start()

        except:
          supabase.table("Tasks").update({
              "status": "failed",
              "finished_at": datetime.datetime.utcnow().isoformat()
          }).eq("task_id", extraReq.task_id).execute()
          
          raise HTTPException(status_code=400, detail="Request failed") 

        return 

    if not cmd_opts.listen:
      runQueue(text2imgapi)



try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(eliai_engine_api)
except:
    pass
