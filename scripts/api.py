import datetime
import threading
import time
import asyncio
import os
from typing import List

from supabase import Client, create_client
import boto3

import numpy as np
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.exceptions import HTTPException
from PIL import Image

import gradio as gr

from modules.api.models import *
from modules.api import api, models
from modules import scripts

from modules.api.api import Api
from modules.call_queue import queue_lock

from scripts.models import EliAIEngineTxt2ImgProcessingAPI


import base64
import io


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


s3client = boto3.client('s3', endpoint_url='https://hn.ss.bfcplatform.vn',
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY'))
bucket_name = "eliai-server"
server_domain = "https://eliai-server.hn.ss.bfcplatform.vn/"

def s3Storage_base64_upload(base64_image: str, task_id: str, index: int):
    image_binary = base64.b64decode(base64_image)
    object_key = f"images/{task_id}/{task_id}_{index}.png"

    s3client.upload_fileobj(
      Fileobj=io.BytesIO(image_binary),
      Bucket=bucket_name,
      Key=object_key,
      ExtraArgs={'ACL': 'public-read'}  # Optional: Set ACL to make the image public
    )

    return server_domain + object_key

url: str = os.environ.get('SUPABASE_ENDPOINT') or "http://localhost:54321"
key: str = os.environ.get('SUPABASE_KEY') or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0"
supabase: Client = create_client(url, key)

def image_uploading(images: List[str], task_id: str, user_id: str):
    # time.sleep(10)
    result = []
    for index, image in enumerate(images):
        image_url = s3Storage_base64_upload(image, task_id, index)
        result.append(image_url)
        supabase.table("Images").insert({
            "image_url": image_url,
            "is_shared": True,
            "seed": 1,
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
    
 


def eliai_engine_api(_: gr.Blocks, app: FastAPI):
    api = Api(app, queue_lock)

    

    @app.post("/eliai_engine/txt2img", status_code=204)
    def text2imgapi(txt2imgreq: StableDiffusionTxt2ImgProcessingAPI, task_id: str, user_id: str):
        # task_id = txt2imgreq.task_id

        print(f"Tassk ID: {task_id}")   

        try:
          result = api.text2imgapi(txt2imgreq)

          controlnet_args = txt2imgreq.alwayson_scripts.get('controlnet', {}).get('args', {})
          controlnet_lenght = len(controlnet_args)

          if controlnet_lenght & controlnet_lenght > 0 :
            images = result.images[:-controlnet_lenght]
          else:
            images = result.images

          # background_tasks.add_task(image_uploading, images, task_id, user_id)

          # Create a new thread to run the background task
          background_thread = threading.Thread(target=image_uploading, args=(images, task_id, user_id))
          background_thread.start()

        except:
          supabase.table("Tasks").update({
              "status": "failed",
              "finished_at": datetime.datetime.utcnow().isoformat()
          }).eq("task_id", task_id).execute()

          raise HTTPException(status_code=400, detail="Request failed") 

        return
    
    @app.post("/eliai_engine/img2img", status_code=204)
    def text2imgapi(txt2imgreq: StableDiffusionImg2ImgProcessingAPI, task_id: str, user_id: str, background_tasks: BackgroundTasks):
        # task_id = txt2imgreq.task_id

        print(f"Tassk ID: {task_id}")   

        try:
          result = api.img2imgapi(txt2imgreq)

          controlnet_args = txt2imgreq.alwayson_scripts.get('controlnet', {}).get('args', {})
          controlnet_lenght = len(controlnet_args)

          if controlnet_lenght & controlnet_lenght > 0 :
            images = result.images[:-controlnet_lenght]
          else:
            images = result.images


          # Create a new thread to run the background task
          background_thread = threading.Thread(target=image_uploading, args=(images, task_id, user_id))
          background_thread.start()

        except:
          supabase.table("Tasks").update({
              "status": "failed",
              "finished_at": datetime.datetime.utcnow().isoformat()
          }).eq("task_id", task_id).execute()
          
          raise HTTPException(status_code=400, detail="Request failed") 

        return 
    
    @app.post("/eliai_engine/extra-single-image", status_code=204)
    def text2imgapi(extraReq: ExtrasSingleImageRequest, task_id: str, user_id: str, background_tasks: BackgroundTasks):
        # task_id = txt2imgreq.task_id

        print(f"Tassk ID: {task_id}")   

        try:
          result = api.extras_single_image_api(extraReq)
          images = [result.image]
          

          # Create a new thread to run the background task
          background_thread = threading.Thread(target=image_uploading, args=(images, task_id, user_id))
          background_thread.start()

        except:
          supabase.table("Tasks").update({
              "status": "failed",
              "finished_at": datetime.datetime.utcnow().isoformat()
          }).eq("task_id", task_id).execute()
          
          raise HTTPException(status_code=400, detail="Request failed") 

        return 




try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(eliai_engine_api)
except:
    pass
