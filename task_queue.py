import datetime
import json
import threading
import time
import redis
from modules.api.models import *
from modules.shared_cmd_options import cmd_opts
from automapper import mapper

import requests
import base64

from supabase_client import supabase
import os

machine_id = os.environ.get('MACHINE_ID')

#Routine that processes whatever you want as background
def YourLedRoutine(text2imgapi):
    redis_uri = 'rediss://default:AVNS_p5SxXC8sjRJE8JkNqB9@task-queue-minhnhatdo0709-a715.a.aivencloud.com:17468'
    time.sleep(30)
    while 1:
        task = None
        try:
            machine_status = supabase.table("Nodes").select("*").eq("machine_id", machine_id).execute()
            if machine_status.get('status', '') == 'shutdown':
                break

            redis_client = redis.from_url(redis_uri, decode_responses=True)
            task = redis_client.rpop('taskQueue')
            redis_client.close()

            if not task:
                continue
            task = json.loads(task)

            supabase.table("Tasks").update({
                "status": "processing",
            }).eq("task_id", task['task_id']).execute()
            print(f"lets go")
            upscaler = task.get('upscaler_1', False)
            if not upscaler:
                image = task.get('image')
                mask = task.get('mask')
                
                
                controlnets = task.get('alwayson_scripts', {}).get('controlNet', {}).get('args', [])
                if len(controlnets) > 0:
                  for control in controlnets:
                      control['image'] = downloadImage(control['image'])
                  task['alwayson_scripts']['controlNet']['args'] = controlnets


                task['override_settings'] = {
                    'sd_model_checkpoint': task['checkpoint']
                }

                task['prompt'] = prompt_builder(task['prompt'], task['loras']) 
                if image:
                    if image:
                        task['image'] = downloadImage(image)
                        task['init_images'] = [task['image']]
                    if mask:
                        task['mask'] = downloadImage(mask)
                    # req = mapper.to(StableDiffusionImg2ImgProcessingAPI).map(task)
                    requests.post(f'http://127.0.0.1:{cmd_opts.port}/eliai_engine/img2img', json=task)
                else:
                    # req = mapper.to(StableDiffusionTxt2ImgProcessingAPI).map(task)
                    # print(f"req: {req.json()}")
                    # print(f"task: {task}")
                    # task = {'batch_count': 1, 'cfg_scale': 9, 'height': 768, 'negative_prompt': 'Disfigured, cartoon, blurry', 'prompt': 'a villa, sunlight, blue sky, relax, soft light, tree, flower\n8k, high resolution', 'sampler': 'Euler a', 'seed': 1, 'steps': 35, 'width': 960, 'alwayson_scripts': {}, 'task_id': 'c603ff98-47f2-4a30-8750-4b5fa4cf6a11', 'user_id': 8}
                    # text2imgapi(task)
                    requests.post(f'http://127.0.0.1:{cmd_opts.port}/eliai_engine/txt2img', json=task)

            else:
                if task['image']:
                    task['image'] = downloadImage(task['image'])
                # req = mapper.to(StableDiffusionImg2ImgProcessingAPI).map(task)
                requests.post(f'http://127.0.0.1:{cmd_opts.port}/eliai_engine/extra-single-image', json=task)
        except Exception as e:
            print(e)
            if task:
                supabase.table("Tasks").update({
                    "status": "failed",
                    "finished_at": datetime.datetime.utcnow().isoformat()
                }).eq("task_id", task['task_id']).execute()
            continue
        time.sleep(1)

def runQueue(text2imgapi):
    #Create a background thread
    t1 = threading.Thread(target=YourLedRoutine, args=(text2imgapi,))
    #Background thread will finish with the main program
    t1.daemon = True
    #Start YourLedRoutine() in a separate thread
    t1.start()



def prompt_builder(prompt, loras):
    lora_prompt = ""

    if loras and len(loras) > 0:
        for item in loras:
            lora_prompt += f" <lora:{item['name'].replace('.safetensors', '')}:{item['weight']}> "
        return prompt + lora_prompt
    else:
        return prompt


def downloadImage(url):
    response = requests.get(url)

    img = response.content
    base64_img = base64.b64encode(img).decode('utf-8')

    return base64_img