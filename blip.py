import gradio as gr
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipForConditionalGeneration, AutoModelForCausalLM, AutoImageProcessor, VisionEncoderDecoderModel, AutoTokenizer

# from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, Blip2ForConditionalGeneration, VisionEncoderDecoderModel
import torch
import open_clip

from huggingface_hub import hf_hub_download

torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')
torch.hub.download_url_to_file('https://huggingface.co/datasets/nielsr/textcaps-sample/resolve/main/stop_sign.png', 'stop_sign.png')
torch.hub.download_url_to_file('https://cdn.openai.com/dall-e-2/demos/text2im/astronaut/horse/photo/0.jpg', 'astronaut.jpg')

# git_processor_base = AutoProcessor.from_pretrained("microsoft/git-base-coco")
# git_model_base = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

# git_processor_large_coco = AutoProcessor.from_pretrained("microsoft/git-large-coco")
# git_model_large_coco = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")

# git_processor_large_textcaps = AutoProcessor.from_pretrained("microsoft/git-large-r-textcaps")
# git_model_large_textcaps = AutoModelForCausalLM.from_pretrained("microsoft/git-large-r-textcaps")

# blip_processor_base = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model_base = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

blip_processor_large = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model_large = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


# blip2_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

# blip2_processor_8_bit = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b")
# blip2_model_8_bit = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b", device_map="auto", load_in_8bit=True)

# vitgpt_processor = AutoImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# vitgpt_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# vitgpt_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# coca_model, _, coca_transform = open_clip.create_model_and_transforms(
#   model_name="coca_ViT-L-14",
#   pretrained="mscoco_finetuned_laion2B-s13B-b90k"
# )

device = "cuda" if torch.cuda.is_available() else "cpu"

# git_model_base.to(device)
# blip_model_base.to(device)
# git_model_large_coco.to(device)
# git_model_large_textcaps.to(device)
blip_model_large.to(device)
# vitgpt_model.to(device)
# coca_model.to(device)
# blip2_model.to(device)

def generate_caption(processor, model, image, tokenizer=None, use_float_16=False):
    inputs = processor(images=image,text="A picture of", return_tensors="pt").to(device)

    if use_float_16:
        inputs = inputs.to(torch.float16)
    
    generated_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=1.0,
        length_penalty=1.0,
        repetition_penalty=1.5,
        max_length=50,
        min_length=20,
        num_beams=5,
        top_p=0.9,)

    if tokenizer is not None:
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_caption)
    modified_caption = generated_caption.replace("a picture of ", "")
    print(modified_caption)
    return modified_caption


def generate_caption_coca(model, transform, image):
    im = transform(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model.generate(im, seq_len=20)
    return open_clip.decode(generated[0].detach()).split("<end_of_text>")[0].replace("<start_of_text>", "")


def generate_captions(image):
    # caption_git_base = generate_caption(git_processor_base, git_model_base, image)

    # caption_git_large_coco = generate_caption(git_processor_large_coco, git_model_large_coco, image)

    # caption_git_large_textcaps = generate_caption(git_processor_large_textcaps, git_model_large_textcaps, image)

    # caption_blip_base = generate_caption(blip_processor_base, blip_model_base, image)

    caption_blip_large = generate_caption(blip_processor_large, blip_model_large, image)

    # caption_vitgpt = generate_caption(vitgpt_processor, vitgpt_model, image, vitgpt_tokenizer)

    # caption_coca = generate_caption_coca(coca_model, coca_transform, image)

    # caption_blip2 = generate_caption(blip2_processor, blip2_model, image, use_float_16=True).strip()

    # caption_blip2_8_bit = generate_caption(blip2_processor_8_bit, blip2_model_8_bit, image, use_float_16=True).strip()

    # return caption_git_large_coco, caption_git_large_textcaps, caption_blip_large, caption_coca, caption_blip2_8_bit
    return caption_blip_large


   
examples = [["cats.jpg"], ["stop_sign.png"], ["astronaut.jpg"]]
# outputs = [gr.outputs.Textbox(label="Caption generated by GIT-large fine-tuned on COCO"), gr.outputs.Textbox(label="Caption generated by GIT-large fine-tuned on TextCaps"), gr.outputs.Textbox(label="Caption generated by BLIP-large"), gr.outputs.Textbox(label="Caption generated by CoCa"), gr.outputs.Textbox(label="Caption generated by BLIP-2 OPT 6.7b")] 
outputs = [
    # gr.outputs.Textbox(label="Caption generated by GIT-base fine-tuned on COCO"), 
           # gr.outputs.Textbox(label="Caption generated by GIT-large fine-tuned on COCO"),
           # gr.outputs.Textbox(label="Caption generated by GIT-large fine-tuned on TextCaps"),
           # gr.outputs.Textbox(label="Caption generated by BLIP-base"),
           gr.outputs.Textbox(label="Caption generated by BLIP-large"),
           # gr.outputs.Textbox(label="Caption generated by vitgpt")
          ] 

title = "Interactive demo: blip-large"
description = "Gradio Demo to compare GIT, BLIP, CoCa, and BLIP-2, 4 state-of-the-art vision+language models. To use it, simply upload your image and click 'submit', or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://huggingface.co/docs/transformers/main/model_doc/blip' target='_blank'>BLIP docs</a> | <a href='https://huggingface.co/docs/transformers/main/model_doc/git' target='_blank'>GIT docs</a></p>"

# interface = gr.Interface(fn=generate_captions, 
#                          inputs=gr.inputs.Image(type="pil"),
#                          outputs=outputs,
#                          examples=examples, 
#                          title=title,
#                          description=description,
#                          article=article, 
#                          enable_queue=True)
# interface.launch()