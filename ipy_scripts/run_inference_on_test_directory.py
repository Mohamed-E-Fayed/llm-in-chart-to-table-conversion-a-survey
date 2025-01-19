from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch


import time
from pathlib import Path

#from my_helping_functions import *

def infere_directory_with_generativeai_api(dir, model='gemini-1.5-flash', output_dir_suffix='initial_prompt'):
    for imgpath in tqdm(list(find_images(dir))):
        tmp = imgpath.parent / imgpath.stem
        output_file = str(tmp).replace('images', f'predictions/{model}/{output_dir_suffix}') + '.json'
        response = convert_chart_to_table_end_to_end(imgpath)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(response, dict):
           write_json_file(response, output_file)
        elif isinstance(response, str):
            write_file(output_file.replace('.json', '.txt'), response)
        time.sleep(0.5)

def infere_image_with_chartgemma(
        image_path, prompt=None,
        model=None, processor=None,
        ):
    if not prompt:
        prompt = "As an expert in visualizations, analyze the following graph and convert it  to table."
    image = Image.open(image_path).convert('RGB')
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    prompt_length = inputs['input_ids'].shape[1]
    inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}
    generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512)
    output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text

def infere_directory_with_chartgemma(dir, model_name='ahmed-masry/chartgemma', output_dir_suffix='initial_prompt', model=None, processor=None, device=None):
    if not model:
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_name).to(get_device(device))
    if not processor:
        processor = AutoProcessor.from_pretrained(model_name)
    for imgpath in tqdm(list(find_images(dir))):
        tmp = imgpath.parent / imgpath.stem
        output_file = str(tmp).replace('images', f'predictions/{model_name}/{output_dir_suffix}') + '.md'
        if Path(output_file).exists():
            continue
        response = infere_image_with_chartgemma(
                imgpath, model=model, processor=processor,
                )
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        write_file(output_file, response)
