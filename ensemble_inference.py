import os, time
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
device = torch.device("cuda")

def load_model(model_id='meta-llama/Meta-Llama-3-8B-Instruct', load_in_4bit=True):
    bnb_config = None
    if load_in_4bit:
        if not torch.cuda.is_available():
            raise ValueError('Quantization with BitsAndBytes requires CUDA.')
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map='auto')
    return model, tokenizer

def create_prompt(example_text, example_label, review_text):
    example_text = example_text.replace('\n','')
    review_text  = review_text.replace('\n','')

    prompt  = ""
    prompt += "### Instruction\n"
    prompt += "You are a helpful assistant evaluating the review texts about the restaurant. Please evaluate the review text and assign an integer score ranging from 1 for the most negative comment to 5 for the most positive comment.\n"
    prompt += "\n"
    prompt += "### Review 1\n"
    prompt += f"User review: {example_text}\n"
    prompt += f"Output: {example_label}\n"
    prompt += "\n"
    prompt += "### Review 2\n"
    prompt += f"User review: {review_text}\n"
    prompt += f"Output: "
    return prompt

def single_inference(prompt, tokenizer, model, model_seed, model_temperature, model_num=0, device='cuda'):
    torch.manual_seed(model_seed)
    # Tokenization
    inputs  = tokenizer(prompt, return_tensors='pt').to(device)
    model = model.to(device)

    # Inference
    time1 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=True,
            temperature=model_temperature,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
            )
    time2 = time.time()
    pred = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)[-1]
    try:
        pred = int(pred)
    except:
        pass
    result = {
        f'time{model_num}': time2-time1,
        f'pred{model_num}': pred
    }
    return result

def ensemble_inference(prompt, tokenizer, model, model_seeds, model_temperature, device):
    num_models = len(model_seeds)
    result = {}
    for model_num, model_seed in enumerate(model_seeds):
        r = single_inference(
            model_num=model_num, 
            prompt=prompt, 
            tokenizer=tokenizer, 
            model=model, 
            model_seed=model_seed, 
            model_temperature=model_temperature, 
            device=device
            )
        result = result|r
    res = pd.DataFrame(result, index=[0])
    res['total_time'] = res.filter(like='time').sum(axis=1)
    res['pred'] = res.filter(like='pred').median(axis=1).astype(int)
    return res
