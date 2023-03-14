import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import abc
import time_utils
import copy
import os
import argparse
from train_funcs import TRAIN_FUNC_DICT
from test_set_parser import test_set, populate_test_set

'''params:
    with_to_k - train to_k or not;
    with_augs - add augmented sentences or not;
    train_func - name of train function ("train_closed_form" or "baseline");
    train_func_ext - optional extension to add to train function name in saved folder. e.g. to differentiate different runs;
    save_dir - directory to save in;
    num_seeds - number of seeds to generate for each prompt;
    begin_idx - index to begin from (inclusive) in dataset;
    end_idx - index to end on (exclusive) in dataset;
    dataset - csv filename of dataset, defaults to "TIMED_test_set_filtered_SD14.csv";
    **train_kwargs - rest of command line args for train function [optional];

example run:
    python apply_time.py --with_to_k --train_func train_closed_form --save_dir results --begin_idx 0 --end_idx 104 --num_seeds 1
'''
## set up argparser
parser = argparse.ArgumentParser()
parser.add_argument('--with_to_k', default=False, action='store_true')
parser.add_argument('--with_augs', default=False, action='store_true')
parser.add_argument('--train_func', type=str)
parser.add_argument('--train_func_ext', type=str, default="", required=False)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--num_seeds', type=int)
parser.add_argument('--begin_idx', type=int)
parser.add_argument('--end_idx', type=int)
parser.add_argument('--dataset', type=str, default="TIMED_test_set_filtered_SD14.csv", required=False)
args, unknown = parser.parse_known_args()
print("SCRIPT ARGUMENTS:")
print(args)
print("---")

for arg in unknown:
    if arg.startswith(("-", "--")):
        parser.add_argument(arg.split('=')[0])
args = parser.parse_args()

## get arguments for our script
with_to_k = args.with_to_k          
with_augs = args.with_augs          
train_func = args.train_func        
train_func_ext = args.train_func_ext
save_dir = args.save_dir 
num_seeds = int(args.num_seeds)  
begin_idx = int(args.begin_idx)  
end_idx = int(args.end_idx)     
dataset = args.dataset    

## get remainder in train_kwargs
train_kwargs = vars(args)
train_kwargs.pop('with_to_k', None)
train_kwargs.pop('with_augs', None)
train_kwargs.pop('train_func', None)
train_kwargs.pop('train_func_ext', None)
train_kwargs.pop('save_dir', None)
train_kwargs.pop('num_seeds', None)
train_kwargs.pop('begin_idx', None)
train_kwargs.pop('end_idx', None)
train_kwargs.pop('dataset', None)
print("TRAIN_KWARGS:")
print(train_kwargs)
print("---")

### load test set
populate_test_set(begin_idx, end_idx, dataset_fname=dataset)

### load model
LOW_RESOURCE = True 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
tokenizer = ldm_stable.tokenizer

### get layers
ca_layers = []
def append_ca(net_):
    if net_.__class__.__name__ == 'CrossAttention':
        ca_layers.append(net_)
    elif hasattr(net_, 'children'):
        for net__ in net_.children():
            append_ca(net__)

sub_nets = ldm_stable.unet.named_children()
for net in sub_nets:
        if "down" in net[0]:
            append_ca(net[1])
        elif "up" in net[0]:
            append_ca(net[1])
        elif "mid" in net[0]:
            append_ca(net[1])

### get projection matrices
ca_clip_layers = [l for l in ca_layers if l.to_v.in_features == 768]
projection_matrices = [l.to_v for l in ca_clip_layers]
og_matrices = [copy.deepcopy(l.to_v) for l in ca_clip_layers]
if with_to_k:
    projection_matrices = projection_matrices + [l.to_k for l in ca_clip_layers]
    og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_clip_layers]

### print number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
params = 0
for l in ca_clip_layers:
    params += l.to_v.in_features * l.to_v.out_features
    if with_to_k:
        params += l.to_k.in_features * l.to_k.out_features
print("Params: ", params)
print("Total params: ", count_parameters(ldm_stable.unet))
print("Percentage: ", (params / count_parameters(ldm_stable.unet)) * 100)

### test set
print("Test set size: ", len(test_set))

### iterate over test set
for curr_item in test_set:
    print("CURRENT TEST SENTENCE: ", curr_item["old"])
    
    #### restart LDM parameters
    num_ca_clip_layers = len(ca_clip_layers)
    for idx_, l in enumerate(ca_clip_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k
    
    #### set up sentences
    old_texts = [curr_item["old"]]
    new_texts = [curr_item["new"]]
    if with_augs:
        base = old_texts[0] if old_texts[0][0:1] != "A" else "a" + old_texts[0][1:]
        old_texts.append("A photo of " + base)
        old_texts.append("An image of " + base)
        old_texts.append("A picture of " + base)
        base = new_texts[0] if new_texts[0][0:1] != "A" else "a" + new_texts[0][1:]
        new_texts.append("A photo of " + base)
        new_texts.append("An image of " + base)
        new_texts.append("A picture of " + base)
    
    #### prepare input k* and v*
    old_embs, new_embs = [], []
    for old_text, new_text in zip(old_texts, new_texts):
        text_input = ldm_stable.tokenizer(
            [old_text, new_text],
            padding="max_length",
            max_length=ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
        old_emb, new_emb = text_embeddings
        old_embs.append(old_emb)
        new_embs.append(new_emb)
    
    #### indetify corresponding destinations for each token in old_emb
    idxs_replaces = []
    for old_text, new_text in zip(old_texts, new_texts):
        tokens_a = tokenizer(old_text).input_ids
        tokens_b = tokenizer(new_text).input_ids
        tokens_a = [tokenizer.encode("a ")[1] if tokenizer.decode(t) == 'an' else t for t in tokens_a]
        tokens_b = [tokenizer.encode("a ")[1] if tokenizer.decode(t) == 'an' else t for t in tokens_b]
        num_orig_tokens = len(tokens_a)
        num_new_tokens = len(tokens_b)
        idxs_replace = []
        j = 0
        for i in range(num_orig_tokens):
            curr_token = tokens_a[i]
            while tokens_b[j] != curr_token:
                j += 1
            idxs_replace.append(j)
            j += 1
        while j < 77:
            idxs_replace.append(j)
            j += 1
        while len(idxs_replace) < 77:
            idxs_replace.append(76)
        idxs_replaces.append(idxs_replace)
    
    #### prepare batch: for each pair of setences, old context and new values
    contexts, valuess = [], []
    for old_emb, new_emb, idxs_replace in zip(old_embs, new_embs, idxs_replaces):
        context = old_emb.detach()
        values = []
        with torch.no_grad():
            for layer in projection_matrices:
                values.append(layer(new_emb[idxs_replace]).detach())
        contexts.append(context)
        valuess.append(values)
    
    #### define training function
    train = TRAIN_FUNC_DICT[train_func]
    
    #### train the model
    train(ldm_stable, projection_matrices, og_matrices, contexts, valuess, old_texts, new_texts, **train_kwargs)
    
    #### set up testing
    #saves in "./{s_dir}/{train_f}/{base_prompt}/{category}/{prompt}/seed_{seed}.png"
    def run_and_save(prompt, s_dir, train_f, category, base_prompt, seed):
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)
        images = time_utils.text2image_ldm_stable(ldm_stable, [prompt], latent=None, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=g, low_resource=LOW_RESOURCE)
        os.makedirs(f"./{s_dir}/{train_f}/{base_prompt}/{category}/{prompt}/", exist_ok = True) 
        time_utils.view_images(images).save(f"./{s_dir}/{train_f}/{base_prompt}/{category}/{prompt}/seed_{str(seed)}.png")
    
    #### run the testing
    kwargs_str = ""
    for key in train_kwargs:
        kwargs_str += "__" + key + "_" + train_kwargs[key]
    train_f_name = train_func + train_func_ext + kwargs_str
    if with_to_k: train_f_name+= "_with_to_k"
    if with_augs: train_f_name+= "_with_augs"
    base_prmpt = curr_item["old"] + "_" + curr_item["new"]
    for seed_ in range(num_seeds):
        run_and_save(curr_item["old"], save_dir, train_f_name, "base", base_prmpt, seed_)
        if train_func == "baseline": run_and_save(curr_item["new"], save_dir, train_f_name, "base", base_prmpt, seed_)
        for positive_item in curr_item["positives"]:
            run_and_save(positive_item["test"], save_dir, train_f_name, "positives", base_prmpt, seed_)
            if train_func == "baseline" and positive_item["gt"] is not None: run_and_save(positive_item["gt"], save_dir, train_f_name, "positives", base_prmpt, seed_)
        for negative_item in curr_item["negatives"]:
            run_and_save(negative_item["test"], save_dir, train_f_name, "negatives", base_prmpt, seed_)
            if train_func == "baseline" and negative_item["gt"] is not None: run_and_save(negative_item["gt"], save_dir, train_f_name, "negatives", base_prmpt, seed_)
