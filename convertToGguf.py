import os
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer

def merge_to_gguf(dir, out):
    m = {}
    for f in os.listdir(dir):
        if f.endswith('.safetensors'):
            p = os.path.join(dir, f)
            sd = load_file(p)
            m.update(sd)
    temp = 'merged_model.safetensors'
    save_file(m, temp)
    convert_to_gguf(temp, out)

def convert_to_gguf(s, g):
    model = AutoModel.from_pretrained("path/to/your/model")  
    model.load_state_dict(torch.load(s))  # Load the merged state dict
    model.save_pretrained(g)  # Save in GGUF format

def main():
    dir = "path/to/your/safetensors"
    out = "merged_model.gguf"
    merge_to_gguf(dir, out)

if __name__ == '__main__':
    main()
