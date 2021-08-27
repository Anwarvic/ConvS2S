import os
import yaml
import torch
import random
import numpy as np
from torchtext.data import get_tokenizer

def load_conf(conf_path="conf.yaml"):
    with open(conf_path, 'r') as stream:
        return yaml.safe_load(stream)


def load_tokenizer(name, lang):
    if name == 'whitespace':
        return get_tokenizer(None) #split
    elif name in {"spacy", "moses", "toktok", "revtok", "subword"}:
        return get_tokenizer(name, lang)
    else:
        raise ValueError("Unknown tokenizer: {}".format(name))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_model(model):
    torch.save(model.state_dict(), os.path.join("checkpoint", 'bestmodel.pt'))


def load_model():
    return torch.load(os.path.join("checkpoint", 'bestmodel.pt'))
