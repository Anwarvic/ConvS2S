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