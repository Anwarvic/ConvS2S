from utils import *
from dataloader import *

# load configurations
CONF = load_conf()

# add special characters to config
# get vocabulary
CONF["special_symbols"] = {
    CONF["unk_token"]: 0,
    CONF["pad_token"]: 1,
    CONF["sos_token"]: 2,
    CONF["eos_token"]: 3,
}
# set seed
set_seed(CONF['seed'])

# load tokenizers
src_tokenizer = load_tokenizer(CONF['tokenizer'], CONF['src'])
tgt_tokenizer = load_tokenizer(CONF['tokenizer'], CONF['tgt'])

# load datset
train_data, valid_data, test_data = load_multi30k(CONF['src'], CONF['tgt'])

tgt_vocab = get_vocab( get_tgt_iter(train_data), tgt_tokenizer, CONF)
src_vocab = get_vocab( get_src_iter(train_data), src_tokenizer, CONF)

print(len(src_vocab), len(tgt_vocab))