from utils import *
from dataloader import *
from transform import TextTransform



# load configurations
CONF = load_conf()

# add special characters to config
# get vocabulary
CONF["special_tokens"] = {
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

# load vocab
src_vocab = get_vocab( get_src_iter(train_data), src_tokenizer, CONF)
tgt_vocab = get_vocab( get_tgt_iter(train_data), tgt_tokenizer, CONF)

# data transforms
src_transform = TextTransform(src_tokenizer, src_vocab, CONF)
tgt_transform = TextTransform(tgt_tokenizer, tgt_vocab, CONF)

# get data loaders
train_dataloader = get_dataloader(train_data, src_transform, tgt_transform, CONF)
valid_dataloader = get_dataloader(valid_data, src_transform, tgt_transform, CONF)
test_dataloader = get_dataloader(test_data, src_transform, tgt_transform, CONF)

